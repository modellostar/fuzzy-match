import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from rapidfuzz import fuzz
from typing import List, Dict, Any, Callable
import json
from tqdm import tqdm
from typing import List, Dict, Any, Optional

class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.data_by_concept_id = {item["concept_id"]: item for item in self.data}

    def get_row_by_concept_id(self, concept_id):
        return self.data_by_concept_id.get(concept_id)


# -----------------------------------
# Global: Load SapBERT model once
# -----------------------------------
TOKENIZER = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR")
MODEL = AutoModel.from_pretrained("cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR")
MODEL.eval()


def load_jsonl_concepts(jsonl_path):
    """
    Reads a JSONL file and returns a list of (text, concept_id) pairs.
    Each line in the JSONL should have:
        "concept_id", "canonical_name", "aliases"
    """
    concepts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading JSONL"):
            data = json.loads(line.strip())
            cui = data["concept_id"]
            # Merge canonical_name with aliases
            aliases = set(data.get("aliases", []))
            aliases.add(data["canonical_name"])
            # Build list of (text, cui) pairs
            for text in aliases:
                concepts.append((text, cui))
    return concepts



class SearchEngine:
    def __init__(
        self,
        data: List[Dict[str, Any]],
        index_path: str,
        concept_alias_mapping_file:str,
        progress_callback: Callable[[float, str], None] = None
    ):
        """
        Initialize the search engine with UMLS concept data.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            The list of concept dictionaries with fields like:
            - 'concept_id'
            - 'canonical_name'
            - 'aliases' (list of strings)
            etc.
        index_path : str
            Path to the pre-trained (already-built) SapBERT FAISS index file,
            e.g. "sapbert_index.faiss".
        progress_callback : Callable[[float, str], None], optional
            A function to report progress. E.g., progress_callback(0.5, "Halfway done").
        """

        self.data = data
        self.progress_callback = progress_callback
        self.index_path = index_path

        # Keep track of concept_id, names, etc., in parallel arrays if needed:
        # (Assuming the pre-built index is in the exact same order as `data`.)
        self.all_ids = [item["concept_id"] for item in data]
        self.all_names = [item["canonical_name"] for item in data]

        my_concepts = load_jsonl_concepts(concept_alias_mapping_file)
        # For demonstration, we might limit to the first 100,000 to save time/memory
        my_concepts_100k = my_concepts

        # Separate the texts and IDs
        self.all_alias_name_idx = [p[0] for p in my_concepts_100k]
        self.all_alias_idx = [p[1] for p in my_concepts_100k]

        self.processor = DataProcessor(self.data)

        # Load the pre-trained FAISS index
        self._create_search_index()

    def _create_search_index(self):
        """Load the pre-built SapBERT-based FAISS index."""
        if self.progress_callback:
            self.progress_callback(0.0, f"Loading SapBERT FAISS index from {self.index_path}...")

        # Read the index from file
        self.index = faiss.read_index(self.index_path)

        if self.progress_callback:
            self.progress_callback(1.0, "SapBERT FAISS index loaded successfully.")

    def fuzzy_search(self, query: str, threshold: float) -> List[Dict]:
        """Perform fuzzy string matching."""
        results = []
        query_lower = query.lower()
        
        for idx, item in enumerate(self.data):
            canonical_name = item['canonical_name'].lower()
            aliases = [alias.lower() for alias in item['aliases']]
            
            # Check if the canonical_name is an exact match
            if canonical_name == query_lower:
                score = 1.0  # Maximum possible score for exact match
                match_type = 'exact'
            else:
                # Compute fuzzy match scores
                score = max(
                    [fuzz.ratio(query_lower, canonical_name) / 100] +
                    [fuzz.ratio(query_lower, alias) / 100 for alias in aliases]
                )
                match_type = 'fuzzy' if score >= threshold else None
            
            if match_type:
                results.append({
                    **item,
                    'score': score,
                    'match_type': match_type
                })

        #if results is empty, return empty list
        if len(results) == 0:
            return []

        # Sort with exact matches on top, followed by fuzzy matches in descending score
        sorted_list = sorted(results, key=lambda x: (-x['score'], x['match_type'] != 'exact'))
        return [sorted_list[0]]

    def semantic_search(self, query: str, threshold: float) -> List[Dict]:
        """
        Perform semantic similarity search using a SapBERT-based FAISS index.

        Parameters
        ----------
        query : str
            The user's query string.
        threshold : float
            Minimum similarity score to include a result (assuming the index uses inner product or similar).

        Returns
        -------
        List[Dict]
            A list of concept dictionaries with a 'score' key indicating the similarity.
        """

        # 1) Encode query using the globally loaded SapBERT tokenizer + model
        toks = TOKENIZER.batch_encode_plus(
            [query],
            padding='max_length',   # or 'longest' if you prefer
            max_length=25,
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            output = MODEL(**toks)
            # SapBERT typically uses the [CLS] token embedding as the representation
            cls_rep = output.last_hidden_state[:, 0, :]
        query_emb = cls_rep.cpu().numpy().astype('float32')

        # 2) Nearest Neighbor Search via FAISS
        k = 1 # or a smaller k if you only want top-K
        distances, indices = self.index.search(query_emb, k)
        best_idx = indices[0][0]

        #Distance must be below 0.000001 else return empty list
        if distances[0][0] > 0.000001:
            return []

        # 3) Collect results
        results = []
        result = self.processor.get_row_by_concept_id(self.all_alias_idx[best_idx])
        results.append({
            **result,
            'score': 1,  # convert np.float32 -> float
            'match_type': 'semantic'
        })

        # Sort descending by similarity score
        return sorted(results, key=lambda x: x['score'], reverse=False)

    def search(
        self,
        query: str,
        search_type: str,
        fuzzy_threshold: float,
        semantic_threshold: float,
        max_results: int
    ) -> List[Dict]:
        """
        Perform a fuzzy, semantic, or hybrid search.

        Parameters
        ----------
        query : str
            User's search text.
        search_type : str
            One of ["fuzzy", "semantic", "hybrid"].
        fuzzy_threshold : float
            Fuzzy match cutoff (0.0 to 1.0).
        semantic_threshold : float
            Semantic similarity cutoff (0.0 to 1.0) if your embeddings are normalized.
        max_results : int
            Maximum number of results to return.

        Returns
        -------
        List[Dict]
            A list of matched concept dicts, each containing at least:
            'concept_id', 'score', 'match_type', etc.
        """

        if search_type == 'fuzzy':
            results = self.fuzzy_search(query, fuzzy_threshold)
        elif search_type == 'semantic':
            results = self.semantic_search(query, semantic_threshold)
        else:  # 'hybrid'
            fuzzy_results = self.fuzzy_search(query, fuzzy_threshold)
            semantic_results = self.semantic_search(query, semantic_threshold)

            # Combine and deduplicate by concept_id
            seen = set()
            combined = []
            for item in fuzzy_results + semantic_results:
                if item['concept_id'] not in seen:
                    seen.add(item['concept_id'])
                    combined.append(item)

            # Sort by 'score' descending
            results = sorted(combined, key=lambda x: x['score'], reverse=True)

        return results[:max_results]
