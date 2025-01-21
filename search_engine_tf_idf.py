from rapidfuzz import fuzz
from typing import List, Dict, Any, Callable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import faiss
import numpy as np

class SearchEngine:
    def __init__(self, data: List[Dict[str, Any]], 
                 progress_callback: Callable[[float, str], None] = None,
                 n_components: int = 300):
        """
        Initialize the search engine with UMLS concept data.
        
        Parameters
        ----------
        data : List[Dict[str, Any]]
            The list of concept dictionaries with fields like 'canonical_name', 'aliases', etc.
        progress_callback : Callable[[float, str], None], optional
            A function to report progress. E.g., progress_callback(0.5, "Halfway done").
        n_components : int
            The number of latent dimensions to use in TruncatedSVD for dimensionality reduction.
        """
        self.data = data
        self.progress_callback = progress_callback
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            analyzer='word'
        )
        
        # TruncatedSVD for dimensionality reduction
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        
        # Create search index
        self._create_search_index()

    def _create_search_index(self):
        """Create search indices for faster lookups."""
        total_steps = 5  # We now have an extra step for SVD
        current_step = 0

        if self.progress_callback:
            self.progress_callback(current_step / total_steps, "Preparing text data...")

        # Prepare text for semantic search
        self.search_texts = []
        total_items = len(self.data)

        for idx, item in enumerate(self.data):
            # Combine canonical name and aliases for search
            text = f"{item['canonical_name']} {' '.join(item['aliases'])}"
            self.search_texts.append(text.lower())

            if self.progress_callback and idx % max(1, total_items // 100) == 0:
                progress = (idx / total_items) * 0.25  # First quarter of progress
                self.progress_callback(progress, "Processing text data...")

        current_step += 1
        if self.progress_callback:
            self.progress_callback(current_step / total_steps, "Creating TF-IDF vectors...")

        # Create TF-IDF vectors (sparse matrix)
        tfidf_matrix = self.vectorizer.fit_transform(self.search_texts)

        current_step += 1
        if self.progress_callback:
            self.progress_callback(current_step / total_steps, 
                                   f"Reducing dimensionality to {self.svd.n_components} components...")

        # Reduce dimensionality with TruncatedSVD (dense but much smaller)
        tfidf_reduced = self.svd.fit_transform(tfidf_matrix)
        self.tfidf_dense = tfidf_reduced.astype('float32')

        current_step += 1
        if self.progress_callback:
            self.progress_callback(current_step / total_steps, "Initializing FAISS index...")

        # Initialize FAISS index
        self.dimension = self.tfidf_dense.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)

        current_step += 1
        if self.progress_callback:
            self.progress_callback(current_step / total_steps, "Building search index...")

        # Normalize vectors for cosine similarity
        faiss.normalize_L2(self.tfidf_dense)
        self.index.add(self.tfidf_dense)

        if self.progress_callback:
            self.progress_callback(1.0, "Indexing complete!")


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

        # Sort with exact matches on top, followed by fuzzy matches in descending score
        return sorted(results, key=lambda x: (-x['score'], x['match_type'] != 'exact'))


    def semantic_search(self, query: str, threshold: float) -> List[Dict]:
        """Perform semantic similarity search using FAISS."""
        # Transform query to a sparse TF-IDF vector
        query_vector_sparse = self.vectorizer.transform([query.lower()])

        # Reduce dimensionality with the same SVD
        query_vector_reduced = self.svd.transform(query_vector_sparse).astype('float32')

        # Normalize
        faiss.normalize_L2(query_vector_reduced)

        # Perform similarity search
        k = len(self.data)  # Search all documents
        similarities, indices = self.index.search(query_vector_reduced, k)

        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if similarity >= threshold:
                results.append({
                    **self.data[idx],
                    'score': float(similarity),
                    'match_type': 'semantic'
                })

        return sorted(results, key=lambda x: x['score'], reverse=True)

    def search(self, query: str, search_type: str, fuzzy_threshold: float,
               semantic_threshold: float, max_results: int) -> List[Dict]:
        """Perform a hybrid search combining fuzzy and semantic matching."""
        if search_type == 'fuzzy':
            results = self.fuzzy_search(query, fuzzy_threshold)
        elif search_type == 'semantic':
            results = self.semantic_search(query, semantic_threshold)
        else:  # hybrid
            fuzzy_results = self.fuzzy_search(query, fuzzy_threshold)
            semantic_results = self.semantic_search(query, semantic_threshold)

            # Combine and deduplicate results
            seen = set()
            results = []

            for result in fuzzy_results + semantic_results:
                if result['concept_id'] not in seen:
                    seen.add(result['concept_id'])
                    results.append(result)

            # Sort by combined score
            results = sorted(results, key=lambda x: x['score'], reverse=True)

        return results[:max_results]
