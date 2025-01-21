#!/usr/bin/env python3
# inference_on_custom_data.py

import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cdist

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

def encode_texts_with_sapbert(texts, batch_size=128):
    """
    Encodes a list of strings (texts) using SapBERT and returns a NumPy array of embeddings.
    """
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model.eval()  # put model in evaluation mode

    # If you have a GPU, uncomment these lines:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        batch_texts = texts[i : i + batch_size]
        toks = tokenizer.batch_encode_plus(
            batch_texts,
            padding="max_length",
            max_length=25,
            truncation=True,
            return_tensors="pt",
        )
        # If using GPU:
        # for k in toks:
        #     toks[k] = toks[k].to(device)

        with torch.no_grad():
            output = model(**toks)
            cls_rep = output.last_hidden_state[:, 0, :]  # [CLS] token
            # If on GPU:
            # cls_rep = cls_rep.cpu()
            all_embeddings.append(cls_rep.numpy())

    # Concatenate all embeddings into a single NumPy array
    return np.concatenate(all_embeddings, axis=0)

def main():
    # -------------------------------------------------------------------------
    # 1) LOAD YOUR CUSTOM JSONL FILE
    # -------------------------------------------------------------------------
    JSONL_PATH = "my_concepts.jsonl"  # <-- Change to your JSONL file path
    my_concepts = load_jsonl_concepts(JSONL_PATH)
    # my_concepts is a list of (text, cui), e.g. [("Neoplasm of abdomen", "C0000735"), ... ]

    print(f"Loaded {len(my_concepts)} (text, concept_id) pairs.")
    print("Example pairs:", my_concepts[:5])

    # For demonstration, we might limit to the first 100,000 to save time/memory
    my_concepts_100k = my_concepts[:100000]

    # Separate the texts and IDs
    all_names = [p[0] for p in my_concepts_100k]
    all_ids = [p[1] for p in my_concepts_100k]

    # -------------------------------------------------------------------------
    # 2) ENCODE ALL LABELS (TEXTS) WITH SAPBERT
    # -------------------------------------------------------------------------
    all_reps_emb = encode_texts_with_sapbert(all_names, batch_size=128)
    print("Embedding shape:", all_reps_emb.shape)

    # -------------------------------------------------------------------------
    # 3) ENCODE A QUERY AND FIND NEAREST NEIGHBOR
    # -------------------------------------------------------------------------
    query = "cardiopathy"
    print(f"\nQuery: {query}")

    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model.eval()

    # If you have GPU, you can move the model to CUDA for the query as well.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    query_toks = tokenizer.batch_encode_plus(
        [query],
        padding="max_length",
        max_length=25,
        truncation=True,
        return_tensors="pt"
    )
    # If on GPU: 
    # for k in query_toks:
    #     query_toks[k] = query_toks[k].to(device)

    with torch.no_grad():
        query_output = model(**query_toks)
        query_cls_rep = query_output.last_hidden_state[:, 0, :]
        # query_cls_rep = query_cls_rep.cpu()  # if GPU used

    # Convert to numpy
    query_emb = query_cls_rep.numpy()

    # -------------------------------------------------------------------------
    # 4) SIMPLE NEAREST NEIGHBOR SEARCH
    # -------------------------------------------------------------------------
    dist = cdist(query_emb, all_reps_emb, metric="euclidean")
    nn_index = np.argmin(dist)
    print("\n--- Nearest Neighbor Search ---")
    print("Nearest concept text:", all_names[nn_index])
    print("Nearest concept_id:", all_ids[nn_index])
    print("---------------------------------\n")

if __name__ == "__main__":
    main()
