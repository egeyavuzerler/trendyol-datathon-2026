import time
import numpy as np
import pandas as pd

DATA_DIR = "data"
t0 = time.time()

print("Embedding'ler ve index dosyaları yükleniyor...")
item_embeddings = np.load(f"{DATA_DIR}/item_embeddings.npy")
term_embeddings = np.load(f"{DATA_DIR}/term_embeddings.npy")
item_index = pd.read_csv(f"{DATA_DIR}/item_embedding_index.csv")
term_index = pd.read_csv(f"{DATA_DIR}/term_embedding_index.csv")

item_id_to_idx = {iid: i for i, iid in enumerate(item_index["item_id"].values)}
term_id_to_idx = {tid: i for i, tid in enumerate(term_index["term_id"].values)}

print("submission_pairs.csv (term_id/item_id için) ve submission_features_v2.csv okunuyor...")
submission_pairs = pd.read_csv(f"{DATA_DIR}/submission_pairs.csv")
features = pd.read_csv(f"{DATA_DIR}/submission_features_v2.csv")

merged = features.merge(submission_pairs[["id", "term_id", "item_id"]], on="id", how="left")

row_item_idx = merged["item_id"].map(item_id_to_idx).values
row_term_idx = merged["term_id"].map(term_id_to_idx).values

print("Semantic cosine similarity hesaplanıyor...")
Q = term_embeddings[row_term_idx]
I = item_embeddings[row_item_idx]
semantic_cosine = np.einsum("ij,ij->i", Q, I).astype(np.float32)

merged["semantic_cosine"] = semantic_cosine
merged = merged.drop(columns=["term_id", "item_id"])

OUT_PATH = f"{DATA_DIR}/submission_features_v3.csv"
merged.to_csv(OUT_PATH, index=False)
print(f"\nKaydedildi: {OUT_PATH}  ({time.time()-t0:.1f}s geçti)")
print(merged.shape)
