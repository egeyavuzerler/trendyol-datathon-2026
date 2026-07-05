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

print("training_features_v2.csv okunuyor...")
df = pd.read_csv(f"{DATA_DIR}/training_features_v2.csv")

row_item_idx = df["item_id"].map(item_id_to_idx).values
row_term_idx = df["term_id"].map(term_id_to_idx).values

print("Semantic cosine similarity hesaplanıyor (dense dot product)...")
# Embedding'ler zaten L2-normalize edilmiş şekilde kaydedildi -> dot product = cosine similarity
Q = term_embeddings[row_term_idx]   # (n_rows, 384)
I = item_embeddings[row_item_idx]   # (n_rows, 384)
semantic_cosine = np.einsum("ij,ij->i", Q, I).astype(np.float32)

df["semantic_cosine"] = semantic_cosine

OUT_PATH = f"{DATA_DIR}/training_features_v3.csv"
df.to_csv(OUT_PATH, index=False)
print(f"\nKaydedildi: {OUT_PATH}  ({time.time()-t0:.1f}s geçti)")
print(df.shape)

print("\nLabel'a göre semantic_cosine ortalaması:")
print(df.groupby("label")["semantic_cosine"].mean())
