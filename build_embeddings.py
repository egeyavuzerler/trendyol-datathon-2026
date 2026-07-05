import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # hafif, çok dilli, Türkçe dahil, hızlı CPU/MPS çıkarımı

t0 = time.time()

print(f"Model yükleniyor: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

print("items_features.csv okunuyor...")
items = pd.read_csv(f"{DATA_DIR}/items_features.csv", usecols=["item_id", "title"])
items["title"] = items["title"].fillna("")

print("terms.csv okunuyor...")
terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
terms["query"] = terms["query"].fillna("")

print(f"\n{len(items)} ürün başlığı encode ediliyor (biraz sürebilir)...")
item_embeddings = model.encode(
    items["title"].tolist(),
    batch_size=256,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,  # L2 normalize -> cosine similarity = dot product
)
print(f"Ürün embedding'leri tamam. shape={item_embeddings.shape}  ({time.time()-t0:.1f}s geçti)")

print(f"\n{len(terms)} arama terimi encode ediliyor...")
term_embeddings = model.encode(
    terms["query"].tolist(),
    batch_size=256,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
print(f"Terim embedding'leri tamam. shape={term_embeddings.shape}  ({time.time()-t0:.1f}s geçti)")

np.save(f"{DATA_DIR}/item_embeddings.npy", item_embeddings.astype(np.float32))
np.save(f"{DATA_DIR}/term_embeddings.npy", term_embeddings.astype(np.float32))

# Sıra referansı için item_id / term_id listelerini de kaydediyoruz (sonraki script'lerde hizalama için)
items[["item_id"]].to_csv(f"{DATA_DIR}/item_embedding_index.csv", index=False)
terms[["term_id"]].to_csv(f"{DATA_DIR}/term_embedding_index.csv", index=False)

print(f"\nKaydedildi: item_embeddings.npy, term_embeddings.npy (+ index csv'ler)")
print(f"Toplam süre: {time.time()-t0:.1f}s")
