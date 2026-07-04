import time
import os
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

DATA_DIR = "data"
VECTORIZER_PATH = f"{DATA_DIR}/tfidf_vectorizer.joblib"
ITEM_MATRIX_PATH = f"{DATA_DIR}/item_tfidf_matrix.npz"

t0 = time.time()

# ============================================================
# 1) items_features.csv oku
# ============================================================
print("items_features.csv okunuyor...")
items = pd.read_csv(
    f"{DATA_DIR}/items_features.csv",
    usecols=["item_id", "title", "category", "brand", "gender", "age_group",
             "raw_text", "attr_renk", "attr_color_detail"],
)
items["raw_text"] = items["raw_text"].fillna("")
items["title"] = items["title"].fillna("")
items["category"] = items["category"].fillna("")
items["brand"] = items["brand"].fillna("")

item_id_to_idx = {iid: i for i, iid in enumerate(items["item_id"].values)}

# ============================================================
# 2) TF-IDF: kaydedilmişse yükle, yoksa fit et ve kaydet
# ============================================================
if os.path.exists(VECTORIZER_PATH) and os.path.exists(ITEM_MATRIX_PATH):
    print("Kaydedilmiş TF-IDF vectorizer ve item_matrix yükleniyor...")
    vectorizer = joblib.load(VECTORIZER_PATH)
    item_matrix = sparse.load_npz(ITEM_MATRIX_PATH)
else:
    print("TF-IDF fit ediliyor (ilk kez, biraz sürebilir)...")
    vectorizer = TfidfVectorizer(max_features=200_000, min_df=2, max_df=0.3, sublinear_tf=True)
    item_matrix = vectorizer.fit_transform(items["raw_text"].values).tocsr()
    joblib.dump(vectorizer, VECTORIZER_PATH)
    sparse.save_npz(ITEM_MATRIX_PATH, item_matrix)
    print("Kaydedildi:", VECTORIZER_PATH, ITEM_MATRIX_PATH)

print("item_matrix shape:", item_matrix.shape, f"({time.time()-t0:.1f}s geçti)")

# ============================================================
# 3) training_full.csv + terms.csv + items birleştir
# ============================================================
print("training_full.csv ve terms.csv okunuyor...")
train_full = pd.read_csv(f"{DATA_DIR}/training_full.csv")
terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
terms["query"] = terms["query"].fillna("")

print("Birleştirme yapılıyor (merge)...")
merged = train_full.merge(terms, on="term_id", how="left")
merged = merged.merge(items, on="item_id", how="left")
merged["query"] = merged["query"].fillna("")
merged["title"] = merged["title"].fillna("")
merged["category"] = merged["category"].fillna("")
merged["brand"] = merged["brand"].fillna("")
merged["attr_renk"] = merged["attr_renk"].fillna("")
merged["attr_color_detail"] = merged["attr_color_detail"].fillna("")
print("merged shape:", merged.shape, f"({time.time()-t0:.1f}s geçti)")

# ============================================================
# 4) TF-IDF cosine similarity (query x item), vektörel işlem
# ============================================================
print("TF-IDF cosine similarity hesaplanıyor...")

# Query tarafı: sadece unique query'leri transform et (verimlilik için)
unique_terms = merged[["term_id", "query"]].drop_duplicates(subset="term_id").reset_index(drop=True)
unique_query_matrix = vectorizer.transform(unique_terms["query"].values).tocsr()
term_id_to_qidx = {tid: i for i, tid in enumerate(unique_terms["term_id"].values)}

row_qidx = merged["term_id"].map(term_id_to_qidx).values
row_iidx = merged["item_id"].map(item_id_to_idx).values

Q = unique_query_matrix[row_qidx]      # (n_rows x vocab) sparse
I = item_matrix[row_iidx]              # (n_rows x vocab) sparse

cosine_sim = np.asarray(Q.multiply(I).sum(axis=1)).ravel()
print(f"cosine_sim hesaplandı ({time.time()-t0:.1f}s geçti)")

# ============================================================
# 5) Basit metin/kategori eşleşme feature'ları (satır satır, hızlı)
# ============================================================
print("Metin/kategori eşleşme feature'ları hesaplanıyor...")

GENDER_KEYWORDS = {
    "kadın": "kadın", "kadin": "kadın", "bayan": "kadın",
    "erkek": "erkek",
    "unisex": "unisex",
    "çocuk": "çocuk", "cocuk": "çocuk",
    "bebek": "bebek",
}

query_arr = merged["query"].str.lower().values
title_arr = merged["title"].str.lower().values
category_arr = merged["category"].str.lower().values
brand_arr = merged["brand"].str.lower().values
gender_arr = merged["gender"].fillna("unknown").str.lower().values
age_group_arr = merged["age_group"].fillna("unknown").str.lower().values
attr_renk_arr = merged["attr_renk"].str.lower().values
attr_color_detail_arr = merged["attr_color_detail"].str.lower().values

n_rows = len(merged)
query_word_count = np.zeros(n_rows, dtype=np.int32)
title_word_count = np.zeros(n_rows, dtype=np.int32)
word_overlap_count = np.zeros(n_rows, dtype=np.int32)
word_overlap_ratio = np.zeros(n_rows, dtype=np.float32)
exact_substring_match = np.zeros(n_rows, dtype=np.int8)
brand_in_query = np.zeros(n_rows, dtype=np.int8)
color_match = np.zeros(n_rows, dtype=np.int8)
category_token_overlap_ratio = np.zeros(n_rows, dtype=np.float32)
gender_match = np.zeros(n_rows, dtype=np.int8)   # query'de cinsiyet geçiyor ve ürünle uyumlu
gender_conflict = np.zeros(n_rows, dtype=np.int8)  # query'de cinsiyet geçiyor ama ürünle çelişiyor

for i in range(n_rows):
    q = query_arr[i]
    t = title_arr[i]
    cat = category_arr[i]
    brand = brand_arr[i]
    renk = attr_renk_arr[i]
    color_detail = attr_color_detail_arr[i]
    gender = gender_arr[i]
    age_group = age_group_arr[i]

    q_tokens = set(q.split())
    t_tokens = set(t.split())
    cat_tokens = set(cat.replace("/", " ").split())

    qwc = len(q_tokens)
    query_word_count[i] = qwc
    title_word_count[i] = len(t_tokens)

    overlap = len(q_tokens & t_tokens)
    word_overlap_count[i] = overlap
    word_overlap_ratio[i] = overlap / qwc if qwc > 0 else 0.0

    exact_substring_match[i] = 1 if (q and q in t) else 0
    brand_in_query[i] = 1 if (brand and brand in q) else 0

    color_hit = 0
    if renk and renk in q:
        color_hit = 1
    if color_detail and any(part.strip() in q for part in color_detail.split("-") if part.strip()):
        color_hit = 1
    color_match[i] = color_hit

    if cat_tokens:
        category_token_overlap_ratio[i] = len(q_tokens & cat_tokens) / len(cat_tokens)

    # Cinsiyet sinyali: query'de geçen cinsiyet kelimesi var mı, ürünle uyumlu mu
    q_gender = None
    for kw, norm in GENDER_KEYWORDS.items():
        if kw in q:
            q_gender = norm
            break

    if q_gender is not None:
        item_gender_norm = gender if gender in ("kadın", "erkek", "unisex") else None
        item_age_norm = "çocuk" if age_group in ("çocuk", "bebek & çocuk") else ("bebek" if age_group == "bebek" else None)

        if q_gender in ("çocuk", "bebek"):
            if item_age_norm == q_gender or item_gender_norm == "unisex":
                gender_match[i] = 1
            elif item_age_norm is not None and item_age_norm != q_gender:
                gender_conflict[i] = 1
        else:
            if item_gender_norm == q_gender or item_gender_norm == "unisex":
                gender_match[i] = 1
            elif item_gender_norm is not None and item_gender_norm != q_gender:
                gender_conflict[i] = 1

    if i % 200_000 == 0:
        print(f"İşlenen satır: {i}/{n_rows}", end="\r")

print(f"\nMetin feature'ları tamamlandı ({time.time()-t0:.1f}s geçti)")

# ============================================================
# 6) Final feature dataframe'i oluştur ve kaydet
# ============================================================
features = pd.DataFrame({
    "id": merged["id"].values,
    "term_id": merged["term_id"].values,
    "item_id": merged["item_id"].values,
    "label": merged["label"].values,
    "tfidf_cosine": cosine_sim,
    "query_word_count": query_word_count,
    "title_word_count": title_word_count,
    "word_overlap_count": word_overlap_count,
    "word_overlap_ratio": word_overlap_ratio,
    "exact_substring_match": exact_substring_match,
    "brand_in_query": brand_in_query,
    "color_match": color_match,
    "category_token_overlap_ratio": category_token_overlap_ratio,
    "gender_match": gender_match,
    "gender_conflict": gender_conflict,
})

OUT_PATH = f"{DATA_DIR}/training_features.csv"
features.to_csv(OUT_PATH, index=False)
print(f"\nKaydedildi: {OUT_PATH}  (toplam süre: {time.time()-t0:.1f}s)")
print(features.shape)
print(features.head())
print("\nLabel'a göre ortalama feature değerleri:")
print(features.groupby("label")[[
    "tfidf_cosine", "word_overlap_ratio", "exact_substring_match",
    "brand_in_query", "color_match", "category_token_overlap_ratio",
    "gender_match", "gender_conflict",
]].mean())
