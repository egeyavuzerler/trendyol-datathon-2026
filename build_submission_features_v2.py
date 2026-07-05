import time
import numpy as np
import pandas as pd
from scipy import sparse
import joblib

DATA_DIR = "data"
t0 = time.time()

ATTR_MATCH_COLS = {
    "attr_materyal": "materyal_match",
    "attr_desen": "desen_match",
    "attr_kumas_tipi": "kumas_tipi_match",
    "attr_urun_tipi": "urun_tipi_match",
}


def row_wise_dot(query_vectors_full, item_vectors_full, row_qidx, row_iidx):
    Q = query_vectors_full[row_qidx]
    I = item_vectors_full[row_iidx]
    return np.asarray(Q.multiply(I).sum(axis=1)).ravel()


print("Kaydedilmiş TF-IDF/BM25 matrisleri yükleniyor...")
vec_raw = joblib.load(f"{DATA_DIR}/tfidf_vectorizer_raw.joblib")
mat_raw = sparse.load_npz(f"{DATA_DIR}/item_tfidf_matrix_raw.npz")
vec_title = joblib.load(f"{DATA_DIR}/tfidf_vectorizer_title.joblib")
mat_title = sparse.load_npz(f"{DATA_DIR}/item_tfidf_matrix_title.npz")
vec_category = joblib.load(f"{DATA_DIR}/tfidf_vectorizer_category.joblib")
mat_category = sparse.load_npz(f"{DATA_DIR}/item_tfidf_matrix_category.npz")
vec_attrs = joblib.load(f"{DATA_DIR}/tfidf_vectorizer_attrs.joblib")
mat_attrs = sparse.load_npz(f"{DATA_DIR}/item_tfidf_matrix_attrs.npz")
count_vec_raw = joblib.load(f"{DATA_DIR}/bm25_countvec_raw.joblib")
bm25_mat_raw = sparse.load_npz(f"{DATA_DIR}/bm25_matrix_raw.npz")

print("items_features.csv okunuyor...")
items = pd.read_csv(
    f"{DATA_DIR}/items_features.csv",
    usecols=["item_id", "title", "category", "brand", "gender", "age_group",
             "raw_text", "attributes_raw", "attr_renk", "attr_color_detail"] + list(ATTR_MATCH_COLS.keys()),
)
for col in ["raw_text", "title", "category", "brand", "attributes_raw", "attr_renk", "attr_color_detail"] + list(ATTR_MATCH_COLS.keys()):
    items[col] = items[col].fillna("")
items["category_text"] = items["category"].str.replace("/", " ", regex=False)
item_id_to_idx = {iid: i for i, iid in enumerate(items["item_id"].values)}

print("submission_pairs.csv ve terms.csv okunuyor...")
submission_pairs = pd.read_csv(f"{DATA_DIR}/submission_pairs.csv")
terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
terms["query"] = terms["query"].fillna("")

print("Birleştirme yapılıyor (merge)...")
merged = submission_pairs.merge(terms, on="term_id", how="left")
merged = merged.merge(items, on="item_id", how="left")
for col in ["query", "title", "category", "category_text", "brand", "attributes_raw", "attr_renk", "attr_color_detail"] + list(ATTR_MATCH_COLS.keys()):
    merged[col] = merged[col].fillna("")
print("merged shape:", merged.shape, f"({time.time()-t0:.1f}s geçti)")

print("Query'ler transform ediliyor...")
unique_terms = merged[["term_id", "query"]].drop_duplicates(subset="term_id").reset_index(drop=True)
term_id_to_qidx = {tid: i for i, tid in enumerate(unique_terms["term_id"].values)}
row_qidx = merged["term_id"].map(term_id_to_qidx).values
row_iidx = merged["item_id"].map(item_id_to_idx).values

q_raw = vec_raw.transform(unique_terms["query"].values).tocsr()
q_title = vec_title.transform(unique_terms["query"].values).tocsr()
q_category = vec_category.transform(unique_terms["query"].values).tocsr()
q_attrs = vec_attrs.transform(unique_terms["query"].values).tocsr()
q_bm25 = count_vec_raw.transform(unique_terms["query"].values).tocsr()

print("Benzerlik skorları hesaplanıyor...")
tfidf_cosine = row_wise_dot(q_raw, mat_raw, row_qidx, row_iidx)
tfidf_title_cosine = row_wise_dot(q_title, mat_title, row_qidx, row_iidx)
tfidf_category_cosine = row_wise_dot(q_category, mat_category, row_qidx, row_iidx)
tfidf_attributes_cosine = row_wise_dot(q_attrs, mat_attrs, row_qidx, row_iidx)
bm25_score = row_wise_dot(q_bm25, bm25_mat_raw, row_qidx, row_iidx)
print(f"Benzerlik skorları tamam ({time.time()-t0:.1f}s geçti)")

print("Metin/kategori/attribute eşleşme feature'ları hesaplanıyor...")

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
attr_extra_arrs = {col: merged[col].fillna("").str.lower().values for col in ATTR_MATCH_COLS.keys()}

n_rows = len(merged)
query_word_count = np.zeros(n_rows, dtype=np.int32)
title_word_count = np.zeros(n_rows, dtype=np.int32)
word_overlap_count = np.zeros(n_rows, dtype=np.int32)
word_overlap_ratio = np.zeros(n_rows, dtype=np.float32)
exact_substring_match = np.zeros(n_rows, dtype=np.int8)
brand_in_query = np.zeros(n_rows, dtype=np.int8)
color_match = np.zeros(n_rows, dtype=np.int8)
category_token_overlap_ratio = np.zeros(n_rows, dtype=np.float32)
gender_match = np.zeros(n_rows, dtype=np.int8)
gender_conflict = np.zeros(n_rows, dtype=np.int8)
attr_match_arrays = {feat_name: np.zeros(n_rows, dtype=np.int8) for feat_name in ATTR_MATCH_COLS.values()}

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

    for attr_col, feat_name in ATTR_MATCH_COLS.items():
        val = attr_extra_arrs[attr_col][i]
        attr_match_arrays[feat_name][i] = 1 if (val and val in q) else 0

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

    if i % 300_000 == 0:
        print(f"İşlenen satır: {i}/{n_rows}", end="\r")

print(f"\nMetin feature'ları tamamlandı ({time.time()-t0:.1f}s geçti)")

features = pd.DataFrame({
    "id": merged["id"].values,
    "tfidf_cosine": tfidf_cosine,
    "tfidf_title_cosine": tfidf_title_cosine,
    "tfidf_category_cosine": tfidf_category_cosine,
    "tfidf_attributes_cosine": tfidf_attributes_cosine,
    "bm25_score": bm25_score,
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
for feat_name in ATTR_MATCH_COLS.values():
    features[feat_name] = attr_match_arrays[feat_name]

OUT_PATH = f"{DATA_DIR}/submission_features_v2.csv"
features.to_csv(OUT_PATH, index=False)
print(f"\nKaydedildi: {OUT_PATH}  (toplam süre: {time.time()-t0:.1f}s)")
print(features.shape)
