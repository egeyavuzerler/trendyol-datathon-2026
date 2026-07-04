import time
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = "data"
random.seed(42)
np.random.seed(42)

TOP_K = 50            # her terim için bakılacak en yakın ürün sayısı
SKIP_TOP = 10         # yanlış-negatif riskini azaltmak için atlanan en tepe sonuç sayısı
NEG_MULTIPLIER = 4     # HER POZİTİF için üretilecek negatif sayısı (1:4 oranı)
HARD_FRAC = 0.5        # negatiflerin yüzde kaçı hard negative (11-50. sıradan)
SAME_CAT_FRAC = 0.25   # yüzde kaçı aynı kategoriden
RANDOM_FRAC = 0.25     # yüzde kaçı tam rastgele
BATCH_SIZE = 300       # bellek/performans dengesi için terimleri batch halinde işle

t0 = time.time()

print("Veriler okunuyor...")
items = pd.read_csv(f"{DATA_DIR}/items_features.csv", usecols=["item_id", "raw_text", "category"])
terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
training_pairs = pd.read_csv(f"{DATA_DIR}/training_pairs.csv")

items["raw_text"] = items["raw_text"].fillna("")
items["category"] = items["category"].fillna("unknown")

item_id_array = items["item_id"].values
category_array = items["category"].values
n_items = len(item_id_array)

item_id_to_idx = {iid: i for i, iid in enumerate(item_id_array)}

print("Kategoriye göre ürün indeksleri gruplanıyor...")
category_to_indices = {}
for idx, cat in enumerate(category_array):
    category_to_indices.setdefault(cat, []).append(idx)

print(f"TF-IDF fit ediliyor ({n_items} ürün)...")
vectorizer = TfidfVectorizer(max_features=200_000, min_df=2, max_df=0.3, sublinear_tf=True)
item_matrix = vectorizer.fit_transform(items["raw_text"].values).tocsr()
item_matrix_T = item_matrix.transpose().tocsr()
print("item_matrix shape:", item_matrix.shape, f"({time.time()-t0:.1f}s geçti)")

# Sadece training_pairs'te geçen unique terimler
train_term_ids = training_pairs["term_id"].unique()
terms_train = terms[terms["term_id"].isin(train_term_ids)].reset_index(drop=True)
terms_train["query"] = terms_train["query"].fillna("")

print(f"{len(terms_train)} terim için query TF-IDF dönüşümü...")
query_matrix = vectorizer.transform(terms_train["query"].values).tocsr()

# term_id -> pozitif item index seti
term_to_pos_idx = (
    training_pairs.groupby("term_id")["item_id"]
    .apply(lambda s: set(item_id_to_idx[i] for i in s if i in item_id_to_idx))
    .to_dict()
)
# term_id -> pozitif sayısı (negatif hedefini bu sayıya göre ölçekleyeceğiz)
term_to_pos_count = training_pairs.groupby("term_id").size().to_dict()

term_ids_list = terms_train["term_id"].values
n_terms = len(term_ids_list)

negative_rows = []

print(f"\n{n_terms} terim için negatif örnek üretimi başlıyor (batch_size={BATCH_SIZE})...")
t1 = time.time()

for start in range(0, n_terms, BATCH_SIZE):
    end = min(start + BATCH_SIZE, n_terms)
    batch_query_matrix = query_matrix[start:end]
    sims = batch_query_matrix.dot(item_matrix_T).tocsr()  # (batch_size x n_items), sparse

    for local_i in range(end - start):
        term_id = term_ids_list[start + local_i]
        row = sims.getrow(local_i)

        if row.nnz > 0:
            data = row.data
            indices = row.indices
            k = min(TOP_K, len(data))
            if k < len(data):
                top_local = np.argpartition(-data, k - 1)[:k]
            else:
                top_local = np.arange(len(data))
            top_local = top_local[np.argsort(-data[top_local])]
            top_indices = indices[top_local].tolist()
        else:
            top_indices = []

        pos_idx_set = term_to_pos_idx.get(term_id, set())
        candidates = [idx for idx in top_indices if idx not in pos_idx_set]

        # Bu terimin pozitif sayısına göre negatif hedefini ölçekle (1:NEG_MULTIPLIER oranı)
        n_pos = term_to_pos_count.get(term_id, 1)
        target_neg = max(1, round(n_pos * NEG_MULTIPLIER))
        hard_target = round(target_neg * HARD_FRAC)
        same_cat_target = round(target_neg * SAME_CAT_FRAC)
        random_target = target_neg - hard_target - same_cat_target

        chosen = set()

        # 1) Hard negative: 11-50. sıradan (yanlış-negatif riskini azaltmak için ilk 10 atlanıyor)
        hard_pool = candidates[SKIP_TOP:TOP_K]
        hard_take = min(hard_target, len(hard_pool))
        hard_negatives = random.sample(hard_pool, hard_take) if hard_take > 0 else []
        chosen.update(hard_negatives)

        # 2) Aynı kategoriden negative
        same_cat_negatives = []
        if pos_idx_set:
            sample_pos_idx = next(iter(pos_idx_set))
            cat = category_array[sample_pos_idx]
            cat_pool = [idx for idx in category_to_indices.get(cat, []) if idx not in pos_idx_set and idx not in chosen]
            same_cat_take = min(same_cat_target, len(cat_pool))
            if same_cat_take > 0:
                same_cat_negatives = random.sample(cat_pool, same_cat_take)
                chosen.update(same_cat_negatives)

        # 3) Tam rastgele negative (kalan tüm hedefi doldur)
        remaining = target_neg - len(chosen)
        random_negatives = []
        attempts = 0
        max_attempts = remaining * 20 + 50
        while len(random_negatives) < remaining and attempts < max_attempts:
            r = random.randrange(n_items)
            attempts += 1
            if r not in pos_idx_set and r not in chosen:
                random_negatives.append(r)
                chosen.add(r)

        final_negatives = hard_negatives + same_cat_negatives + random_negatives

        for neg_idx in final_negatives:
            negative_rows.append((term_id, item_id_array[neg_idx], 0))

    elapsed = time.time() - t1
    print(f"İşlenen terim: {end}/{n_terms}  ({elapsed:.1f}s geçti)", end="\r")

print(f"\n\nToplam üretilen negatif satır: {len(negative_rows)}  (toplam süre: {time.time()-t0:.1f}s)")

neg_df = pd.DataFrame(negative_rows, columns=["term_id", "item_id", "label"])
neg_df.insert(0, "id", [f"NEG_{i}" for i in range(len(neg_df))])

pos_df = training_pairs[["id", "term_id", "item_id", "label"]]

full_train = pd.concat([pos_df, neg_df], ignore_index=True)
full_train.to_csv(f"{DATA_DIR}/training_full.csv", index=False)

print("\nKaydedildi:", f"{DATA_DIR}/training_full.csv")
print(full_train.shape)
print(full_train["label"].value_counts())