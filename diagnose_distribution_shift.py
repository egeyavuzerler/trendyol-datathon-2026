import pandas as pd

DATA_DIR = "data"

print("training_features_v2.csv okunuyor...")
train_df = pd.read_csv(f"{DATA_DIR}/training_features_v2.csv")

print("submission_features_v2.csv okunuyor...")
sub_df = pd.read_csv(f"{DATA_DIR}/submission_features_v2.csv")

KEY_FEATURES = [
    "tfidf_cosine", "bm25_score", "tfidf_title_cosine", "tfidf_category_cosine",
    "word_overlap_ratio", "brand_in_query", "exact_substring_match",
    "category_token_overlap_ratio", "gender_conflict",
]

pos = train_df[train_df["label"] == 1]
neg = train_df[train_df["label"] == 0]

print("\n=== DAĞILIM KARŞILAŞTIRMASI (ortalama değerler) ===")
print(f"{'feature':<30} {'train_pos':>12} {'train_neg':>12} {'submission':>12}  submission_pos'a_mı_neg'e_mi_yakın")
for feat in KEY_FEATURES:
    m_pos = pos[feat].mean()
    m_neg = neg[feat].mean()
    m_sub = sub_df[feat].mean()
    # submission ortalamasının pos/neg ortalamasına göreceli konumu (0=neg'e özdeş, 1=pos'a özdeş)
    if m_pos != m_neg:
        relative_pos = (m_sub - m_neg) / (m_pos - m_neg)
    else:
        relative_pos = float("nan")
    print(f"{feat:<30} {m_pos:>12.4f} {m_neg:>12.4f} {m_sub:>12.4f}  relative_position={relative_pos:.2f}")

print("\nYorum: relative_position ~0 ise submission negatiflerimize benziyor (varsayımımız doğru).")
print("relative_position ~1 veya üstü ise submission pozitiflerimize daha çok benziyor")
print("(yani gerçek pozitif oranı muhtemelen bizim %20 varsayımımızdan çok daha yüksek).")

print("\n=== GENEL İSTATİSTİKLER (quantile karşılaştırması, tfidf_cosine) ===")
print("train_pos quantiles:\n", pos["tfidf_cosine"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))
print("\ntrain_neg quantiles:\n", neg["tfidf_cosine"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))
print("\nsubmission quantiles:\n", sub_df["tfidf_cosine"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))
