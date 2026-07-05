import numpy as np
import pandas as pd
import lightgbm as lgb

DATA_DIR = "data"

FEATURE_COLS = [
    "tfidf_cosine", "tfidf_title_cosine", "tfidf_category_cosine", "tfidf_attributes_cosine",
    "bm25_score", "query_word_count", "title_word_count", "word_overlap_count",
    "word_overlap_ratio", "exact_substring_match", "brand_in_query", "color_match",
    "category_token_overlap_ratio", "gender_match", "gender_conflict",
    "materyal_match", "desen_match", "kumas_tipi_match", "urun_tipi_match",
]

print("Kaydedilmiş 5 fold modeli yükleniyor (v2, semantic_cosine olmadan)...")
models = [lgb.Booster(model_file=f"{DATA_DIR}/lgbm_final_fold{i}.txt") for i in range(5)]

print("submission_features_v2.csv okunuyor...")
sub_features = pd.read_csv(f"{DATA_DIR}/submission_features_v2.csv")
X_sub = sub_features[FEATURE_COLS].values

print("Ensemble tahmin yapılıyor (olasılık, threshold uygulanmadan)...")
probs = np.zeros(len(sub_features))
for model in models:
    probs += model.predict(X_sub, num_iteration=model.best_iteration)
probs /= len(models)

# Ham olasılıkları kaydet (tekrar tekrar hesaplamamak için)
prob_df = pd.DataFrame({"id": sub_features["id"], "prob": probs})
prob_df.to_csv(f"{DATA_DIR}/submission_probs.csv", index=False)
print(f"Ham olasılıklar kaydedildi: {DATA_DIR}/submission_probs.csv")

print("\nOlasılık dağılımı özeti:")
print(prob_df["prob"].describe())

sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

# Test edilecek threshold'lar: geniş bir aralık, gerçek dağılımı anlamak için
THRESHOLDS_TO_TEST = [0.15, 0.20, 0.25, 0.30, 0.34, 0.40, 0.50]

print("\nFarklı threshold'lar için tahmin dağılımı ve dosya üretimi:")
for th in THRESHOLDS_TO_TEST:
    preds = (probs >= th).astype(int)
    positive_rate = preds.mean()
    submission = pd.DataFrame({"id": sub_features["id"], "prediction": preds})
    submission = submission.set_index("id").loc[sample["id"]].reset_index()
    out_path = f"{DATA_DIR}/submission_threshold_{th:.2f}.csv"
    submission.to_csv(out_path, index=False)
    print(f"  threshold={th:.2f}  relevant_oranı=%{100*positive_rate:.1f}  -> {out_path}")

print("\nTümü kaydedildi. Kaggle'a birkaçını (örn. 0.15, 0.25, 0.40) sırayla yükleyip")
print("public skorun nasıl değiştiğini gözlemleyelim.")
