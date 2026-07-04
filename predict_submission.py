import pandas as pd
import lightgbm as lgb

DATA_DIR = "data"

FEATURE_COLS = [
    "tfidf_cosine",
    "query_word_count",
    "title_word_count",
    "word_overlap_count",
    "word_overlap_ratio",
    "exact_substring_match",
    "brand_in_query",
    "color_match",
    "category_token_overlap_ratio",
    "gender_match",
    "gender_conflict",
]

print("Model ve threshold yükleniyor...")
model = lgb.Booster(model_file=f"{DATA_DIR}/lgbm_model.txt")
with open(f"{DATA_DIR}/best_threshold.txt") as f:
    threshold = float(f.read().strip())
print("Kullanılan threshold:", threshold)

print("submission_features.csv okunuyor...")
features = pd.read_csv(f"{DATA_DIR}/submission_features.csv")

print("Tahmin yapılıyor...")
probs = model.predict(features[FEATURE_COLS])
preds = (probs >= threshold).astype(int)

print("Tahmin dağılımı:")
print(pd.Series(preds).value_counts())

submission = pd.DataFrame({
    "id": features["id"],
    "prediction": preds,
})

# sample_submission.csv ile aynı sıra/format kontrolü
sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
assert len(submission) == len(sample), "Satır sayısı uyuşmuyor!"
assert set(submission["id"]) == set(sample["id"]), "id kümesi uyuşmuyor!"

# sample_submission ile aynı id sırasına göre hizala
submission = submission.set_index("id").loc[sample["id"]].reset_index()

OUT_PATH = f"{DATA_DIR}/submission.csv"
submission.to_csv(OUT_PATH, index=False)
print(f"\nKaydedildi: {OUT_PATH}")
print(submission.head())
print(submission.shape)
