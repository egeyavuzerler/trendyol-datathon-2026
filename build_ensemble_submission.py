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
    "semantic_cosine",
]

print("Cross-encoder olasılıkları okunuyor...")
ce_probs = pd.read_csv(f"{DATA_DIR}/cross_encoder_submission_probs.csv")

print("LightGBM v3 (5 fold) modelleri yükleniyor ve submission_features_v3.csv üzerinde tahmin yapılıyor...")
models = [lgb.Booster(model_file=f"{DATA_DIR}/lgbm_v3_fold{i}.txt") for i in range(5)]
sub_features = pd.read_csv(f"{DATA_DIR}/submission_features_v3.csv")
X_sub = sub_features[FEATURE_COLS].values

lgbm_probs = np.zeros(len(sub_features))
for model in models:
    lgbm_probs += model.predict(X_sub, num_iteration=model.best_iteration)
lgbm_probs /= len(models)

lgbm_df = pd.DataFrame({"id": sub_features["id"], "lgbm_prob": lgbm_probs})

merged = ce_probs.merge(lgbm_df, on="id", how="left")
merged = merged.rename(columns={"prob": "ce_prob"})

# İki modelin olasılıklarını RANK bazında normalize edip ortalamak, farklı ölçeklerdeki
# olasılık dağılımlarını (cross-encoder çok keskin/aşırı-güvenli, LightGBM daha yumuşak) adil şekilde birleştirir
merged["ce_rank"] = merged["ce_prob"].rank(pct=True)
merged["lgbm_rank"] = merged["lgbm_prob"].rank(pct=True)

# Cross-encoder çok daha güçlü olduğu için ona daha fazla ağırlık veriyoruz (0.7 / 0.3)
merged["ensemble_score"] = 0.7 * merged["ce_rank"] + 0.3 * merged["lgbm_rank"]

merged.to_csv(f"{DATA_DIR}/ensemble_scores.csv", index=False)
print(f"Kaydedildi: {DATA_DIR}/ensemble_scores.csv")

sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print("\nHedef pozitif oranlarına göre ensemble submission dosyaları üretiliyor:")
for target_rate in [0.22, 0.24, 0.26]:
    th = merged["ensemble_score"].quantile(1 - target_rate)
    preds = (merged["ensemble_score"] >= th).astype(int)
    actual_rate = preds.mean()
    sub = pd.DataFrame({"id": merged["id"], "prediction": preds})
    sub = sub.set_index("id").loc[sample["id"]].reset_index()
    out_path = f"{DATA_DIR}/submission_ensemble_rate{int(target_rate*100)}.csv"
    sub.to_csv(out_path, index=False)
    print(f"  rate{int(target_rate*100)}: gerçek_oran=%{100*actual_rate:.1f}  -> {out_path}")
