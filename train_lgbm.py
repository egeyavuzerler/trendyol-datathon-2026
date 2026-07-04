import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

DATA_DIR = "data"
t0 = time.time()

print("training_features.csv okunuyor...")
df = pd.read_csv(f"{DATA_DIR}/training_features.csv")

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

X = df[FEATURE_COLS]
y = df["label"]

# Terim bazlı sızıntıyı önlemek için term_id'ye göre grup bazlı split yapıyoruz:
# aynı terim hem train hem validation'da karışık şekilde bulunmasın diye
unique_terms = df["term_id"].unique()
train_terms, val_terms = train_test_split(unique_terms, test_size=0.15, random_state=42)

train_mask = df["term_id"].isin(train_terms)
val_mask = df["term_id"].isin(val_terms)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

print("Model eğitiliyor...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, val_data],
    valid_names=["train", "val"],
    callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=50)],
)

print(f"\nEğitim tamamlandı ({time.time()-t0:.1f}s geçti). En iyi iterasyon: {model.best_iteration}")

# ============================================================
# Validation üzerinde macro F1 için en iyi threshold'u bul
# ============================================================
val_probs = model.predict(X_val, num_iteration=model.best_iteration)

print("\nFarklı threshold'lar için macro F1:")
best_f1 = -1
best_threshold = 0.5
for threshold in np.arange(0.1, 0.9, 0.05):
    preds = (val_probs >= threshold).astype(int)
    f1 = f1_score(y_val, preds, average="macro")
    marker = ""
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        marker = "  <-- en iyi"
    print(f"threshold={threshold:.2f}  macro_F1={f1:.4f}{marker}")

print(f"\nEn iyi threshold: {best_threshold:.2f}  (macro F1 = {best_f1:.4f})")

final_preds = (val_probs >= best_threshold).astype(int)
print("\nSınıflandırma raporu (validation, en iyi threshold ile):")
print(classification_report(y_val, final_preds, target_names=["irrelevant(0)", "relevant(1)"]))

# ============================================================
# Feature importance (açıklanabilirlik için)
# ============================================================
importance = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": model.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False)

print("\n=== FEATURE IMPORTANCE (gain) ===")
print(importance.to_string(index=False))

# ============================================================
# Modeli ve threshold'u kaydet
# ============================================================
model.save_model(f"{DATA_DIR}/lgbm_model.txt")
with open(f"{DATA_DIR}/best_threshold.txt", "w") as f:
    f.write(str(best_threshold))

print(f"\nModel kaydedildi: {DATA_DIR}/lgbm_model.txt")
print(f"Threshold kaydedildi: {DATA_DIR}/best_threshold.txt ({best_threshold:.2f})")
