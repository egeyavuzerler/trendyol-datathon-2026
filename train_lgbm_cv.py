import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, classification_report

DATA_DIR = "data"
N_FOLDS = 5
t0 = time.time()

print("training_features_v2.csv okunuyor...")
df = pd.read_csv(f"{DATA_DIR}/training_features_v2.csv")

FEATURE_COLS = [c for c in df.columns if c not in ("id", "term_id", "item_id", "label")]
print("Kullanılan feature sayısı:", len(FEATURE_COLS))
print(FEATURE_COLS)

X = df[FEATURE_COLS].values
y = df["label"].values
groups = df["term_id"].values  # aynı terim hep aynı fold'da kalsın (veri sızıntısını önlemek için)

gkf = GroupKFold(n_splits=N_FOLDS)

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

oof_probs = np.zeros(len(df))
models = []
importances = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    print(f"\n=== FOLD {fold+1}/{N_FOLDS} ===")
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=FEATURE_COLS)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=100)],
    )

    fold_probs = model.predict(X_val, num_iteration=model.best_iteration)
    oof_probs[val_idx] = fold_probs

    model.save_model(f"{DATA_DIR}/lgbm_model_fold{fold}.txt")
    models.append(model)
    importances.append(model.feature_importance(importance_type="gain"))

    fold_f1 = f1_score(y_val, (fold_probs >= 0.5).astype(int), average="macro")
    print(f"Fold {fold+1} macro F1 (threshold=0.5): {fold_f1:.4f}   (best_iteration={model.best_iteration})")

print(f"\nTüm foldlar tamamlandı ({time.time()-t0:.1f}s geçti)")

# ============================================================
# OOF (out-of-fold) tahminleri üzerinden genel değerlendirme
# ============================================================
print("\n=== OOF DEĞERLENDİRME (tüm veri, her satır kendi fold'unun modeliyle tahmin edilmiş) ===")
print("Farklı threshold'lar için macro F1:")
best_f1 = -1
best_threshold = 0.5
for threshold in np.arange(0.1, 0.9, 0.02):
    preds = (oof_probs >= threshold).astype(int)
    f1 = f1_score(y, preds, average="macro")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"En iyi threshold: {best_threshold:.2f}  (OOF macro F1 = {best_f1:.4f})")

final_preds = (oof_probs >= best_threshold).astype(int)
print("\nSınıflandırma raporu (OOF, en iyi threshold ile):")
print(classification_report(y, final_preds, target_names=["irrelevant(0)", "relevant(1)"]))

# ============================================================
# Ortalama feature importance (5 fold ortalaması)
# ============================================================
avg_importance = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": np.mean(importances, axis=0),
}).sort_values("importance", ascending=False)

print("\n=== ORTALAMA FEATURE IMPORTANCE (5-fold, gain) ===")
print(avg_importance.to_string(index=False))

with open(f"{DATA_DIR}/best_threshold_cv.txt", "w") as f:
    f.write(str(best_threshold))

print(f"\nThreshold kaydedildi: {DATA_DIR}/best_threshold_cv.txt ({best_threshold:.2f})")
print(f"5 model kaydedildi: {DATA_DIR}/lgbm_model_fold0.txt ... fold4.txt")
print(f"\nToplam süre: {time.time()-t0:.1f}s")
