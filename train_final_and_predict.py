import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, classification_report

DATA_DIR = "data"
N_FOLDS = 5
t0 = time.time()

print("Optuna en iyi parametreleri yükleniyor...")
with open(f"{DATA_DIR}/best_lgbm_params.json") as f:
    tuned_params = json.load(f)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "verbose": -1,
    "seed": 42,
}
params.update(tuned_params)
print("Kullanılacak parametreler:", params)

print("\ntraining_features_v2.csv okunuyor...")
df = pd.read_csv(f"{DATA_DIR}/training_features_v2.csv")
FEATURE_COLS = [c for c in df.columns if c not in ("id", "term_id", "item_id", "label")]

X = df[FEATURE_COLS].values
y = df["label"].values
groups = df["term_id"].values

gkf = GroupKFold(n_splits=N_FOLDS)

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
        num_boost_round=1500,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(period=200)],
    )

    fold_probs = model.predict(X_val, num_iteration=model.best_iteration)
    oof_probs[val_idx] = fold_probs

    model.save_model(f"{DATA_DIR}/lgbm_final_fold{fold}.txt")
    models.append(model)
    importances.append(model.feature_importance(importance_type="gain"))

    fold_f1 = f1_score(y_val, (fold_probs >= 0.35).astype(int), average="macro")
    print(f"Fold {fold+1} macro F1 (threshold=0.35): {fold_f1:.4f}  (best_iteration={model.best_iteration})")

print(f"\nTüm foldlar tamamlandı ({time.time()-t0:.1f}s geçti)")

print("\n=== OOF DEĞERLENDİRME ===")
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
print("\nSınıflandırma raporu (OOF, final model):")
print(classification_report(y, final_preds, target_names=["irrelevant(0)", "relevant(1)"]))

avg_importance = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": np.mean(importances, axis=0),
}).sort_values("importance", ascending=False)
print("\n=== ORTALAMA FEATURE IMPORTANCE (final model) ===")
print(avg_importance.to_string(index=False))

with open(f"{DATA_DIR}/best_threshold_final.txt", "w") as f:
    f.write(str(best_threshold))

# ============================================================
# Submission üzerinde ensemble tahmin (5 modelin ortalaması)
# ============================================================
print("\nsubmission_features_v2.csv okunuyor...")
sub_features = pd.read_csv(f"{DATA_DIR}/submission_features_v2.csv")
X_sub = sub_features[FEATURE_COLS].values

print("5 model ile ensemble tahmin yapılıyor...")
sub_probs = np.zeros(len(sub_features))
for model in models:
    sub_probs += model.predict(X_sub, num_iteration=model.best_iteration)
sub_probs /= len(models)

sub_preds = (sub_probs >= best_threshold).astype(int)
print("Tahmin dağılımı:")
print(pd.Series(sub_preds).value_counts())

submission = pd.DataFrame({"id": sub_features["id"], "prediction": sub_preds})

sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
assert len(submission) == len(sample), "Satır sayısı uyuşmuyor!"
assert set(submission["id"]) == set(sample["id"]), "id kümesi uyuşmuyor!"
submission = submission.set_index("id").loc[sample["id"]].reset_index()

OUT_PATH = f"{DATA_DIR}/submission_v2.csv"
submission.to_csv(OUT_PATH, index=False)
print(f"\nKaydedildi: {OUT_PATH}")
print(submission.head())
print(submission.shape)
print(f"\nToplam süre: {time.time()-t0:.1f}s")
