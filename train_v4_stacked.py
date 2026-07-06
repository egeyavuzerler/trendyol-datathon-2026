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
    "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
    "verbose": -1, "seed": 42,
}
params.update(tuned_params)

print("\ntraining_features_v4.csv okunuyor...")
df = pd.read_csv(f"{DATA_DIR}/training_features_v4.csv")
FEATURE_COLS = [c for c in df.columns if c not in ("id", "term_id", "item_id", "label")]
print("Kullanılan feature sayısı:", len(FEATURE_COLS))
print(FEATURE_COLS)

X = df[FEATURE_COLS].values
y = df["label"].values
groups = df["term_id"].values

gkf = GroupKFold(n_splits=N_FOLDS)

oof_probs = np.zeros(len(df))
models = []
importances = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    print(f"\n=== FOLD {fold+1}/{N_FOLDS} ===")
    train_data = lgb.Dataset(X[train_idx], label=y[train_idx], feature_name=FEATURE_COLS)
    val_data = lgb.Dataset(X[val_idx], label=y[val_idx], reference=train_data, feature_name=FEATURE_COLS)

    model = lgb.train(
        params, train_data, num_boost_round=1500,
        valid_sets=[train_data, val_data], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(period=200)],
    )
    oof_probs[val_idx] = model.predict(X[val_idx], num_iteration=model.best_iteration)
    model.save_model(f"{DATA_DIR}/lgbm_v4_fold{fold}.txt")
    models.append(model)
    importances.append(model.feature_importance(importance_type="gain"))
    fold_f1 = f1_score(y[val_idx], (oof_probs[val_idx] >= 0.35).astype(int), average="macro")
    print(f"Fold {fold+1} macro F1 (threshold=0.35): {fold_f1:.4f}  (best_iteration={model.best_iteration})")

print(f"\nTüm foldlar tamamlandı ({time.time()-t0:.1f}s geçti)")

print("\n=== OOF DEĞERLENDİRME (v4, stacked) ===")
best_f1 = -1
best_threshold = 0.5
for threshold in np.arange(0.1, 0.9, 0.02):
    f1 = f1_score(y, (oof_probs >= threshold).astype(int), average="macro")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
print(f"En iyi threshold: {best_threshold:.2f}  (OOF macro F1 = {best_f1:.4f})")
print("\nSınıflandırma raporu (OOF, v4 stacked):")
print(classification_report(y, (oof_probs >= best_threshold).astype(int), target_names=["irrelevant(0)", "relevant(1)"]))

avg_importance = pd.DataFrame({
    "feature": FEATURE_COLS, "importance": np.mean(importances, axis=0),
}).sort_values("importance", ascending=False)
print("\n=== ORTALAMA FEATURE IMPORTANCE (v4 stacked) ===")
print(avg_importance.to_string(index=False))

# ============================================================
# Submission üzerinde ensemble tahmin
# ============================================================
print("\nsubmission_features_v4.csv okunuyor...")
sub_features = pd.read_csv(f"{DATA_DIR}/submission_features_v4.csv")
X_sub = sub_features[FEATURE_COLS].values

sub_probs = np.zeros(len(sub_features))
for model in models:
    sub_probs += model.predict(X_sub, num_iteration=model.best_iteration)
sub_probs /= len(models)

prob_df = pd.DataFrame({"id": sub_features["id"], "prob": sub_probs})
prob_df.to_csv(f"{DATA_DIR}/lgbm_v4_submission_probs.csv", index=False)

sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
print("\nHedef pozitif oranlarına göre v4 (stacked) submission dosyaları üretiliyor:")
for target_rate in [0.20, 0.23, 0.26, 0.28, 0.30]:
    th = float(np.quantile(sub_probs, 1 - target_rate))
    preds = (sub_probs >= th).astype(int)
    sub = pd.DataFrame({"id": prob_df["id"], "prediction": preds})
    sub = sub.set_index("id").loc[sample["id"]].reset_index()
    out_path = f"{DATA_DIR}/submission_v4_stacked_rate{int(target_rate*100)}.csv"
    sub.to_csv(out_path, index=False)
    print(f"  rate{int(target_rate*100)}: threshold={th:.3f}  gerçek_oran=%{100*preds.mean():.1f}  -> {out_path}")

print(f"\nToplam süre: {time.time()-t0:.1f}s")
