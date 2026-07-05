import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import lightgbm as lgb

DATA_DIR = "data"

print("training_features_v2.csv okunuyor...")
df = pd.read_csv(f"{DATA_DIR}/training_features_v2.csv")
FEATURE_COLS = [c for c in df.columns if c not in ("id", "term_id", "item_id", "label")]

X = df[FEATURE_COLS].values
y = df["label"].values
groups = df["term_id"].values  # mevcut CV stratejimiz: sadece term_id bazlı gruplama

N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)

params = {
    "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
    "num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.9,
    "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1, "seed": 42,
}

oof_probs = np.zeros(len(df))
oof_item_seen_in_train = np.zeros(len(df), dtype=bool)

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    train_items = set(df["item_id"].values[train_idx])
    val_items = df["item_id"].values[val_idx]
    seen_mask = np.array([iid in train_items for iid in val_items])
    oof_item_seen_in_train[val_idx] = seen_mask

    train_data = lgb.Dataset(X[train_idx], label=y[train_idx])
    val_data = lgb.Dataset(X[val_idx], label=y[val_idx], reference=train_data)
    model = lgb.train(
        params, train_data, num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    oof_probs[val_idx] = model.predict(X[val_idx], num_iteration=model.best_iteration)
    print(f"Fold {fold+1} tamam. Val'deki item'ların train'de görülme oranı: %{100*seen_mask.mean():.1f}")

# ============================================================
# Karşılaştırma: item'ı training'de görülmüş satırlar vs hiç görülmemiş satırlar
# ============================================================
threshold = 0.34  # önceki en iyi threshold

preds = (oof_probs >= threshold).astype(int)

seen_mask_all = oof_item_seen_in_train
unseen_mask_all = ~oof_item_seen_in_train

f1_seen = f1_score(y[seen_mask_all], preds[seen_mask_all], average="macro")
f1_unseen = f1_score(y[unseen_mask_all], preds[unseen_mask_all], average="macro")
f1_overall = f1_score(y, preds, average="macro")

print("\n=== ITEM-LEVEL LEAKAGE TANI SONUCU ===")
print(f"Genel OOF macro F1: {f1_overall:.4f}")
print(f"Item training'de GÖRÜLMÜŞ satırlarda macro F1: {f1_seen:.4f}  (n={seen_mask_all.sum()}, %{100*seen_mask_all.mean():.1f} of val)")
print(f"Item training'de HİÇ GÖRÜLMEMİŞ satırlarda macro F1: {f1_unseen:.4f}  (n={unseen_mask_all.sum()}, %{100*unseen_mask_all.mean():.1f} of val)")
print(f"\nFark (seen - unseen): {f1_seen - f1_unseen:+.4f}")
print("Not: Bu fark büyükse (>0.03-0.05 gibi), item-level leakage OOF skorunu şişiriyor demektir.")
print("Submission'da item'ların %78.9'u tamamen görülmemiş olduğu için, gerçek performans 'unseen' sütununa daha yakın olabilir.")
