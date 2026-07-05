import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

DATA_DIR = "data"
N_FOLDS = 3          # tuning sırasında hız için 3 fold (final eğitimde 5 fold kullanacağız)
N_TRIALS = 40        # deneme sayısı (zamanın bol olduğunu belirttiğin için orta-yüksek bir değer)

t0 = time.time()
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("training_features_v2.csv okunuyor...")
df = pd.read_csv(f"{DATA_DIR}/training_features_v2.csv")
FEATURE_COLS = [c for c in df.columns if c not in ("id", "term_id", "item_id", "label")]

X = df[FEATURE_COLS].values
y = df["label"].values
groups = df["term_id"].values

gkf = GroupKFold(n_splits=N_FOLDS)
fold_indices = list(gkf.split(X, y, groups))


def objective(trial):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "verbose": -1,
        "seed": 42,
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }

    fold_f1s = []
    for train_idx, val_idx in fold_indices:
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )
        probs = model.predict(X_val, num_iteration=model.best_iteration)

        # Her fold içinde basit bir threshold taraması (hız için kaba adımlarla)
        best_fold_f1 = max(
            f1_score(y_val, (probs >= th).astype(int), average="macro")
            for th in np.arange(0.2, 0.6, 0.05)
        )
        fold_f1s.append(best_fold_f1)

    return float(np.mean(fold_f1s))


print(f"Optuna optimizasyonu başlıyor ({N_TRIALS} deneme, {N_FOLDS} fold)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\nOptimizasyon tamamlandı ({time.time()-t0:.1f}s geçti)")
print("En iyi macro F1 (3-fold ortalama):", study.best_value)
print("En iyi parametreler:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# Kaydet
import json
with open(f"{DATA_DIR}/best_lgbm_params.json", "w") as f:
    json.dump(study.best_params, f, indent=2)

print(f"\nKaydedildi: {DATA_DIR}/best_lgbm_params.json")
