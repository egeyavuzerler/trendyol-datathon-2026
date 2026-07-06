import pandas as pd

DATA_DIR = "data"

print("training_features_v3.csv + cross_encoder_train_probs.csv birleştiriliyor...")
train_df = pd.read_csv(f"{DATA_DIR}/training_features_v3.csv")
ce_train = pd.read_csv(f"{DATA_DIR}/cross_encoder_train_probs.csv")
train_v4 = train_df.merge(ce_train, on="id", how="left")
print("Eksik ce_prob sayısı (train):", train_v4["ce_prob"].isna().sum())
train_v4.to_csv(f"{DATA_DIR}/training_features_v4.csv", index=False)
print(f"Kaydedildi: {DATA_DIR}/training_features_v4.csv  shape={train_v4.shape}")

print("\nsubmission_features_v3.csv + cross_encoder_submission_probs.csv birleştiriliyor...")
sub_df = pd.read_csv(f"{DATA_DIR}/submission_features_v3.csv")
ce_sub = pd.read_csv(f"{DATA_DIR}/cross_encoder_submission_probs.csv")
ce_sub = ce_sub.rename(columns={"prob": "ce_prob"})
sub_v4 = sub_df.merge(ce_sub, on="id", how="left")
print("Eksik ce_prob sayısı (submission):", sub_v4["ce_prob"].isna().sum())
sub_v4.to_csv(f"{DATA_DIR}/submission_features_v4.csv", index=False)
print(f"Kaydedildi: {DATA_DIR}/submission_features_v4.csv  shape={sub_v4.shape}")

print("\nLabel'a göre ce_prob ortalaması (train):")
print(train_v4.groupby("label")["ce_prob"].mean())