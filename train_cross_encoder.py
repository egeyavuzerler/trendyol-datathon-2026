import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset

DATA_DIR = "data"
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
MAX_LENGTH = 64            # 96 -> 64: çoğu metin kısa, hız için düşürüldü
NEG_PER_POS_SUBSAMPLE = 2  # BERT eğitimi pahalı olduğu için negatifleri 1:4 -> 1:2'ye düşürüyoruz
BATCH_SIZE = 32            # 16 -> 32: M5 GPU'sunu daha verimli kullanmak için
NUM_EPOCHS = 1             # 2 -> 1: 750K satır için genelde yeterli, süreyi yarıya indirir
LEARNING_RATE = 2e-5

# Gerçek submission dağılımına yakınsamak için negatif karışımı hedefi:
# Diagnostic bulgumuz: submission'ın medyan tfidf_cosine'ı 0.0 (yarısından fazlası hiç örtüşmüyor)
# Bu yüzden düşük benzerlikli (kolay) negatiflere ağırlık veriyoruz
NEG_BUCKET_TARGET_RATIOS = {"low": 0.5, "mid": 0.3, "high": 0.2}

t0 = time.time()

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılacak cihaz: {device}")

# ============================================================
# 1) Veriyi hazırla: query + item metni çiftleri
# ============================================================
print("Veriler okunuyor...")
train_full = pd.read_csv(f"{DATA_DIR}/training_full.csv")
terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
terms["query"] = terms["query"].fillna("")
items = pd.read_csv(f"{DATA_DIR}/items_features.csv", usecols=["item_id", "title", "category", "brand", "attributes_raw"])
for col in ["title", "category", "brand", "attributes_raw"]:
    items[col] = items[col].fillna("")

# Negatifleri alt örnekle, ama artık RASTGELE değil, gerçek dağılıma göre AĞIRLIKLI:
# training_features_v2.csv'den tfidf_cosine'ı çekip düşük/orta/yüksek benzerlik kovalarına ayırıyoruz
pos_df = train_full[train_full["label"] == 1]
neg_df = train_full[train_full["label"] == 0]

print("Negatiflerin benzerlik skorları için training_features_v2.csv okunuyor...")
neg_features = pd.read_csv(f"{DATA_DIR}/training_features_v2.csv", usecols=["id", "tfidf_cosine"])
neg_df = neg_df.merge(neg_features, on="id", how="left")

LOW_MAX = 0.02
MID_MAX = 0.20
neg_low = neg_df[neg_df["tfidf_cosine"] <= LOW_MAX]
neg_mid = neg_df[(neg_df["tfidf_cosine"] > LOW_MAX) & (neg_df["tfidf_cosine"] <= MID_MAX)]
neg_high = neg_df[neg_df["tfidf_cosine"] > MID_MAX]
print(f"Negatif kovaları -> low(<= {LOW_MAX}): {len(neg_low)}, mid: {len(neg_mid)}, high(> {MID_MAX}): {len(neg_high)}")

target_neg_count = min(len(neg_df), len(pos_df) * NEG_PER_POS_SUBSAMPLE)
target_low = int(target_neg_count * NEG_BUCKET_TARGET_RATIOS["low"])
target_mid = int(target_neg_count * NEG_BUCKET_TARGET_RATIOS["mid"])
target_high = target_neg_count - target_low - target_mid

sampled_parts = []
for bucket_df, target_n, name in [(neg_low, target_low, "low"), (neg_mid, target_mid, "mid"), (neg_high, target_high, "high")]:
    n = min(target_n, len(bucket_df))
    sampled_parts.append(bucket_df.sample(n=n, random_state=42))
    print(f"  {name}: hedef={target_n}, alınan={n}")

neg_df_sampled = pd.concat(sampled_parts, ignore_index=True).drop(columns=["tfidf_cosine"])
train_subset = pd.concat([pos_df, neg_df_sampled], ignore_index=True)
print(f"Alt örneklenmiş eğitim seti: {len(pos_df)} pozitif + {len(neg_df_sampled)} negatif = {len(train_subset)} satır")

merged = train_subset.merge(terms, on="term_id", how="left")
merged = merged.merge(items, on="item_id", how="left")
merged["query"] = merged["query"].fillna("")

# Item metni: title + category + brand + attributes (kısa ve öz, BERT için)
merged["item_text"] = (
    merged["title"] + " | " +
    merged["category"].str.replace("/", " ", regex=False) + " | " +
    merged["brand"] + " | " +
    merged["attributes_raw"].str.slice(0, 200)
)

print(f"merged shape: {merged.shape}  ({time.time()-t0:.1f}s geçti)")

# ============================================================
# 2) Terim bazlı train/val split
# ============================================================
unique_terms = merged["term_id"].unique()
train_terms, val_terms = train_test_split(unique_terms, test_size=0.1, random_state=42)
train_mask = merged["term_id"].isin(train_terms)
val_mask = merged["term_id"].isin(val_terms)

train_data = merged[train_mask].reset_index(drop=True)
val_data = merged[val_mask].reset_index(drop=True)
print(f"Train: {len(train_data)}, Val: {len(val_data)}")

# ============================================================
# 3) Tokenizer ve Dataset sınıfı
# ============================================================
print(f"\nTokenizer yükleniyor: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class PairDataset(Dataset):
    def __init__(self, queries, item_texts, labels):
        self.queries = queries
        self.item_texts = item_texts
        self.labels = labels

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.queries[idx],
            self.item_texts[idx],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


train_dataset = PairDataset(train_data["query"].tolist(), train_data["item_text"].tolist(), train_data["label"].tolist())
val_dataset = PairDataset(val_data["query"].tolist(), val_data["item_text"].tolist(), val_data["label"].tolist())

# ============================================================
# 4) Model ve eğitim
# ============================================================
print(f"\nModel yükleniyor: {MODEL_NAME}...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}


training_args = TrainingArguments(
    output_dir=f"{DATA_DIR}/cross_encoder_checkpoints",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=200,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print(f"\nEğitim başlıyor ({len(train_dataset)} örnek, {NUM_EPOCHS} epoch, batch_size={BATCH_SIZE})...")
print("Bu adım uzun sürebilir, sabırlı ol...\n")
trainer.train()

print(f"\nEğitim tamamlandı ({time.time()-t0:.1f}s geçti)")

# ============================================================
# 5) Validation üzerinde en iyi threshold'u bul
# ============================================================
print("\nValidation üzerinde tahmin yapılıyor...")
predictions = trainer.predict(val_dataset)
logits = predictions.predictions
probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()  # relevant(1) sınıfı olasılığı
y_val = val_data["label"].values

print("\nFarklı threshold'lar için macro F1:")
best_f1 = -1
best_threshold = 0.5
for threshold in np.arange(0.1, 0.9, 0.02):
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(y_val, preds, average="macro")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"En iyi threshold (internal val F1): {best_threshold:.2f}  (macro F1 = {best_f1:.4f})")

# Leaderboard probing'den öğrendiğimiz hedef: submission'da ~%26-30 relevant tahmin oranı en iyi skoru verdi
print("\nHedef pozitif oranına (%26-30) denk gelen threshold referansı:")
for target_rate in [0.26, 0.28, 0.30]:
    th_for_rate = np.quantile(probs, 1 - target_rate)
    print(f"  %{target_rate*100:.0f} relevant oranı için threshold ≈ {th_for_rate:.3f}")

final_preds = (probs >= best_threshold).astype(int)
print("\nSınıflandırma raporu (validation, cross-encoder):")
print(classification_report(y_val, final_preds, target_names=["irrelevant(0)", "relevant(1)"]))

# ============================================================
# 6) Modeli kaydet
# ============================================================
model.save_pretrained(f"{DATA_DIR}/cross_encoder_model")
tokenizer.save_pretrained(f"{DATA_DIR}/cross_encoder_model")
with open(f"{DATA_DIR}/cross_encoder_threshold.txt", "w") as f:
    f.write(str(best_threshold))

print(f"\nModel kaydedildi: {DATA_DIR}/cross_encoder_model")
print(f"Threshold kaydedildi: {DATA_DIR}/cross_encoder_threshold.txt")
print(f"\nToplam süre: {time.time()-t0:.1f}s")