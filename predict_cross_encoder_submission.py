import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_DIR = "data"
MODEL_PATH = f"{DATA_DIR}/cross_encoder_model"
MAX_LENGTH = 64
INFERENCE_BATCH_SIZE = 256  # eğitimdekinden çok daha büyük (gradient yok, daha fazla bellek kullanılabilir)

t0 = time.time()

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılacak cihaz: {device}")

print("Model ve tokenizer yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

print("Veriler okunuyor...")
submission_pairs = pd.read_csv(f"{DATA_DIR}/submission_pairs.csv")
terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
terms["query"] = terms["query"].fillna("")
items = pd.read_csv(f"{DATA_DIR}/items_features.csv", usecols=["item_id", "title", "category", "brand", "attributes_raw"])
for col in ["title", "category", "brand", "attributes_raw"]:
    items[col] = items[col].fillna("")

print("Birleştirme yapılıyor...")
merged = submission_pairs.merge(terms, on="term_id", how="left")
merged = merged.merge(items, on="item_id", how="left")
merged["query"] = merged["query"].fillna("")

merged["item_text"] = (
    merged["title"] + " | " +
    merged["category"].str.replace("/", " ", regex=False) + " | " +
    merged["brand"] + " | " +
    merged["attributes_raw"].str.slice(0, 200)
)
print(f"merged shape: {merged.shape}  ({time.time()-t0:.1f}s geçti)")

queries = merged["query"].tolist()
item_texts = merged["item_text"].tolist()
n_rows = len(merged)

print(f"\nInference başlıyor ({n_rows} satır, batch_size={INFERENCE_BATCH_SIZE})...")
all_probs = np.zeros(n_rows, dtype=np.float32)

with torch.no_grad():
    for start in range(0, n_rows, INFERENCE_BATCH_SIZE):
        end = min(start + INFERENCE_BATCH_SIZE, n_rows)
        batch_queries = queries[start:end]
        batch_items = item_texts[start:end]

        encoding = tokenizer(
            batch_queries,
            batch_items,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(device)

        logits = model(**encoding).logits
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs[start:end] = probs

        if start % (INFERENCE_BATCH_SIZE * 50) == 0:
            elapsed = time.time() - t0
            rate = (start + 1) / elapsed if elapsed > 0 else 0
            remaining = (n_rows - start) / rate if rate > 0 else float("inf")
            print(f"İşlenen: {start}/{n_rows}  ({elapsed:.0f}s geçti, tahmini kalan: {remaining/60:.1f} dk)", end="\r")

print(f"\n\nInference tamamlandı ({time.time()-t0:.1f}s geçti)")

prob_df = pd.DataFrame({"id": merged["id"], "prob": all_probs})
prob_df.to_csv(f"{DATA_DIR}/cross_encoder_submission_probs.csv", index=False)
print(f"Kaydedildi: {DATA_DIR}/cross_encoder_submission_probs.csv")

print("\nOlasılık dağılımı özeti:")
print(prob_df["prob"].describe())

# Hedef pozitif oranlarına (leaderboard probing'den öğrendiğimiz %26-30) denk gelen threshold'lar
sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
print("\nHedef pozitif oranlarına göre submission dosyaları üretiliyor:")
for target_rate in [0.20, 0.26, 0.28, 0.30, 0.35, 0.44]:
    if target_rate == 0.44:
        th = 0.44  # modelin kendi internal-F1-optimal threshold'u referans olarak
        label = "internal_best"
    else:
        th = float(np.quantile(all_probs, 1 - target_rate))
        label = f"rate{int(target_rate*100)}"
    preds = (all_probs >= th).astype(int)
    actual_rate = preds.mean()
    sub = pd.DataFrame({"id": prob_df["id"], "prediction": preds})
    sub = sub.set_index("id").loc[sample["id"]].reset_index()
    out_path = f"{DATA_DIR}/submission_ce_{label}.csv"
    sub.to_csv(out_path, index=False)
    print(f"  {label}: threshold={th:.3f}  gerçek_oran=%{100*actual_rate:.1f}  -> {out_path}")

print(f"\nToplam süre: {time.time()-t0:.1f}s")
