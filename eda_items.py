import pandas as pd

DATA_DIR = "data"
CHUNK_SIZE = 100_000

# ---- Sayaçlar / biriktiriciler ----
total_rows = 0
category_counts = {}
brand_counts = {}
gender_counts = {}
age_group_counts = {}
title_lengths = []
attribute_key_counts = {}
sample_attr_rows = []

first_chunk = True

for chunk in pd.read_csv(f"{DATA_DIR}/items.csv", chunksize=CHUNK_SIZE):
    total_rows += len(chunk)

    # Kategori dağılımı
    for cat, cnt in chunk["category"].value_counts().items():
        category_counts[cat] = category_counts.get(cat, 0) + cnt

    # Marka dağılımı
    for brand, cnt in chunk["brand"].value_counts(dropna=False).items():
        brand_counts[brand] = brand_counts.get(brand, 0) + cnt

    # Gender dağılımı
    for g, cnt in chunk["gender"].value_counts(dropna=False).items():
        gender_counts[g] = gender_counts.get(g, 0) + cnt

    # Age group dağılımı
    for ag, cnt in chunk["age_group"].value_counts(dropna=False).items():
        age_group_counts[ag] = age_group_counts.get(ag, 0) + cnt

    # Title uzunlukları (kelime sayısı)
    title_lengths.extend(chunk["title"].fillna("").str.split().str.len().tolist())

    # Attributes: "anahtar: değer, anahtar: değer, ..." formatını parse et
    for attr_str in chunk["attributes"].dropna().head(2000):  # her chunk'tan örnek al, hız için sınırlı
        pairs = [p.strip() for p in str(attr_str).split(",")]
        for p in pairs:
            if ":" in p:
                key = p.split(":", 1)[0].strip()
                attribute_key_counts[key] = attribute_key_counts.get(key, 0) + 1

    if first_chunk:
        sample_attr_rows = chunk["attributes"].dropna().head(3).tolist()
        first_chunk = False

    print(f"İşlenen satır: {total_rows}", end="\r")

print(f"\n\nToplam satır: {total_rows}")

print("\n=== KATEGORİ (Top 15) ===")
for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1])[:15]:
    print(f"{cnt:>8}  {cat}")
print(f"Toplam unique kategori: {len(category_counts)}")

print("\n=== MARKA (Top 15) ===")
for brand, cnt in sorted(brand_counts.items(), key=lambda x: -x[1])[:15]:
    print(f"{cnt:>8}  {brand}")
print(f"Toplam unique marka: {len(brand_counts)}")

print("\n=== GENDER ===")
for g, cnt in sorted(gender_counts.items(), key=lambda x: -x[1]):
    print(f"{cnt:>8}  {g}")

print("\n=== AGE GROUP ===")
for ag, cnt in sorted(age_group_counts.items(), key=lambda x: -x[1]):
    print(f"{cnt:>8}  {ag}")

print("\n=== TITLE UZUNLUĞU (kelime sayısı) ===")
title_series = pd.Series(title_lengths)
print(title_series.describe())

print("\n=== ATTRIBUTES: EN SIK GEÇEN ANAHTARLAR (örneklem üzerinden, Top 20) ===")
for key, cnt in sorted(attribute_key_counts.items(), key=lambda x: -x[1])[:20]:
    print(f"{cnt:>6}  {key}")

print("\n=== ÖRNEK ATTRIBUTES SATIRLARI ===")
for row in sample_attr_rows:
    print("-", row)
