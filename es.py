import pandas as pd

DATA_DIR = "data"

# Küçük dosyalar - direkt okunabilir
terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
training_pairs = pd.read_csv(f"{DATA_DIR}/training_pairs.csv")

print("=== TERMS ===")
print(terms.shape)
print(terms.head())

print("\n=== TRAINING PAIRS ===")
print(training_pairs.shape)
print(training_pairs.head())
print("\nLabel dağılımı:")
print(training_pairs["label"].value_counts())

print("\nUnique term sayısı (training_pairs içinde):", training_pairs["term_id"].nunique())
print("Unique item sayısı (training_pairs içinde):", training_pairs["item_id"].nunique())
print("\nTerim başına ortalama pozitif ürün sayısı:")
print(training_pairs.groupby("term_id").size().describe())
