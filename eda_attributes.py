import pandas as pd
import re

DATA_DIR = "data"
CHUNK_SIZE = 100_000

key_doc_counts = {}   # kaç farklı üründe bu key geçiyor
key_value_samples = {}  # her key için birkaç örnek value
total_rows = 0
rows_with_attributes = 0

def parse_attributes(attr_str):
    """'anahtar: değer, anahtar: değer' formatını parse eder.
    Değerlerin içinde virgül olabileceği için basit split yerine
    'anahtar:' desenini bulup aradaki metni value olarak alıyoruz."""
    if not isinstance(attr_str, str) or not attr_str.strip():
        return {}

    # "kelime(ler): " desenini anahtar olarak yakala
    # Anahtarlar genelde küçük harf + boşluk + parantez içerebilir, sonra ':' gelir
    pattern = re.compile(r'([a-zçğıöşü0-9 &()/.-]{2,60}?):\s*')
    matches = list(pattern.finditer(attr_str))

    result = {}
    for i, m in enumerate(matches):
        key = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(attr_str)
        value = attr_str[start:end].strip().rstrip(",").strip()
        if key:
            result[key] = value
    return result


for chunk in pd.read_csv(f"{DATA_DIR}/items.csv", chunksize=CHUNK_SIZE):
    total_rows += len(chunk)
    for attr_str in chunk["attributes"]:
        if isinstance(attr_str, str) and attr_str.strip():
            rows_with_attributes += 1
            parsed = parse_attributes(attr_str)
            for k, v in parsed.items():
                key_doc_counts[k] = key_doc_counts.get(k, 0) + 1
                if k not in key_value_samples:
                    key_value_samples[k] = []
                if len(key_value_samples[k]) < 3:
                    key_value_samples[k].append(v)
    print(f"İşlenen satır: {total_rows}", end="\r")

print(f"\n\nToplam satır: {total_rows}")
print(f"Attributes dolu olan satır: {rows_with_attributes} (%{100*rows_with_attributes/total_rows:.1f})")
print(f"\nToplam unique attribute key sayısı: {len(key_doc_counts)}")

print("\n=== TÜM KEY'LER, DÜŞEN ÜRÜN SAYISINA GÖRE (Top 40) ===")
sorted_keys = sorted(key_doc_counts.items(), key=lambda x: -x[1])
for key, cnt in sorted_keys[:40]:
    pct = 100 * cnt / total_rows
    samples = key_value_samples.get(key, [])
    print(f"{cnt:>8} (%{pct:5.1f})  {key:<35} örnek: {samples}")

print(f"\n... ve {max(0, len(sorted_keys)-40)} key daha (uzun kuyruk)")

# Kaç key sadece 1-2 kez geçiyor (muhtemelen parse hatası / gürültü)
rare_keys = [k for k, c in key_doc_counts.items() if c <= 2]
print(f"\n1-2 kez geçen (muhtemelen gürültü) key sayısı: {len(rare_keys)}")
print("Örnekler:", rare_keys[:15])
