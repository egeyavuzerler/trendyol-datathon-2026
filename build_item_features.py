import pandas as pd
import re
import unicodedata

DATA_DIR = "data"
CHUNK_SIZE = 100_000
OUTPUT_PATH = f"{DATA_DIR}/items_features.csv"

# EDA'da en sık geçen ~35 key (gürültülü uzun kuyruk hariç, temiz ve anlamlı olanlar)
TOP_KEYS = [
    "renk", "color detail", "menşei", "materyal", "yıkama talimatı", "desen",
    "ortam", "kumaş tipi", "sürdürülebilirlik detayı", "materyal bileşeni",
    "özellik", "kalıp", "kutu durumu", "boy", "ek özellik", "parça sayısı",
    "siluet", "koleksiyon", "yaka tipi", "kol boyu", "kol tipi",
    "garanti süresi", "bakım talimatları (genel)", "cep", "tema / stil",
    "paket içeriği", "dokuma tipi", "ürün tipi", "sezon", "topuk boyu",
    "persona", "ürün detayı", "kemer/kuşak durumu", "topuk tipi",
    "boyut/ebat", "dış materyal", "bağlama şekli", "saya materyali",
    "taban materyali",
]


def slugify_key(key: str) -> str:
    """'color detail' -> 'attr_color_detail', 'yaka tipi' -> 'attr_yaka_tipi' gibi güvenli kolon adı üretir."""
    key = unicodedata.normalize("NFKD", key)
    key = key.encode("ascii", "ignore").decode("ascii")
    key = re.sub(r"[^a-zA-Z0-9]+", "_", key.strip().lower()).strip("_")
    return f"attr_{key}"


TOP_KEYS_SLUG = {k: slugify_key(k) for k in TOP_KEYS}

# Aynı parse fonksiyonu (EDA scriptiyle tutarlı)
_pattern = re.compile(r'([a-zçğıöşü0-9 &()/.-]{2,60}?):\s*')


def parse_attributes(attr_str):
    if not isinstance(attr_str, str) or not attr_str.strip():
        return {}
    matches = list(_pattern.finditer(attr_str))
    result = {}
    for i, m in enumerate(matches):
        key = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(attr_str)
        value = attr_str[start:end].strip().rstrip(",").strip()
        if key:
            result[key] = value
    return result


def build_raw_text(row):
    """Model için tek bir metin bloğu: title + category + brand + ham attributes."""
    parts = [
        str(row["title"]) if pd.notna(row["title"]) else "",
        str(row["category"]).replace("/", " ") if pd.notna(row["category"]) else "",
        str(row["brand"]) if pd.notna(row["brand"]) else "",
        str(row["gender"]) if pd.notna(row["gender"]) and row["gender"] != "unknown" else "",
        str(row["age_group"]) if pd.notna(row["age_group"]) and row["age_group"] != "unknown" else "",
        str(row["attributes"]) if pd.notna(row["attributes"]) else "",
    ]
    return " ".join(p for p in parts if p).strip()


first_chunk = True
total_rows = 0

for chunk in pd.read_csv(f"{DATA_DIR}/items.csv", chunksize=CHUNK_SIZE):
    # --- Katman 1: yapılandırılmış attribute kolonları ---
    parsed_list = chunk["attributes"].apply(parse_attributes)
    for key, col_name in TOP_KEYS_SLUG.items():
        chunk[col_name] = parsed_list.apply(lambda d: d.get(key, ""))

    # --- Katman 2: ham metin bloğu ---
    chunk["raw_text"] = chunk.apply(build_raw_text, axis=1)

    # Çıktıya yazılacak kolonlar
    out_cols = ["item_id", "title", "category", "brand", "gender", "age_group", "raw_text"] + list(TOP_KEYS_SLUG.values())
    chunk_out = chunk[out_cols]

    chunk_out.to_csv(
        OUTPUT_PATH,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False,
    )
    first_chunk = False
    total_rows += len(chunk)
    print(f"İşlenen satır: {total_rows}", end="\r")

print(f"\n\nBitti. Toplam {total_rows} satır -> {OUTPUT_PATH}")

# Kontrol için birkaç satır göster
preview = pd.read_csv(OUTPUT_PATH, nrows=3)
print("\n=== ÖNİZLEME ===")
print(preview.T)
