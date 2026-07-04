# Trendyol E-Ticaret Yarışması 2026 — Arama Alaka Düzeyi Tahmini

Kaggle yarışması: [trendyol-e-ticaret-yarismasi-2026-kaggle](https://www.kaggle.com/competitions/trendyol-e-ticaret-yarismasi-2026-kaggle)

## Problem

Verilen bir (arama terimi, ürün) çifti için ürünün terimle alakalı (1) mi alakasız (0) mı olduğunu tahmin eden bir binary classification modeli.

**Kritik nokta:** Submission setindeki arama terimlerinin **%100'ü** training verisinde hiç görülmemiş (tam cold-start). Bu yüzden term_id/item_id hash'leri kullanılamaz — model tamamen **metin tabanlı** (query ↔ ürün başlığı/kategori/özellikleri) bir alaka skoru öğrenmeli.

## Kurulum

```bash
python -m pip install kaggle pandas scikit-learn lightgbm joblib
```

Kaggle API kimlik doğrulaması için: `kaggle.com/settings/api` üzerinden bir API token oluşturup talimatlara göre `~/.kaggle/` altına kaydet.

## Veri İndirme

```bash
python -m kaggle competitions download -c trendyol-e-ticaret-yarismasi-2026-kaggle
unzip trendyol-e-ticaret-yarismasi-2026-kaggle.zip -d data
```

`data/` klasörü `.gitignore` ile hariç tutulmuştur, her takım üyesi veriyi kendi Kaggle hesabından indirmelidir.

## Pipeline — Çalıştırma Sırası

| # | Script | Ne yapar | Çıktı |
|---|--------|----------|-------|
| 1 | `eda.py` | terms.csv ve training_pairs.csv temel keşfi | - |
| 2 | `eda_coldstart.py` | training/submission term-item örtüşme analizi | - |
| 3 | `eda_items.py` | items.csv kategori/marka/gender/age dağılımları | - |
| 4 | `eda_attributes.py` | attributes kolonunun tam key analizi | - |
| 5 | `build_item_features.py` | attributes'ı yapılandırılmış kolonlara + ham metin bloğuna ayırır | `data/items_features.csv` |
| 6 | `build_negative_samples.py` | TF-IDF tabanlı hard negative mining (1:4 pozitif:negatif oranı) | `data/training_full.csv` |
| 7 | `build_train_features.py` | TF-IDF cosine similarity + metin/kategori eşleşme feature'ları (training) | `data/training_features.csv`, `data/tfidf_vectorizer.joblib`, `data/item_tfidf_matrix.npz` |
| 8 | `train_lgbm.py` | LightGBM binary classifier eğitimi + macro-F1 threshold tuning | `data/lgbm_model.txt`, `data/best_threshold.txt` |
| 9 | `build_submission_features.py` | Aynı feature pipeline'ı submission_pairs.csv için üretir | `data/submission_features.csv` |
| 10 | `predict_submission.py` | Modeli submission'a uygular, Kaggle formatında dosya üretir | `data/submission.csv` |

## Kaggle'a Yükleme

```bash
python -m kaggle competitions submit -c trendyol-e-ticaret-yarismasi-2026-kaggle -f data/submission.csv -m "Açıklama"
```

## Yöntem Özeti

- **Negatif örnekleme:** DPR/ANCE gibi endüstri-standardı retrieval çalışmalarında kullanılan BM25/TF-IDF tabanlı hard negative mining. Her terim için TF-IDF ile en yakın 50 ürün bulunur, ilk 10 atlanır (yanlış-negatif riski), 11-50 arası hard negative olarak kullanılır. Buna aynı kategoriden ve tam rastgele negatifler eklenir.
- **Model:** LightGBM (GBDT) — Amazon'un ESCI (query-product relevance) araştırmasında ve benzer e-ticaret arama sistemlerinde kullanılan, tabular feature'lar için endüstri standardı yaklaşım.
- **Feature'lar:** TF-IDF cosine similarity, kelime örtüşme oranı, marka/kategori/renk/cinsiyet eşleşme sinyalleri.
- **Validation:** Terim bazlı train/val split (satır bazlı değil) — çünkü submission'daki terimler training'de hiç görülmemiş, bu yüzden validation skorunun gerçek cold-start performansını yansıtması gerekiyor.

## Güncel Sonuç

Validation macro F1: **0.7204** (threshold = 0.35)

En önemli feature'lar: `tfidf_cosine` > `word_overlap_ratio` > `brand_in_query` > `category_token_overlap_ratio`
