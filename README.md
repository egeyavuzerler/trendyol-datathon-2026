# Trendyol E-Ticaret Yarışması 2026 — Arama Alaka Düzeyi Tahmini

Kaggle yarışması: [trendyol-e-ticaret-yarismasi-2026-kaggle](https://www.kaggle.com/competitions/trendyol-e-ticaret-yarismasi-2026-kaggle)

## Problem

Verilen bir (arama terimi, ürün) çifti için ürünün terimle alakalı (1) mi alakasız (0) mı olduğunu tahmin eden bir binary classification modeli.

**Kritik nokta:** Submission setindeki arama terimlerinin **%100'ü** training verisinde hiç görülmemiş (tam cold-start). Bu yüzden term_id/item_id hash'leri kullanılamaz — model tamamen **metin tabanlı** (query ↔ ürün başlığı/kategori/özellikleri) bir alaka skoru öğrenmeli.

## Kurulum

```bash
python -m pip install kaggle pandas scikit-learn lightgbm joblib optuna sentence-transformers
```

Kaggle API kimlik doğrulaması için: `kaggle.com/settings/api` üzerinden bir API token oluşturup talimatlara göre `~/.kaggle/` altına kaydet.

## Veri İndirme

```bash
python -m kaggle competitions download -c trendyol-e-ticaret-yarismasi-2026-kaggle
unzip trendyol-e-ticaret-yarismasi-2026-kaggle.zip -d data
```

`data/` klasörü `.gitignore` ile hariç tutulmuştur, her takım üyesi veriyi kendi Kaggle hesabından indirmelidir.

## Pipeline — Çalıştırma Sırası

### 1) Keşif ve Ön Hazırlık

| # | Script | Ne yapar | Çıktı |
|---|--------|----------|-------|
| 1 | `eda.py` | terms.csv ve training_pairs.csv temel keşfi | - |
| 2 | `eda_coldstart.py` | training/submission term-item örtüşme analizi (cold-start tespiti) | - |
| 3 | `eda_items.py` | items.csv kategori/marka/gender/age dağılımları | - |
| 4 | `eda_attributes.py` | attributes kolonunun tam key analizi | - |
| 5 | `build_item_features.py` | attributes'ı yapılandırılmış kolonlara + ham metin bloğuna (raw_text, attributes_raw) ayırır | `data/items_features.csv` |
| 6 | `build_negative_samples.py` | TF-IDF tabanlı hard negative mining (top-150 havuz, 1:4 pozitif:negatif oranı) | `data/training_full.csv` |

### 2) Feature Engineering (v2: BM25 + Alan-Bazı Skorlar)

| # | Script | Ne yapar | Çıktı |
|---|--------|----------|-------|
| 7 | `build_train_features_v2.py` | TF-IDF (raw/title/category/attributes) + BM25 + genişletilmiş attribute eşleşmeleri | `data/training_features_v2.csv` + vectorizer/matris dosyaları |
| 8 | `build_submission_features_v2.py` | Aynı feature pipeline'ı submission_pairs.csv için üretir | `data/submission_features_v2.csv` |

### 3) Semantik Embedding (v3)

| # | Script | Ne yapar | Çıktı |
|---|--------|----------|-------|
| 9 | `build_embeddings.py` | Çok dilli sentence-transformer (`paraphrase-multilingual-MiniLM-L12-v2`) ile ürün başlığı + query embedding'lerini çıkarır | `data/item_embeddings.npy`, `data/term_embeddings.npy` |
| 10 | `add_semantic_feature_train.py` | Embedding cosine similarity'yi training feature setine ekler | `data/training_features_v3.csv` |
| 11 | `add_semantic_feature_submission.py` | Aynısını submission için yapar | `data/submission_features_v3.csv` |

### 4) Model Eğitimi ve Tahmin

| # | Script | Ne yapar | Çıktı |
|---|--------|----------|-------|
| 12 | `tune_lgbm_optuna.py` | LightGBM hiperparametrelerini Optuna ile optimize eder (3-fold CV, macro F1 hedefi) | `data/best_lgbm_params.json` |
| 13 | `train_final_v3_and_predict.py` | Optuna parametreleriyle 5-fold CV ensemble eğitir, OOF macro F1 raporlar, submission üretir | `data/submission_v3.csv` |

Ara/eski sürümler (`build_train_features.py`, `train_lgbm.py`, `train_lgbm_cv.py`, `train_final_and_predict.py`) karşılaştırma amacıyla repoda tutulmuştur.

### 5) Teşhis (CV-Leaderboard Uyuşmazlığı) ve Cross-Encoder (v4)

İlk Kaggle submission'ı (v3, OOF macro F1=0.7175) public leaderboard'da sadece **0.65** aldı — belirgin bir CV-LB farkı. Bu fark araştırılmadan yeni model eklenmedi:

| # | Script | Ne yapar | Sonuç |
|---|--------|----------|-------|
| 14 | `diagnose_item_leakage.py` | GroupKFold'da item-level leakage olup olmadığını test eder | **Leakage yok** — görülmemiş item'larda skor daha yüksek çıktı |
| 15 | `diagnose_distribution_shift.py` | Training pozitif/negatif dağılımını submission dağılımıyla karşılaştırır | Submission'ın medyan tfidf_cosine'ı **0.0** — negatif örneklememizden bile daha "kolay" negatifler içeriyor |
| 16 | `generate_threshold_variants.py` / `generate_single_threshold.py` | Farklı threshold'larda submission dosyaları üretir (leaderboard probing için) | threshold=0.20 (%26.6 relevant) → **public 0.69** (threshold=0.34'ün 0.65'inden çok daha iyi) |

**Sonuç:** Gerçek pozitif oranı bizim varsaydığımız ~%20'den belirgin şekilde yüksek (~%26-30 civarı en iyi sonucu verdi). Ayrıca feature-engineering + GBDT yaklaşımının bir tavana yaklaştığı görüldü.

Bu bulgularla, **BERTurk tabanlı cross-encoder** yaklaşımına geçildi — query ve ürün metnini ayrı ayrı vektörleyip karşılaştırmak yerine, ikisini birlikte transformer'a verip uçtan uca fine-tune etme:

| # | Script | Ne yapar | Çıktı |
|---|--------|----------|-------|
| 17 | `train_cross_encoder.py` | `dbmdz/bert-base-turkish-cased` modelini query+ürün metni çiftleri üzerinde fine-tune eder. Negatif karışımı, diagnostic bulgusuna göre düşük-benzerlikli (kolay) örneklere ağırlıklı (%50 low/%30 mid/%20 high tfidf_cosine kovaları) | `data/cross_encoder_model/` |
| 18 | `predict_cross_encoder_submission.py` | Fine-tune edilmiş modeli submission_pairs.csv üzerinde optimize edilmiş batch inference ile çalıştırır, birden fazla threshold için submission dosyası üretir | `data/cross_encoder_submission_probs.csv`, `data/submission_ce_*.csv` |

**Cross-encoder sonucu:** Internal validation macro F1 = **0.8697** — LightGBM+feature-engineering yaklaşımından (0.7175 OOF) çok büyük bir sıçrama. Bu, literatürdeki bulguyu doğruluyor: cross-encoder mimarisi, ayrık benzerlik skorları + GBDT kombinasyonundan search-relevance görevlerinde sistematik olarak daha güçlü.



```bash
python -m kaggle competitions submit -c trendyol-e-ticaret-yarismasi-2026-kaggle -f data/submission_v3.csv -m "Açıklama"
```

## Yöntem Özeti

- **Negatif örnekleme:** DPR/ANCE gibi endüstri-standardı retrieval çalışmalarında kullanılan BM25/TF-IDF tabanlı hard negative mining. Her terim için TF-IDF ile en yakın 150 ürün bulunur, ilk 15 atlanır (yanlış-negatif riski), 15-150 arası hard negative olarak kullanılır. Buna aynı kategoriden ve tam rastgele negatifler eklenir (1:4 pozitif:negatif oranı).
- **Lexical skorlar:** TF-IDF cosine similarity (raw_text, title, category, attributes alanları ayrı ayrı) + BM25 (Elasticsearch/Solr/Bing'in varsayılan sıralama fonksiyonu, kısa metin eşleştirmede TF-IDF'ten daha güçlü).
- **Semantik skor:** Çok dilli sentence-transformer embedding'leri ile cosine similarity — eş anlamlı/yakın anlamlı ifadeleri (örn. "spor ayakkabı" ↔ "koşu ayakkabısı") yakalamak için (Amazon ESCI ve benzeri güncel arama-relevance sistemlerinde lexical+semantic kombinasyonu standart pratiktir).
- **Diğer feature'lar:** kelime örtüşme oranı, marka/kategori/renk/materyal/desen/cinsiyet eşleşme sinyalleri.
- **Model:** LightGBM (GBDT) — Amazon'un ESCI (query-product relevance) araştırmasında kullanılan, tabular feature'lar için endüstri standardı yaklaşım.
- **Validation:** Terim bazlı 5-fold GroupKFold (satır bazlı değil) — çünkü submission'daki terimler training'de hiç görülmemiş, validation skorunun gerçek cold-start performansını yansıtması için gerekli. Nihai submission, 5 fold modelinin ortalaması (ensemble).
- **Hiperparametre optimizasyonu:** Optuna ile 3-fold CV üzerinde 40 deneme (num_leaves, learning_rate, feature/bagging_fraction, min_child_samples, L1/L2 regularizasyon).

## Güncel Sonuç

**v4 (Cross-Encoder, BERTurk):** Internal validation macro F1 = **0.8697**

**v3 (Feature-engineering + LightGBM):** OOF macro F1 = 0.7175, ancak public LB = 0.65 (CV-LB uyuşmazlığı — bkz. teşhis bölümü)

En önemli feature'lar (v3, LightGBM): `tfidf_cosine` > `bm25_score` > `tfidf_category_cosine` > `brand_in_query` > `word_overlap_ratio` > ... > `semantic_cosine`

### Sürüm geçmişi

| Sürüm | Açıklama | Skor |
|---|---|---|
| v1 | TF-IDF cosine + basit eşleşme feature'ları, tek train/val split | OOF 0.7204 (tek split) |
| v2 | + BM25, alan-bazı TF-IDF, genişletilmiş negatif havuzu, 5-fold CV | OOF 0.7165 |
| v2 + Optuna | Hiperparametre optimizasyonu | OOF 0.7165 |
| v3 | + Semantik embedding (multilingual sentence-transformer) | OOF 0.7175 / **public LB 0.65** (threshold=0.34) |
| v3 + threshold probing | Aynı model, leaderboard'dan kalibre edilmiş threshold (0.20) | **public LB 0.69** |
| v4 | BERTurk cross-encoder (query+ürün metni birlikte fine-tune), diagnostic-bilgili negatif karışımı | internal val macro F1 **0.8697** |

### Öğrenilen Dersler

- **CV skoru ile public LB skoru arasında büyük fark olabilir** — kendi ürettiğimiz sentetik negatiflerle eğitip değerlendirmek, gerçek test dağılımını yansıtmayabilir. Model eklemeden önce bu farkı teşhis etmek kritik.
- **Threshold kalibrasyonu, macro F1'de model kalitesi kadar önemli olabilir** — aynı model, farklı threshold'larla 0.65 ile 0.69 arası skor verdi.
- **Cross-encoder (uçtan uca fine-tune edilmiş transformer), feature-engineering + GBDT'den ciddi şekilde daha güçlü** — bu, arama-relevance literatüründe (MS MARCO, Amazon ESCI) tekrar tekrar doğrulanan bir bulgu.

## Kaggle'a Yükleme
