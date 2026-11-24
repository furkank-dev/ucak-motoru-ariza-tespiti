# Uçak motoru arıza tespiti (NASA C-MAPSS FD001)

Bir turbofan motorunun sensör ölçümlerinden "arızaya yakın" (kalan ömür ≤ 30 döngü) olup olmadığını sınıflandıran Yapay Zekâ dersi projesi. Beş klasik makine öğrenmesi modeli (Lojistik Regresyon, Random Forest, XGBoost, SVM, MLP) ve iki tekrarlayan sinir ağı (RNN, LSTM) motor bazlı çapraz doğrulama ve NASA'nın resmî test kümesiyle karşılaştırılıyor. Etiket eşiğinin (20 / 30 / 50 döngü) etkisi de inceleniyor.

- Ne yapıldı, sonuçlar, kararlar, dosya envanteri: `YAPILANLAR.md`
- Sunum ve sözlü için anlatım: `ANLATIM.md`
- Etiketlerin nasıl kurulduğu: `ETIKETLEME.md`
- Veri kaynağı, atıf, SHA-256: `data/README.md`

## Sonuçlar kısaca
ESIK = 30, karar eşiği 0,5. CV: 5 katlı motor bazlı çapraz doğrulama (ortalama ± std); test: 100 test motorunun son döngüsü (25 pozitif).

| Model | Girdi | CV F1 | CV Recall | Test F1 | Test Recall |
|---|---|---|---|---|---|
| Lojistik Regresyon | tek döngü | 0,823 ± 0,036 | 0,947 | 0,846 | 0,88 |
| Random Forest | tek döngü | 0,855 ± 0,026 | 0,897 | 0,826 | 0,76 |
| XGBoost | tek döngü | 0,846 ± 0,026 | 0,879 | 0,826 | 0,76 |
| SVM (RBF) | tek döngü | 0,821 ± 0,038 | 0,956 | 0,830 | 0,88 |
| MLP | tek döngü | 0,820 ± 0,037 | 0,951 | 0,880 | 0,88 |
| RNN | son 30 döngü | 0,838 ± 0,035 | 0,958 | 0,880 | 0,88 |
| LSTM | son 30 döngü | 0,891 ± 0,020 | 0,964 | 0,962 | 1,00 |

- Tek döngüye bakan beş klasik model arasındaki farklar katlar arası dalgalanmadan küçük. Eşik 20 ya da 50 yapıldığında da sıralama değişmiyor; RF her eşikte CV F1'de birinci.
- LSTM 5 katın 5'inde de en iyi klasik modeli geçiyor. Aynı LSTM'e tek döngü verildiğinde ise F1 0,824'e iniyor. Yani kazanç mimariden değil, son 30 döngünün geçmişini görmesinden geliyor.
- Karıştırılmış etiket testlerinde AUC yaklaşık 0,5 çıkıyor; sızıntı bulunmadı.

Bütün sayılar `outputs/tables/` altındaki csv'lerden. Ayrıntılı yorum ve sınırlılıklar `YAPILANLAR.md`'de.

## Kurulum
Python 3 ve internet bağlantısı gerekir (veri ilk çalıştırmada indirilir). Paketler sanal ortama kurulur, sistem pip'ine dokunulmaz. Geliştirme Python 3.14 ile yapıldı. Sürümler `YAPILANLAR.md`'de; `requirements.txt`'de sabitlenmedi.

```bash
cd ucak-motoru-ariza-tespiti
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # önce CPU sürümü; yoksa çok büyük CUDA sürümü iner
.venv/bin/pip install -r requirements.txt
```

Ekran kartı gerekmiyor. Bütün modeller normal bir bilgisayarın işlemcisinde eğitiliyor.

## Çalıştırma sırası
Scriptler `scripts/` içinden ya da proje kökünden bu sırayla çalıştırılmalı:

```bash
.venv/bin/python scripts/01_indir_ve_etiketle.py   # veriyi indirir, doğrular, etiketler -> data/raw, data/processed
.venv/bin/python scripts/02_inceleme.py            # keşifsel grafikler ve tablolar -> outputs/
.venv/bin/python scripts/03_baz_modeller.py        # hep 0, LR, RF: 5 katlı motor bazlı CV + resmî test -> outputs/
.venv/bin/python scripts/04_etiket_tamamla.py      # test motorlarının tüm döngüleri etiketli + FD002-FD004 yedekleri -> data/processed/
.venv/bin/python scripts/05_gelismis_modeller.py   # XGBoost, SVM, MLP: iç içe CV + GridSearchCV + resmî test -> outputs/
.venv/bin/python scripts/06_esik_duyarliligi.py    # beş model, ESIK 20 / 30 / 50 -> outputs/
.venv/bin/python scripts/07_rnn.py                 # RNN ve LSTM (son 30 döngü) + sızıntı ve pencere kontrolleri -> outputs/
```

Bağımlılıklar: `05`, `03`'ün tablolarını okur. `06`, `03` ve `05`'in tablolarını okur ve ESIK = 30 sonuçlarının onlarla aynı olduğunu kontrol eder. `07`, `04`'ün ürettiği test tüm döngüler dosyasını ve `05`'in birleşik tablosunu kullanır.

Süreler (8 çekirdek): indirme yaklaşık 12 MB; `01`-`04` birkaç saniye; `05` yaklaşık 2 dakika; `06` yaklaşık 6 dakika; `07` yaklaşık 2 dakika. Toplamda 10 dakika civarı. `01` scripti `data/raw/` altında dosyaları bulursa yeniden indirmez; yeniden indirmek için `data/raw/` klasörünü boşaltın.

Bütün scriptler sabit tohumla (`RASTGELE_TOHUM = 42`) çalışır. Aynı makinede tekrar çalıştırınca tablolar birebir aynı çıkar.

## Klasörler
```
data/raw/          indirilen orijinal zip ve FD001 txt dosyaları (git'e girmez)
data/processed/    etiketli eğitim, test son döngü ve test tüm döngüler csv'leri; yedek/ altında FD002-FD004 (git'e girmez)
scripts/           ortak.py (sabitler, ortak fonksiyonlar) + 7 adım scripti
outputs/figures/   png grafikler (01-13)
outputs/tables/    csv tablolar
```

Önemli tablolar ve grafikler:

| Dosya | İçerik |
|---|---|
| `tables/tum_modeller_rnn_dahil.csv` | yedi model + hep 0, CV ve test, beş metrik |
| `tables/esik_duyarliligi.csv` | beş model, ESIK 20 / 30 / 50 |
| `tables/rnn_kontroller.csv` | karıştırılmış etiket ve pencere = 1 kontrolleri |
| `tables/en_iyi_parametreler.csv` | XGBoost / SVM / MLP için GridSearchCV'nin seçtiği ayarlar |
| `figures/10_model_karsilastirma.png` | beş klasik modelin karşılaştırması |
| `figures/11_esik_duyarliligi.png` | F1 ve Recall'un eşiğe göre değişimi |
| `figures/13_tum_modeller_rnn_dahil.png` | klasik modeller, RNN ve LSTM birlikte |

## Veri atfı
A. Saxena and K. Goebel (2008), "Turbofan Engine Degradation Simulation Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
