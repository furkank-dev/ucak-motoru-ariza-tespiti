# Uçak motoru arıza tespiti (NASA C-MAPSS FD001)

Bir turbofan motorunun sensör ölçümlerinden "arızaya yakın" (kalan ömür ≤ 30 döngü) olup olmadığını sınıflandıran Yapay Zekâ dersi projesi.

- Ne yapıldı, sonuçlar, dosya envanteri: `YAPILANLAR.md`
- Sunum ve sözlü için anlatım: `ANLATIM.md`
- Etiketlerin nasıl kurulduğu: `ETIKETLEME.md`
- Veri kaynağı, atıf, SHA-256: `data/README.md`

## Kurulum
Python 3 ve internet bağlantısı gerekir (veri ilk çalıştırmada indirilir). Paketler sanal ortama kurulur, sistem pip'ine dokunulmaz.

```bash
cd ucak-motoru-ariza-tespiti
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Çalıştırma sırası
Scriptler bu sırayla çalıştırılmalı. Her biri bir öncekinin çıktısını kullanır.

```bash
.venv/bin/python scripts/01_indir_ve_etiketle.py   # veriyi indirir, doğrular, etiketler -> data/raw, data/processed
.venv/bin/python scripts/02_inceleme.py            # keşifsel grafikler ve tablolar -> outputs/figures, outputs/tables
.venv/bin/python scripts/03_baz_modeller.py        # LR ve RF, 5 katlı motor bazlı CV + resmî test -> outputs/
.venv/bin/python scripts/04_etiket_tamamla.py      # test tüm döngüler etiketi + FD002-FD004 yedek csv'leri -> data/processed/ (modelde kullanılmaz)
.venv/bin/python scripts/05_gelismis_modeller.py   # XGBoost, SVM, MLP: iç içe motor bazlı CV + GridSearchCV + resmî test -> outputs/
.venv/bin/python scripts/06_esik_duyarliligi.py    # beş model, ESIK 20/30/50 (03 ve 05'in yöntemiyle) -> outputs/
.venv/bin/python scripts/07_rnn.py                 # RNN ve LSTM (son 30 döngü), aynı katlar + sızıntı ve pencere kontrolleri -> outputs/
```

Toplam süre birkaç dakikadan az: indirme yaklaşık 12 MB, baz modeller yaklaşık 10 saniye, `05` 8 çekirdekte yaklaşık 2 dakika, `06` yaklaşık 6 dakika, `07` yaklaşık 2 dakika. `torch` için CPU sürümü yeterli: `.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu`. `01` scripti `data/raw/` altında dosyaları bulursa yeniden indirmez. Yeniden indirmek için `data/raw/` klasörünü boşaltın.

## Klasörler
```
data/raw/          indirilen orijinal zip ve FD001 txt dosyaları (git'e girmez)
data/processed/    etiketli eğitim csv'si ve test son döngü csv'si (git'e girmez)
scripts/           ortak.py + 7 adım scripti
outputs/figures/   png grafikler
outputs/tables/    csv tablolar (sonuclar_baz.csv, sonuclar_gelismis.csv, tum_modeller_karsilastirma.csv dahil)
```

## Veri atfı
A. Saxena and K. Goebel (2008), "Turbofan Engine Degradation Simulation Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
