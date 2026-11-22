# Veri: NASA C-MAPSS FD001

**Atıf:** A. Saxena and K. Goebel (2008), "Turbofan Engine Degradation Simulation Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

## Çalışan indirme kaynağı
Kaynaklar `scripts/01_indir_ve_etiketle.py` içinde sırayla denenir. **1. kaynak çalıştı:**

`https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`

(HTTP 200, 12.429.152 bayt). Dış zip'in içinde `CMAPSSData.zip` var. FD001 dosyaları bu iç zip'ten çıkarıldı. 2, 3 ve 4. kaynaklar ilk kaynak çalıştığı için **hiç denenmedi**. Hugging Face ve GitHub kaynaklarının dosya yolları (`resolve/main/<dosya>`, `master/<dosya>`) tahmine dayanıyor ve doğrulanmadı.

Orijinal zip `data/raw/` altında duruyor. Bu klasör git'e girmez (`.gitignore`). Script çalışınca indirilen kaynağın adı `data/raw/kaynak.txt` dosyasına yazılır.

## SHA-256 (2026-11-22, iki ayrı indirmede aynı çıktı)
| Dosya | SHA-256 |
|---|---|
| `6._Turbofan_Engine_Degradation_Simulation_Data_Set.zip` (orijinal zip; adındaki `+` işaretleri `_` yapıldı) | `c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2` |
| `train_FD001.txt` | `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8` |
| `test_FD001.txt` | `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851` |
| `RUL_FD001.txt` | `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca` |

## Biçim
Dosyalar boşlukla ayrılmış ve başlıksız. Her satırda 26 sütun var: `unit, cycle, os1, os2, os3, s1 ... s21`. Satır sonundaki boşluklar yüzünden oluşan boş sütunlar okunurken atılır.
