# Yaptıklarımız

Projede şimdiye kadar ne yaptığımızı, hangi sonuçları aldığımızı ve neden öyle karar verdiğimizi buraya yazıyoruz. Sayılar scriptlerin çıktılarından ve `outputs/` altındaki dosyalardan.

Kısaca: FD001 verisini indirip etiketledik, veriyi inceledik, iki baz model (Lojistik Regresyon, Random Forest) ve üç gelişmiş model (XGBoost, SVM, MLP) kurduk. Hepsi her şeye "sağlıklı" diyen modeli rahat geçiyor. CV'de F1'e göre en iyisi RF (0,8552 ± 0,0258), resmî testte MLP (0,8800). Beş modelin CV F1'leri 0,8195 ile 0,8552 arasında ve farklar standart sapmalardan küçük; yani gelişmiş modeller baz modellere belirgin bir üstünlük sağlamadı. Etiket eşiği 20/30/50 yapıldığında da sıralama pek değişmiyor (RF her eşikte CV F1'de birinci). Son olarak her satır için motorun son 30 döngüsünü gören bir RNN ve LSTM denedik. LSTM bütün klasik modelleri geçti (CV F1 0,8914 ± 0,0202, test F1 0,9615). Ama aynı LSTM'e tek döngü verildiğinde F1 klasik modellerin seviyesine (0,8242) iniyor; kazanç mimariden değil, geçmiş döngülerin bilgisinden geliyor.

## Terimler

| Terim | Anlamı |
|---|---|
| Döngü (cycle) | Motorun bir uçuşu. Veride her satır bir motorun bir döngüsü. |
| Motor (unit) | Motor numarası. Eğitimde 100, testte 100 ayrı motor var. |
| RUL | Kalan faydalı ömür: motor arızalanmadan önce kaç döngü kaldığı. |
| Etiket | Tahmin ettiğimiz şey: RUL ≤ 30 ise 1 (arızaya yakın), değilse 0. |
| ESIK | Etiketin eşiği, 30 döngü. Karşılaştırma için 20 ve 50 de var. |
| GroupKFold | Veriyi motorlara göre bölen çapraz doğrulama. Bir motorun bütün satırları aynı parçada kalır. |
| OOF tahmin | Bir satır için, o satırı eğitimde görmemiş modelin verdiği tahmin. |
| Sızıntı (leakage) | Modelin eğitimde gerçek hayatta elinde olmayacak bir bilgiyi görmesi. Sonuçları sahte şekilde yükseltir. |
| Pipeline | Ölçekleme ve modeli tek nesnede birleştirir, böylece ölçekleyici sadece eğitim verisini görür. |
| Precision | "Arızaya yakın" dediklerimizin ne kadarı gerçekten öyle: TP / (TP + FP). |
| Recall | Gerçekten arızaya yakın olanların ne kadarını yakaladık: TP / (TP + FN). |
| F1 | Precision ile Recall'un harmonik ortalaması. |
| ROC-AUC | Rastgele bir pozitif örneğe rastgele bir negatiften daha yüksek olasılık verme ihtimali. 0,5 yazı tura, 1 mükemmel. |
| TP / FP / FN / TN | Doğru pozitif / yanlış alarm / kaçırılan arıza / doğru negatif. |
| class_weight="balanced" | Az olan sınıftaki hatalara daha çok ceza verir. |

## Adım adım

### Kurulum
Sanal ortamı `python -m venv .venv` ile kurduk (Python 3.14.10). Kullandığımız sürümler: pandas 3.1.1, numpy 2.5.5, matplotlib 3.11.2, seaborn 0.13.2, scikit-learn 1.9.1, tabulate 0.10.0. `requirements.txt` içinde sürümleri sabitlemedik.

### 1. Veri ve etiket (`01_indir_ve_etiketle.py`)
Veriyi NASA'nın PHM S3 adresinden indirdik (12.429.152 bayt). Zip iç içe, içinde bir de `CMAPSSData.zip` var. Script içteki zip'i açıp sadece üç FD001 dosyasını alıyor. Scriptte üç yedek adres daha var ama ilk adres çalıştığı için onları hiç denemedik. Hugging Face ve GitHub yolları tahmine dayanıyor, ilk adres bir gün kaybolursa düzeltmek gerekebilir.

Veriyi iki kez ayrı ayrı indirdik, SHA-256 değerleri aynı çıktı (`data/README.md`). Sayıları `assert` ile kontrol ediyoruz:
- Eğitim: 20.631 satır, 100 motor. Test: 13.096 satır, 100 motor. RUL dosyası: 100 satır.
- Eğitim etiketleri: ESIK=30'da 3.100 pozitif / 17.531 negatif, ESIK=20'de 2.100 / 18.531, ESIK=50'de 5.100 / 15.531.
- Test (son döngü): ESIK=30'da 25 / 75, ESIK=20'de 16 / 84, ESIK=50'de 33 / 67.

### 2. İnceleme (`02_inceleme.py`)
5 grafik ve 7 tablo çıkardık. Sabit sütunları veriden hesapladık. Korelasyon tablosunun ilk sütununun adı boş çıkıyordu ("Unnamed: 0"), ona `sensor` adını verip scripti tekrar çalıştırdık.

### 3. Baz modeller (`03_baz_modeller.py`)
"Hep 0" modeli, Lojistik Regresyon ve Random Forest. 5 katlı motor bazlı CV, sonra 100 motorun hepsiyle eğitip resmî testte değerlendirme. Çalışması yaklaşık 9 saniye sürüyor.

4. katta ROC-AUC iki modelde de 0,99'u biraz geçti. Bunun sızıntı olup olmadığını anlamak için scripte `sizinti_incelemesi` adımını ekledik (aşağıda).

ROC grafiğindeki CV eğrisi birleştirilmiş OOF tahminlerden çiziliyor ama etiketinde ilk başta katların ortalaması (0,9895) yazıyordu. Etiketi çizilen eğriyle aynı olsun diye OOF AUC'ye çevirdik (LR 0,9886, RF 0,9880). `sonuclar_baz.csv` içindeki CV değerleri hâlâ katların ortalaması.

### 4. Etiketleri tamamlama (`04_etiket_tamamla.py`)
İki şey ekledik:
- `test_FD001_tum_donguler.csv`: test motorlarının sadece son döngüsü değil, bütün döngüleri etiketli. RUL geriye doğru hesaplanıyor: `RUL_FD001[motor] + (son döngü − döngü)`. 13.096 satır, RUL 7 ile 340 arasında. ESIK=30'da 332 pozitif / 12.764 negatif (%2,54). Son döngüler `RUL_FD001.txt` ve `test_FD001_son_dongu.csv` ile birebir tutuyor.
- `data/processed/yedek/`: FD002, FD003 ve FD004'ü aynı kuralla etiketledik. Zip zaten elimizde olduğu için tekrar indirmedik. Bunları modelde kullanmıyoruz, sadece ileride lazım olursa diye duruyorlar.

| Alt küme | Eğitim satır / motor | Test satır / motor | Eğitim ESIK=30 poz / neg | Test son döngü ESIK=20 / 30 / 50 poz | Motor ömrü |
|---|---|---|---|---|---|
| FD002 | 53.759 / 260 | 33.991 / 259 | 8.060 / 45.699 (%14,99) | 53 / 61 / 88 | 128-378 |
| FD003 | 24.720 / 100 | 16.596 / 100 | 3.100 / 21.620 (%12,54) | 14 / 20 / 29 | 145-525 |
| FD004 | 61.249 / 249 | 41.214 / 248 | 7.719 / 53.530 (%12,60) | 40 / 53 / 80 | 128-543 |

Bu adımdan sonra `03_baz_modeller.py`'yi tekrar çalıştırdık, `sonuclar_baz.csv` hiç değişmedi.

Etiketlemenin ayrıntılı anlatımı `ETIKETLEME.md` dosyasında.

### 5. Gelişmiş modeller (`05_gelismis_modeller.py`)
XGBoost, SVM (RBF çekirdek) ve MLP. Özellikler, katlar ve test 03 ile aynı (14 sensör, GroupKFold(5), ölçekleme pipeline içinde), böylece beş model aynı koşullarda karşılaştırılıyor. Plandaki adı `04` idi ama 04 numarası etiket tamamlamaya gittiği için bu script `05` oldu; eşik duyarlılığı `06` olacak. XGBoost'u (3.4.1) ortama bu adımda kurduk ve `requirements.txt`'ye ekledik. Çalışması 8 çekirdekte yaklaşık 1 dakika 40 saniye.

**Hiperparametre araması iç içe (nested) yapıldı.** Dış döngü 03'teki 5 kat. Her dış katta sadece o katın 80 eğitim motoru içinde `GroupKFold(3)` ile `GridSearchCV` çalışıyor (skor F1), en iyi ayar o 80 motorla fit edilip dışarıda bırakılan 20 motorda ölçülüyor. Böylece ayar seçerken doğrulama motorlarını hiç görmüyoruz; aramayı tüm veride yapıp aynı katlarda raporlasaydık sonuçlar iyimser çıkardı. Resmî test için arama 100 motorun tamamında tekrar yapıldı.

Izgaralar küçük tutuldu:

| Model | Aranan | Sabit |
|---|---|---|
| XGBoost | `max_depth` 3/5, `n_estimators` 200/400, `learning_rate` 0,05/0,1 (8 ayar) | `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42` |
| SVM | `C` 0,1/1/10, `gamma` scale/0,01/0,1 (9 ayar) | `kernel="rbf"`, `class_weight="balanced"` |
| MLP | gizli katman (32) / (64, 32), `alpha` 0,0001/0,01, `learning_rate_init` 0,001/0,01 (8 ayar) | `max_iter=500`, `early_stopping=True`, `n_iter_no_change=20`, `random_state=42` |

**Dengesizlik.** SVM'de LR/RF'deki gibi `class_weight="balanced"`. MLP `class_weight` desteklemediği için XGBoost ve MLP'ye aynı etkiyi yapan "balanced" örnek ağırlıklarını (`compute_sample_weight`) verdik. Ağırlıklar her aramada sadece o aramanın eğitim satırlarından hesaplanıyor.

**SVM'de olasılık yok.** `SVC(probability=True)` olasılığı ayrı bir iç CV ile tahmin ediyor ve `predict()` ile çelişebiliyor. Onun yerine `decision_function` kullandık: sınıf için eşik 0 (`predict()` ile aynı), ROC-AUC için skorun kendisi. ROC-AUC sadece sıralamaya baktığı için olasılık olmasına gerek yok.

**Seçilen ayarlar** (`en_iyi_parametreler.csv`):
- XGBoost: 6 aramanın 6'sında `max_depth=5, n_estimators=400, learning_rate=0.1`. Bu ızgaranın köşesi; daha derin/uzun ayarlar denenmedi.
- SVM: 5 dış katın hepsinde `C=1, gamma=0.01`, ama 100 motorla yapılan aramada `C=10, gamma=0.1`. İç CV F1'leri birbirine çok yakın (0,815-0,826), yani seçim kararsız ve ayarın sonuca etkisi küçük.
- MLP: katlarda (64, 32) / `alpha=0.01` ile (32) / `alpha=0.0001` arasında gidip geliyor. 100 motorda (32), `alpha=0.0001`, `learning_rate_init=0.001`.

**Çıktılar:** `sonuclar_gelismis.csv`, `sonuclar_gelismis_katlar.csv`, `en_iyi_parametreler.csv`, `karisiklik_matrisleri_gelismis.csv`, `zor_bolge_auc_gelismis.csv`, beş modeli birleştiren `tum_modeller_karsilastirma.csv` ve `kat_bazli_f1_tum_modeller.csv`; grafikler `08_karisiklik_matrisleri_gelismis.png`, `09_roc_egrileri_gelismis.png`, `10_model_karsilastirma.png`. Birleştirilmiş tablolar LR ve RF satırlarını `sonuclar_baz.csv`'den okuyor, yani 05'ten önce 03 çalışmış olmalı.

### 6. Eşik duyarlılığı (`06_esik_duyarliligi.py`)
Beş model ve hep 0, ESIK = 20, 30 ve 50 için. Yöntem 03 ve 05 ile birebir aynı; script 03 ve 05'in fonksiyonlarını içe aktarıyor. XGBoost, SVM ve MLP için iç içe arama her eşikte yeniden yapılıyor, çünkü etiket değişince en iyi ayar da değişebilir. Script, ESIK = 30 satırlarının `sonuclar_baz.csv` ve `sonuclar_gelismis.csv` ile birebir aynı çıktığını `assert` ile kontrol ediyor; aynı çıktı. Süre yaklaşık 6 dakika.

Çıktılar: `esik_duyarliligi.csv`, `esik_duyarliligi_parametreler.csv`, `11_esik_duyarliligi.png`.

CV F1 (5 kat ortalaması ± std):

| Model | ESIK 20 (%10,2 pozitif) | ESIK 30 (%15,0) | ESIK 50 (%24,7) |
|---|---|---|---|
| Lojistik Regresyon | 0,8076 ± 0,0355 | 0,8229 ± 0,0357 | 0,8382 ± 0,0301 |
| Random Forest | 0,8487 ± 0,0221 | 0,8552 ± 0,0258 | 0,8564 ± 0,0191 |
| XGBoost | 0,8469 ± 0,0236 | 0,8463 ± 0,0262 | 0,8484 ± 0,0204 |
| SVM (RBF) | 0,8107 ± 0,0324 | 0,8205 ± 0,0380 | 0,8457 ± 0,0260 |
| MLP | 0,8141 ± 0,0515 | 0,8195 ± 0,0373 | 0,8439 ± 0,0272 |

Resmî test F1 (pozitif sayısı 16 / 25 / 33):

| Model | ESIK 20 | ESIK 30 | ESIK 50 |
|---|---|---|---|
| Lojistik Regresyon | 0,8571 | 0,8462 | 0,8571 |
| Random Forest | 0,9091 | 0,8261 | 0,8824 |
| XGBoost | 0,9375 | 0,8261 | 0,8571 |
| SVM (RBF) | 0,8571 | 0,8302 | 0,8889 |
| MLP | 0,9091 | 0,8800 | 0,8649 |

Okurken:
- **Eşikler arası karşılaştırma dikkat ister.** Eşik değişince problem de değişiyor: pozitif oranı %10'dan %25'e çıkıyor, hep 0 modelinin Accuracy'si 0,898'den 0,753'e iniyor. F1 pozitif oranına bağlı olduğu için "ESIK 50'de F1 daha yüksek, demek ki daha iyi" denemez.
- **ROC-AUC eşik büyüdükçe düşüyor** (LR CV: 0,9934 → 0,9895 → 0,9765; diğerleri de aynı yönde). RUL 31-50 arasındaki motorlar sensörlerde sağlıklı motorlara hâlâ çok benziyor. Eşik büyüdükçe "sağlıklıya benzeyen pozitif" sayısı artıyor ve ayırmak zorlaşıyor.
- **Ağaç modelleri eşikten az etkileniyor.** RF ve XGBoost'un CV F1'i 0,85 civarında sabit. LR, SVM ve MLP'nin F1'i eşikle artıyor ve ESIK 50'de aradaki fark kapanıyor (0,838-0,856). RF her eşikte birinci.
- **Recall genelde eşik büyüdükçe düşüyor** (CV'de LR 0,967 → 0,947 → 0,907). Arızaya 50 döngü kalmış bir motoru yakalamak, 20 döngü kalmışı yakalamaktan zor.
- **Testte ESIK 30 beş modelin dördünde en düşük F1'i veriyor** (MLP'de en düşüğü ESIK 50). Bunu yorumlamıyoruz: test 100 motor ve tek bir motor F1'i birkaç puan oynatıyor. RUL'u 30'un hemen çevresinde olan test motorlarının etkisini ayrıca incelemedik.
- **Seçilen ayarlar eşiğe göre değişiyor** (`esik_duyarliligi_parametreler.csv`). XGBoost hep `max_depth=5` seçiyor; SVM ESIK 20'de `C=10, gamma=0.1`, ESIK 50'de katlar arasında dağınık.
- **Sonuç:** Model sıralaması (RF önde, farklar küçük) eşikten bağımsız. ESIK = 30 seçimi sonucu belirleyen bir şey değil.

### 7. RNN ve LSTM (`07_rnn.py`)
Derin modeller ilk planda kapsam dışıydı; grubun isteğiyle eklendi. PyTorch 2.15 (CPU sürümü) kuruldu, `requirements.txt`'ye `torch` eklendi. Süre yaklaşık 2 dakika (kontroller dahil).

**Klasik modellerden farkı girdi.** Klasik modeller her satırda yalnızca o döngünün 14 sensör değerini görüyor. RNN/LSTM ise aynı satır için o motorun **son 30 döngüsünü** sıra halinde görüyor, yani sensörlerin nasıl değiştiğini de görüyor. Motorun ilk 29 döngüsünde 30 döngülük geçmiş olmadığı için pencere ilk satır tekrarlanarak dolduruluyor. Pencere sadece geçmişten oluşuyor, gelecekteki döngüler hiç kullanılmıyor.

**Değerlendirme klasik modellerle birebir aynı:** aynı GroupKFold(5) katları, CV'de aynı 20.631 satır, resmî testte aynı 100 motorun son döngüsü. Testte pencere, test motorunun kendi geçmişinden (`test_FD001_tum_donguler.csv`) alınıyor. Script, bu dosyanın son döngülerinin `test_FD001_son_dongu.csv` ile aynı olduğunu kontrol ediyor.

**Mimari ve eğitim** (hiperparametre araması yapılmadı, sabit ayarlar):

| Ayar | Değer |
|---|---|
| Yapı | 1 katman RNN ya da LSTM, 32 gizli birim → son adımın gizli durumu → dropout 0,2 → tek çıktı |
| Kayıp | ikili çapraz entropi, pozitif sınıfa negatif/pozitif oranı kadar ağırlık (dengesizlik için) |
| Optimizasyon | Adam, öğrenme oranı 0,001, grup boyu 256, en fazla 40 epoch |
| Erken durdurma | her katta eğitim motorlarının %10'u (8 motor, motor bazlı) ayrılıyor; bu motorlardaki kayıp 5 epoch iyileşmezse eğitim durup en iyi epoch'a dönülüyor |
| Ölçekleme | `StandardScaler`, sadece o katın eğitim satırlarıyla fit |
| Resmî test | 100 motorun hepsiyle, CV'deki en iyi epoch'ların ortalaması kadar (ikisinde de 11) eğitiliyor |

"RNN" PyTorch'un en basit tekrarlayan hücresi (`nn.RNN`). LSTM, uzun dizileri daha iyi öğrenmesi için kapılar (gate) ekleyen sürümü. C-MAPSS literatüründe en sık kullanılan derin model olduğu için onu da ekledik.

**Sonuçlar** (ESIK = 30, `sonuclar_rnn.csv`, `tum_modeller_rnn_dahil.csv`):

| Model | Değerlendirme | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| RNN | CV 5 kat | 0,9439 ± 0,0146 | 0,7473 ± 0,0606 | 0,9577 ± 0,0217 | 0,8381 ± 0,0349 | 0,9911 ± 0,0018 |
| RNN | Resmî test | 0,9400 | 0,8800 | 0,8800 | 0,8800 | 0,9819 |
| LSTM | CV 5 kat | 0,9646 ± 0,0078 | 0,8300 ± 0,0377 | 0,9642 ± 0,0211 | 0,8914 ± 0,0202 | 0,9953 ± 0,0014 |
| LSTM | Resmî test | 0,9800 | 0,9259 | 1,0000 | 0,9615 | 0,9952 |

- LSTM, en iyi klasik model RF'den 5 katın 5'inde de yüksek F1 alıyor (ör. kat 1: 0,8606 / 0,8170; kat 2: 0,9150 / 0,8477). Fark (0,036) katlar arası std'den büyük ve her katta aynı yönde. Klasik modeller arasındaki farklardan farklı olarak bu tutarlı bir fark.
- LSTM aynı anda hem az kaçırıyor hem az yanlış alarm veriyor. CV'de 111 kaçırma / 620 yanlış alarm; RF'de 319 / 630, LR'de 164 / 1.115. Klasik modellerdeki "Recall mı Precision mı" ikilemini kırıyor.
- Testte LSTM 25 arızanın 25'ini yakaladı ve 2 yanlış alarm verdi. Test 100 motor olduğu için bu sayıyı tek başına öne çıkarmamak lazım; CV sonucu daha güvenilir.
- Zor bölgede (RUL 15-45) OOF AUC LSTM'de 0,9075, RNN'de 0,8510. Klasik modeller 0,833-0,847 aralığındaydı (`zor_bolge_auc_rnn.csv`). Kazanç tam da eşiğe yakın zor satırlarda.
- Basit RNN klasik modellerden belirgin şekilde iyi değil (CV F1 0,8381, RF'nin altında). Basit RNN uzun dizileri öğrenmekte zorlanıyor; öğrenme eğrilerinde de LSTM'in kaybı daha aşağı iniyor (`12_rnn_ogrenme_egrileri.png`).

**Kontroller** (`rnn_kontroller.csv`). LSTM'in CV ROC-AUC'si 0,99'u geçtiği için iki kontrol ekledik:

| Kontrol | OOF AUC | CV F1 |
|---|---|---|
| LSTM, etiketler rastgele karıştırılmış | 0,4978 | 0,2143 |
| LSTM, pencere = 1 (geçmiş yok) | 0,9893 | 0,8242 |
| RNN, pencere = 1 (geçmiş yok) | 0,9885 | 0,8190 |

- Karıştırılmış etiketle AUC yazı tura seviyesinde, yani boru hattında etiketi sızdıran bir şey yok.
- **Asıl önemli sonuç ikinci satır.** LSTM'e klasik modellerle aynı bilgiyi (tek döngü) verince F1 0,8242'ye iniyor, bu LR (0,8229) ile aynı. Yani LSTM'i iyi yapan "derin" olması değil, son 30 döngünün gidişatını görmesi. Klasik modellere de geçmiş bilgisi verilseydi (son birkaç döngünün ortalaması ya da eğimi gibi pencere özellikleriyle) farkın büyük kısmı kapanabilirdi. Bunu denemedik, çünkü pencere özellikleri planda hâlâ kapsam dışı. Sunumda "LSTM klasik ML'den iyi" yerine "geçmiş bilgisini kullanan model, tek döngüye bakan modellerden iyi" demek daha doğru.

Çıktılar: `sonuclar_rnn.csv`, `sonuclar_rnn_katlar.csv`, `karisiklik_matrisleri_rnn.csv`, `zor_bolge_auc_rnn.csv`, `rnn_kontroller.csv`, `rnn_egitim_gecmisi.csv`, `tum_modeller_rnn_dahil.csv`; grafikler `12_rnn_ogrenme_egrileri.png`, `13_tum_modeller_rnn_dahil.png`.

Tekrarlanabilirlik: aynı tohumla iki kez çalıştırdık, metrik tabloları birebir aynı çıktı. `rnn_egitim_gecmisi.csv`'deki kayıp değerleri 8. ondalıkta farklılaşıyor (çok çekirdekli toplamadan kaynaklanan kayan nokta farkı); tahminleri ve metrikleri etkilemiyor.

## Dosyalar

```
data/README.md         veri kaynağı, atıf, SHA-256
data/raw/              indirilen zip ve FD001 txt dosyaları (git'te yok)
data/processed/        etiketli csv'ler (git'te yok, scriptlerle yeniden üretiliyor)
scripts/ortak.py       sabitler (ESIK=30, RASTGELE_TOHUM=42, 14 sensör) ve ortak fonksiyonlar
scripts/01-07          adım scriptleri
outputs/figures/       grafikler
outputs/tables/        tablolar
```

İşlenmiş csv'lerde sütunlar şöyle: `unit`, `cycle`, `os1-os3` (çalışma ayarları), `s1-s21` (sensörler), `RUL`, `etiket` (RUL ≤ 30), `etiket_20`, `etiket_50`. Eğitim ve test dosyaları aynı sütun adlarını kullanıyor.

`sonuclar_baz.csv` içinde test satırlarının `_std` sütunları boş. Tek bir test kümesinin standart sapması olmadığı için bilerek boş bıraktık.

## Grafikler ve tablolar

**Sınıf dağılımı** (`01_sinif_dagilimi.png`, `sinif_oranlari.csv`): 17.531 satır sağlıklı (%84,97), 3.100 satır arızaya yakın (%15,03). Veri dengesiz, bu yüzden Accuracy tek başına yetmiyor.

**Motor ömrü** (`02_motor_omru_histogram.png`, `motor_omru_ozet.csv`): ömür 128 ile 362 döngü arasında, ortalama 206,31, medyan 199, en sık değer 213 (5 motor). Ortalama medyandan büyük, yani birkaç motor çok uzun yaşıyor. Her motorun son 31 döngüsü pozitif olduğu için 100 × 31 = 3.100 pozitif satır çıkıyor.

**Betimsel istatistik** (`betimsel_istatistik.csv`): sensörlerin ölçekleri çok farklı. s9'un ortalaması 9065,24 (std 22,08), s15'in ortalaması 8,44 (std 0,0375). Bu yüzden ölçekleme yapıyoruz. Sürekli sensörlerde mod anlamlı olmadığı için modu sadece motor ömrü ve etiket için hesapladık (`mod_ayrik_sutunlar.csv`).

**Sabit sütunlar** (`sabit_sutunlar.csv`): `os3, s1, s5, s10, s16, s18, s19` hiç değişmiyor. `s6` sadece iki değer alıyor (21,6 ve 21,61). Bunları atınca 14 sensör kalıyor.

**Örnek motorlar** (`03_ornek_motorlar_sensorler.png`): 1-4 numaralı motorlarda s2, s4 ve s11 arızaya yaklaştıkça artıyor, s12 azalıyor. Değişim son 50-100 döngüde hızlanıyor ama veri gürültülü.

**Korelasyon** (`04_korelasyon_isi_haritasi.png`, `korelasyon_14_sensor.csv`): en güçlü ilişki s9 ile s14 arasında (0,963). s11-s12 arası -0,847, s4-s11 arası 0,830. En zayıflar s8-s9 (-0,032) ve s9-s13 (-0,035). Sensörlerin çoğu birbiriyle ilişkili, o yüzden LR katsayılarını tek tek yorumlarken dikkatli olmak lazım.

**s11 kutu grafiği** (`05_s11_kutu_grafigi.png`, `s11_sinifa_gore.csv`): medyan sağlıklı sınıfta 47,47, arızaya yakın sınıfta 47,97. Kutular (47,32-47,61 ve 47,85-48,08) örtüşmüyor, bıyıklar örtüşüyor. Yani tek bir sensör bile sınıfları kısmen ayırabiliyor.

**Karışıklık matrisleri** (`06_karisiklik_matrisleri.png`, `karisiklik_matrisleri.csv`): CV'de LR 164 arızayı kaçırıp 1.115 yanlış alarm veriyor, RF 319 arızayı kaçırıp 630 yanlış alarm veriyor. Testte (25 pozitif) LR 22'sini yakalıyor (3 kaçırma, 5 yanlış alarm), RF 19'unu yakalıyor (6 kaçırma, 2 yanlış alarm).

**ROC eğrileri** (`07_roc_egrileri.png`): solda CV, sağda test. CV'de iki model neredeyse üst üste (LR 0,989, RF 0,988), testte LR 0,978, RF 0,983. Test eğrisi basamaklı çünkü sadece 100 örnek var. İki değerlendirmeyi aynı eksende çizmek kafa karıştırırdı, o yüzden yan yana iki panel yaptık.

**Gelişmiş modellerin karışıklık matrisleri** (`08_karisiklik_matrisleri_gelismis.png`, `karisiklik_matrisleri_gelismis.csv`): CV'de XGBoost 374 arızayı kaçırıp 621 yanlış alarm veriyor (RF'ye benziyor), SVM 137 kaçırma / 1.176 yanlış alarm, MLP 152 / 1.162 (ikisi LR'ye benziyor). Testte XGBoost 19/25 yakalıyor (6 kaçırma, 2 yanlış alarm; RF ile birebir aynı matris), SVM 22/25 (3 kaçırma, 6 yanlış alarm), MLP 22/25 (3 kaçırma, 3 yanlış alarm).

**Gelişmiş modellerin ROC eğrileri** (`09_roc_egrileri_gelismis.png`): CV'de OOF AUC XGBoost 0,987, SVM 0,990, MLP 0,988; üç eğri neredeyse üst üste. Testte 0,971 / 0,970 / 0,980.

**Beş modelin karşılaştırması** (`10_model_karsilastirma.png`): solda CV (hata çubukları katlar arası std), sağda test. y ekseni 0,5'ten başlıyor, farklar bu yüzden olduğundan büyük görünüyor; sunumda bunu söylemek lazım. Göze çarpan desen: ağaç modelleri (RF, XGBoost) Precision'da, LR/SVM/MLP Recall'da önde.

## Sonuçlar

Karar eşiği 0,5, pozitif sınıf 1 (`sonuclar_baz.csv`):

| Model | Değerlendirme | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Hep 0 | CV 5 kat | 0,8497 ± 0,0002 | 0,0000 | 0,0000 | 0,0000 | 0,5000 |
| Hep 0 | Resmî test | 0,7500 | 0,0000 | 0,0000 | 0,0000 | 0,5000 |
| Lojistik Regresyon | CV 5 kat | 0,9380 ± 0,0168 | 0,7303 ± 0,0631 | 0,9471 ± 0,0208 | 0,8229 ± 0,0357 | 0,9895 ± 0,0021 |
| Lojistik Regresyon | Resmî test | 0,9200 | 0,8148 | 0,8800 | 0,8462 | 0,9776 |
| Random Forest | CV 5 kat | 0,9540 ± 0,0106 | 0,8202 ± 0,0600 | 0,8971 ± 0,0307 | 0,8552 ± 0,0258 | 0,9887 ± 0,0028 |
| Random Forest | Resmî test | 0,9200 | 0,9048 | 0,7600 | 0,8261 | 0,9835 |
| XGBoost | CV 5 kat | 0,9518 ± 0,0100 | 0,8184 ± 0,0544 | 0,8794 ± 0,0339 | 0,8463 ± 0,0262 | 0,9881 ± 0,0032 |
| XGBoost | Resmî test | 0,9200 | 0,9048 | 0,7600 | 0,8261 | 0,9712 |
| SVM (RBF) | CV 5 kat | 0,9364 ± 0,0179 | 0,7213 ± 0,0634 | 0,9558 ± 0,0155 | 0,8205 ± 0,0380 | 0,9904 ± 0,0021 |
| SVM (RBF) | Resmî test | 0,9100 | 0,7857 | 0,8800 | 0,8302 | 0,9696 |
| MLP | CV 5 kat | 0,9363 ± 0,0173 | 0,7225 ± 0,0619 | 0,9510 ± 0,0216 | 0,8195 ± 0,0373 | 0,9894 ± 0,0031 |
| MLP | Resmî test | 0,9400 | 0,8800 | 0,8800 | 0,8800 | 0,9797 |

İlk altı satır `sonuclar_baz.csv`'den, son altı satır `sonuclar_gelismis.csv`'den; hepsi bir arada `tum_modeller_karsilastirma.csv`'de. Gelişmiş modellerde SVM'in sınıf kararı `decision_function ≥ 0`, diğerlerinde olasılık ≥ 0,5.

Hangisi daha iyi, ölçüte göre değişiyor:
- CV'de RF'nin F1'i daha yüksek (0,8552'ye karşı 0,8229), Accuracy ve Precision'da da önde. LR'nin Recall'u daha yüksek (0,9471'e karşı 0,8971). ROC-AUC neredeyse aynı. F1 farkı standart sapmalardan küçük ama RF 5 katın 5'inde de LR'den yüksek F1 alıyor (`sonuclar_baz_katlar.csv`).
- Testte LR'nin F1'i (0,8462'ye karşı 0,8261) ve Recall'u (0,88'e karşı 0,76) daha yüksek, RF'nin Precision ve ROC-AUC'si daha yüksek. Aradaki fark birkaç motorluk.
- Arıza tespitinde bir arızayı kaçırmak yanlış alarmdan daha pahalı, bu açıdan LR öne çıkıyor. F1'e göre ise RF biraz önde. CV daha çok örneğe dayandığı için ona daha çok güveniyoruz. Bizim özetimiz: "F1'e göre RF biraz daha iyi, ama fark küçük ve testte tersine dönüyor."

Gelişmiş modeller eklenince:
- CV F1 sıralaması: RF 0,8552 > XGBoost 0,8463 > LR 0,8229 > SVM 0,8205 > MLP 0,8195. En iyi ile en kötü arası 0,036; katlar arası std 0,026-0,038. Kat bazında (`kat_bazli_f1_tum_modeller.csv`) RF 5 katın 5'inde de birinci ve her katta XGBoost'tan yüksek; en yakın olduğu yer 3. kat (RF 0,8564, LR 0,8512). Modeller arasındaki fark, katlar arasındaki farktan küçük: 1. kat her modelde en zor kat (F1 0,76-0,82).
- İki grup var. Ağaç modelleri (RF, XGBoost) daha az yanlış alarm veriyor ama daha çok arıza kaçırıyor (CV Recall 0,88-0,90). LR, SVM ve MLP daha çok arıza yakalıyor (CV Recall 0,95-0,96) ama yanlış alarmları neredeyse iki katı. SVM ve MLP sonuçta LR ile hemen hemen aynı noktaya geliyor.
- Testte MLP en yüksek F1'i aldı (0,88): 22 arızayı yakaladı, 3 yanlış alarm verdi. LR'den farkı 2 yanlış alarm, yani 2 motor. CV'de ise MLP beşinci. Bu yüzden "MLP en iyisi" demiyoruz; 100 motorluk testte bu kadar fark gürültü sınırında.
- XGBoost testte RF ile birebir aynı karışıklık matrisini verdi (19 TP, 2 FP, 6 FN), ROC-AUC'si biraz düşük (0,9712'ye karşı 0,9835).
- ROC-AUC'de beş model CV'de 0,9881-0,9904 aralığında; bu metrik modelleri ayırmıyor.
- Zor bölgede (RUL 15-45) OOF AUC: XGBoost 0,8341, SVM 0,8467, MLP 0,8385 (LR 0,8371, RF 0,8332). Yani daha esnek modeller eşiğe yakın zor satırlarda da belirgin bir kazanç getirmiyor (`zor_bolge_auc_gelismis.csv`).
- Yorumumuz: bu özelliklerle (her döngünün 14 ham sensör değeri, geçmiş bilgisi yok) beş model de aşağı yukarı aynı tavana dayanıyor. Darboğaz model değil, özellikler: tek bir döngünün gürültülü ölçümü RUL 25 ile RUL 35'i ayırmaya yetmiyor. Daha fazla kazanç büyük ihtimalle model değiştirmekten değil, birkaç döngünün bilgisini kullanan özelliklerden gelir (bu kapsam dışı, grup kararı).

CV ile test neden farklı:
1. Sınıf oranı farklı: CV'de %15, testte %25 pozitif var.
2. CV bir motorun bütün döngülerine bakıyor ve bu satırların %51'inde RUL > 100, yani çok kolay örnekler. Testte sadece son döngü var. Test RUL değerleri 7 ile 145 arasında (medyan 86) ve sadece %33'ünde RUL > 100. Eşiğe yakın zor örneklerin (RUL 15-45) oranı eğitimde %15,03, testte %19.
3. Testte 100 örnek ve 25 pozitif var. Tek bir motor Recall'u 0,04 değiştiriyor. Güven aralığı hesaplamadık.
4. Aynı motorun ardışık döngüleri birbirine çok benzediği için CV'deki 20.631 satır 20.631 bağımsız örnek sayılmaz.
5. CV'de model 80 motorla, testte 100 motorla eğitiliyor.

## Kararlarımız

- **ESIK = 30.** Proje tanımında böyle verildi. Her motorun son 31 döngüsü pozitif oluyor, oran %15. Bakım planlamak için de makul bir uyarı süresi. 20 ve 50 için de etiket ürettik, karşılaştırmasını sonra yapacağız.
- **Motor bazlı bölme.** Aynı motorun ardışık döngüleri neredeyse aynı. Rastgele bölseydik model bir motorun 150. döngüsünü görüp 151. döngüsünü "tahmin ederdi". Gerçekte ise hiç görmediği bir motor için tahmin yapacak. Her katta eğitim ve doğrulama motorlarının çakışmadığını `assert` ile kontrol ediyoruz (80 eğitim, 20 doğrulama motoru).
- **Özellikler: 14 sensörün ham değeri.** Sabit sütunları ve `s6`'yı attık. FD001'de tek çalışma koşulu olduğu için `os1` ve `os2`'yi de kullanmadık. `cycle` ve `unit` kesinlikle özellik değil: `cycle` etiketi neredeyse doğrudan veriyor, `unit` sadece bir numara.
- **Ölçekleme Pipeline içinde.** `StandardScaler` her katta sadece eğitim motorlarıyla fit ediliyor. RF'nin ölçeklemeye ihtiyacı yok ama üç model aynı yapıda olsun diye onu da pipeline'a koyduk.
- **Model ayarları.** LR: `class_weight="balanced"`, `max_iter=1000` (yakınsama uyarısı çıkmadı). RF: `n_estimators=200`, `class_weight="balanced"`, `random_state=42`, `n_jobs=-1`. Hiperparametre araması yapmadık.
- **Karar eşiği 0,5.** Değiştirerek Precision-Recall dengesi ayarlanabilir, bunu şimdilik yapmadık.
- **Hep 0 modeli.** `DummyClassifier(strategy="most_frequent")`. Sabit olasılık verdiği için ROC-AUC tam 0,5.
- **Test: son döngü.** C-MAPSS'in resmî değerlendirme şekli bu, `RUL_FD001.txt` sadece son döngünün RUL'unu veriyor.

## ROC-AUC 0,99'u geçince ne yaptık

4. katta LR 0,9932, RF 0,9924 çıktı (ortalamalar 0,9895 ve 0,9887). Bu kadar yüksek bir değer sızıntı belirtisi olabileceği için şunlara baktık (`sizinti_incelemesi.csv`):
- Kodu baştan okuduk. Özellikler sadece 14 sensör, her katta motorlar ayrık, ölçekleme pipeline içinde ve her katta `clone` ile sıfırdan fit ediliyor.
- Etiketleri rastgele karıştırıp modeli tekrar eğittik. LR'nin AUC'si 0,5028 çıktı, yani yazı tura. Model etiketle ilgisi olmayan bir bilgiden faydalanıyor olsaydı bu değer belirgin şekilde 0,5'in üstünde olurdu.
- Eşiğe yakın satırlara baktık. Bütün satırlarda OOF AUC LR 0,9886, RF 0,9880. RUL 0-60 bandında (6.100 satır) LR 0,9423, RF 0,9400. RUL 15-45 bandında (3.100 satır) LR 0,8371, RF 0,8332. Yani yüksek AUC, arızadan çok uzak ve ayırması kolay satırlardan geliyor. Zor bölgede model çok daha az başarılı.
- Bağımsız test kümesinde AUC 0,9776 / 0,9835 çıktı, CV ile tutarlı.

Sonuç olarak sızıntı bulmadık. Yüksek değer, her döngüyü ayrı örnek saymamızın doğal sonucu. Sunumda ROC-AUC'nin bu yüzden iyimser olduğunu söyleyip F1 ve Recall'u öne çıkaracağız.

## Kontroller

Her adımdan sonra çıktıları silip scripti baştan çalıştırdık ve dosyaların doğru üretildiğine baktık:
- Bütün csv'lerde NaN yok (sadece `sonuclar_baz.csv`'deki bilerek boş test `_std` hücreleri var).
- Bütün metrikler 0-1 arasında, bu script içinde kontrol ediliyor.
- Grafikleri tek tek açıp baktık, yazılar üst üste binmiyor ve Türkçe karakterler düzgün.
- `03_baz_modeller.py`'yi aynı tohumla iki kez çalıştırdık, tablolar birebir aynı çıktı.
- Etiketlerde her motor 0'dan 1'e bir kez geçiyor, geri dönmüyor. Motor başına pozitif sayısı eşik + 1 (21 / 31 / 51).

## Eksikler ve dikkat edilecekler

- Testte 100 örnek var, modeller arasındaki farkların anlamlı olup olmadığını test etmedik.
- XGBoost'un seçtiği ayar ızgaranın köşesinde (en derin, en çok ağaç, en yüksek öğrenme oranı). Izgara genişletilirse biraz daha iyi bir ayar çıkabilir; denemedik.
- MLP'nin `early_stopping` için ayırdığı %10'luk doğrulama payı satır bazlı rastgele, motor bazlı değil. Bu sadece eğitimin ne zaman duracağını etkiliyor, dış değerlendirme yine motor bazlı; ama tam temiz değil.
- Hiperparametre aramasında skor F1 ve karar eşiği 0,5 (SVM'de 0). Recall'u öne çıkarmak istersek skor ya da eşik değiştirilebilir.
- RNN/LSTM'de hiperparametre araması yok (pencere 30, 32 gizli birim sabit). LSTM klasik modellerle aynı aramadan geçmediği için bu açıdan dezavantajlı; buna rağmen önde.
- RNN/LSTM sonuçları sadece ESIK = 30 için var, 20 ve 50'de denemedik.
- Pencerenin ilk satırla doldurulması, modele "bu motorun ilk 29 döngüsündeyim" bilgisini dolaylı olarak verebilir. Bu satırların hepsi RUL ≥ 99 olan kolay negatifler (en kısa motor 128 döngü), eşik çevresindeki kararları etkilemez.
- PyTorch'ta CPU üzerinde aynı tohumla metrikler tekrarlanıyor, ama farklı bir işlemci ya da iş parçacığı sayısıyla küçük farklar çıkabilir.
- Sensörler arasında güçlü korelasyon var, LR katsayıları "hangi sensör önemli" sorusuna doğrudan cevap değil.
- Eğitim etiketi, her eğitim motorunun son satırında arıza olduğu varsayımına dayanıyor. Veri setinin tanımı böyle.
- Test tüm döngüler dosyasındaki RUL değerleri "RUL her döngüde bir azalır" kabulüne dayanıyor. Son döngülerde dosyayla tutuyor, öncekiler için bağımsız bir kaynak yok.
- Proje tanımında 30 eşiğinin literatürde yaygın olduğu söyleniyor, kaynağına biz bakmadık.
- Python 3.14.10 çok yeni. Daha eski bir sürümde küçük sayısal farklar çıkabilir, bunu denemedik.
- Veri simülasyon, gerçek motor verisi değil.

## Sırada ne var

Grubun karar vermesi gerekenler:
- Klasik modellere pencere özellikleri (son N döngünün ortalaması, eğimi) eklemek. LSTM'in üstünlüğünün ne kadarının geçmiş bilgisinden geldiğini klasik tarafta göstermenin en doğrudan yolu bu.
- RNN/LSTM'i ESIK 20 ve 50 için de çalıştırmak (scriptte küçük bir değişiklik yeterli).
