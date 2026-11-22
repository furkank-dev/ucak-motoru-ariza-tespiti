# Yaptıklarımız

Projede şimdiye kadar ne yaptığımızı, hangi sonuçları aldığımızı ve neden öyle karar verdiğimizi buraya yazıyoruz. Sayılar scriptlerin çıktılarından ve `outputs/` altındaki dosyalardan.

Kısaca: FD001 verisini indirip etiketledik, veriyi inceledik ve iki baz model kurduk (Lojistik Regresyon ve Random Forest). İkisi de her şeye "sağlıklı" diyen modeli rahat geçiyor. Resmî testte F1 LR için 0,8462, RF için 0,8261. Çapraz doğrulamada (CV) LR 0,8229 ± 0,0357, RF 0,8552 ± 0,0258.

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

## Dosyalar

```
data/README.md         veri kaynağı, atıf, SHA-256
data/raw/              indirilen zip ve FD001 txt dosyaları (git'te yok)
data/processed/        etiketli csv'ler (git'te yok, scriptlerle yeniden üretiliyor)
scripts/ortak.py       sabitler (ESIK=30, RASTGELE_TOHUM=42, 14 sensör) ve ortak fonksiyonlar
scripts/01-04          adım scriptleri
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

Hangisi daha iyi, ölçüte göre değişiyor:
- CV'de RF'nin F1'i daha yüksek (0,8552'ye karşı 0,8229), Accuracy ve Precision'da da önde. LR'nin Recall'u daha yüksek (0,9471'e karşı 0,8971). ROC-AUC neredeyse aynı. F1 farkı standart sapmalardan küçük ama RF 5 katın 5'inde de LR'den yüksek F1 alıyor (`sonuclar_baz_katlar.csv`).
- Testte LR'nin F1'i (0,8462'ye karşı 0,8261) ve Recall'u (0,88'e karşı 0,76) daha yüksek, RF'nin Precision ve ROC-AUC'si daha yüksek. Aradaki fark birkaç motorluk.
- Arıza tespitinde bir arızayı kaçırmak yanlış alarmdan daha pahalı, bu açıdan LR öne çıkıyor. F1'e göre ise RF biraz önde. CV daha çok örneğe dayandığı için ona daha çok güveniyoruz. Bizim özetimiz: "F1'e göre RF biraz daha iyi, ama fark küçük ve testte tersine dönüyor."

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

- Testte 100 örnek var, LR ile RF arasındaki farkların anlamlı olup olmadığını test etmedik.
- Sensörler arasında güçlü korelasyon var, LR katsayıları "hangi sensör önemli" sorusuna doğrudan cevap değil.
- Eğitim etiketi, her eğitim motorunun son satırında arıza olduğu varsayımına dayanıyor. Veri setinin tanımı böyle.
- Test tüm döngüler dosyasındaki RUL değerleri "RUL her döngüde bir azalır" kabulüne dayanıyor. Son döngülerde dosyayla tutuyor, öncekiler için bağımsız bir kaynak yok.
- Proje tanımında 30 eşiğinin literatürde yaygın olduğu söyleniyor, kaynağına biz bakmadık.
- Python 3.14.10 çok yeni. Daha eski bir sürümde küçük sayısal farklar çıkabilir, bunu denemedik.
- Veri simülasyon, gerçek motor verisi değil.

## Sırada ne var

XGBoost, SVM ve MLP gibi daha güçlü modeller, küçük hiperparametre aramaları, 20/30/50 eşik karşılaştırması, son birkaç döngünün ortalaması gibi pencere özellikleri ve LSTM / 1D-CNN gibi derin modeller.
