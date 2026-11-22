# Etiketleme: etiketler nasıl kuruldu?

Modelin öğrendiği "doğru cevapların", yani etiketlerin nereden geldiğini kod bilmeyen biri de anlasın diye yazdık. Sayılar `01_indir_ve_etiketle.py` ve `04_etiket_tamamla.py` çıktılarından ve `outputs/tables/sinif_oranlari.csv` dosyasından.

## 1. RUL nedir, etiket nedir?

**RUL (Remaining Useful Life, kalan faydalı ömür):** bir motorun arızalanmadan önce kaç döngü (uçuş) daha çalışabileceği.

**Etiket:** modelin tahmin etmeye çalıştığı cevap. Bizim projede iki değer alabilir:
- **1 = arızaya yakın**: RUL ≤ 30
- **0 = sağlıklı**: RUL > 30

**Gerçek bir örnek (eğitim verisi, 1 numaralı motor):** bu motor 192 döngü çalıştıktan sonra arızalanmış. Bu durumda RUL = 192 − döngü numarası olur:

| Döngü | RUL | Etiket (eşik 30) | Eşik 20 | Eşik 50 |
|---|---|---|---|---|
| 1 | 191 | 0 | 0 | 0 |
| 100 | 92 | 0 | 0 | 0 |
| 161 | 31 | 0 | 0 | 1 |
| 162 | 30 | **1** | 0 | 1 |
| 192 | 0 | 1 | 1 | 1 |

Yani motor 162. döngüden itibaren "arızaya yakın" sayılıyor. O andan arızaya kadar 31 döngü var (162, 163, …, 192). Bu yüzden eşik 30'da her motor tam **31** pozitif satır veriyor.

## 2. Neden ana eşik 30? 20 ve 50 neden duyarlılık için tutuldu?

- **30**, proje tanımında ana eşik olarak verildi. Pratik bir anlamı da var: 30 uçuşluk bir uyarı, bakımı planlamak için makul bir süre. Veride pozitif oranı %15,03 çıkıyor, ne çok nadir ne çok sık. Proje tanımında 30'un literatürde yaygın olduğu da yazıyor ama kaynağına biz bakmadık.
- **20 ve 50**, sonuçların seçtiğimiz eşiğe ne kadar bağlı olduğunu görmek için tutuldu. Daha dar bir uyarı penceresinde (20) pozitifler azalıyor (%10,18), daha geniş bir pencerede (50) artıyor (%24,72). Bu eşiklerle modelleme henüz yapılmadı, sonraki adımın konusu.

## 3. Etiketleri kim koydu? Elle mi, yapay zekâyla mı?

**İkisi de değil.** Etiketler bir **kuralla, otomatik olarak** üretildi. Kimse satırlara tek tek bakıp "bu arızalı" demedi, hiçbir yapay zekâ modeli de etiket tahmin etmedi.

Bu mümkün, çünkü veri setinin kendisi gerekli bilgiyi taşıyor: eğitimdeki her motor **arızalanana kadar** çalıştırılmış ve kaydın son satırı arıza anı. Arıza anı bilindiği için her satırın RUL değeri basit bir çıkarmayla bulunuyor (son döngü − o anki döngü). Etiket de "RUL ≤ 30 mu?" sorusunun cevabı.

Bu kuralın kodu `scripts/ortak.py` içindeki `etiketle` fonksiyonunda. Bütün veri dosyaları aynı fonksiyonla etiketlendi.

## 4. Test etiketi nasıl bulunuyor?

Test motorları arızadan **önce**, rastgele bir noktada kesilmiş. Verinin içinde arıza anı yok. NASA, her test motorunun son kaydedilen döngüsünde gerçekte kaç döngü kaldığını ayrı bir dosyada veriyor: **`RUL_FD001.txt`** (100 satır, her motor için bir sayı).

- **Resmî test (son döngü):** her test motorunun yalnızca son satırı alınır. RUL değeri `RUL_FD001.txt`'den okunur ve aynı kural uygulanır. Örnek: 34 numaralı motor 203. döngüde kesilmiş. Dosyadaki değeri 7, yani RUL = 7 ve etiket 1. 1 numaralı motor 31. döngüde kesilmiş, RUL = 112, etiket 0.
- **Test, bütün döngüler** (`test_FD001_tum_donguler.csv`, yeni): bir test motorunun daha önceki döngülerinin RUL değeri de geriye doğru hesaplanabilir: `RUL = dosyadaki değer + (motorun son döngüsü − o döngü)`. Örnek: 34 numaralı motorun 1. döngüsünde RUL = 7 + (203 − 1) = 209. Bu hesap, benchmark'ın şu kabulüne dayanır: test motorları arızadan önce kesilmiştir ve her döngüde RUL bir azalır. Kontrol ettik: her motorun son döngüsündeki RUL, `RUL_FD001.txt` ile birebir aynı. Son döngü satırları da mevcut `test_FD001_son_dongu.csv` ile birebir aynı.

## 5. Gerçek sayılarla sınıf dağılımı

| Veri | Satır | Eşik 20 (pozitif / negatif) | **Eşik 30** (pozitif / negatif) | Eşik 50 (pozitif / negatif) |
|---|---|---|---|---|
| Eğitim (100 motor, bütün döngüler) | 20.631 | 2.100 / 18.531 (%10,18) | **3.100 / 17.531 (%15,03)** | 5.100 / 15.531 (%24,72) |
| Resmî test (100 motor, son döngü) | 100 | 16 / 84 (%16) | **25 / 75 (%25)** | 33 / 67 (%33) |
| Test, bütün döngüler (yeni, modellemede kullanılmadı) | 13.096 | 123 / 12.973 (%0,94) | 332 / 12.764 (%2,54) | 886 / 12.210 (%6,77) |

**Sınıflar dengeli değil.** Eğitimde her 100 satırdan yaklaşık 15'i "arızaya yakın". Her şeye "sağlıklı" diyen bir model bile %85 doğruluk alır, ama hiçbir arızayı yakalamaz. Bu yüzden modelleri yalnızca doğrulukla (Accuracy) değil, Recall, Precision ve F1 ile de değerlendiriyoruz. Test bütün döngüler dosyasında pozitifler çok daha az (%2,54), çünkü test motorları arızadan önce kesildiği için çoğu satır arızadan uzak.

**Etiket kontrolleri (hepsi geçti):** eğitimde her motorda etiket bir kez 0'dan 1'e geçiyor ve bir daha geri dönmüyor. Her motordaki pozitif satır sayısı tam olarak eşik + 1 (eşik 20'de 21, 30'da 31, 50'de 50+1 = 51). Test son döngü etiketleri, `RUL_FD001.txt`'den bağımsız olarak yeniden hesaplandığında aynı sayıları verdi (16 / 25 / 33 pozitif).

## 6. Bilinen sınırlar

- **Eşik keyfi.** 30 yerine 29 ya da 31 de seçilebilirdi. "29 döngü kalan motor arızaya yakın, 31 döngü kalan sağlıklı" ayrımı doğal değil, bizim koyduğumuz bir çizgi.
- **Eşiğin çevresindeki satırları ayırt etmek zor.** RUL'u 28 olan bir motorla 33 olan bir motorun sensör değerleri neredeyse aynı. Modelin hatalarının çoğu bu bölgede çıkıyor: Aşama 3'teki incelemede ROC-AUC bütün satırlarda 0,99 civarındayken RUL 15-45 bandında yaklaşık 0,84'e düştü (`outputs/tables/sizinti_incelemesi.csv`).
- **Test yalnızca 100 örnek.** Resmî testte 25 pozitif var. Tek bir motor, yakalanan arıza oranını 4 puan değiştiriyor.
- **Simülasyon verisi.** Motorlar gerçek değil, NASA'nın simülasyonundan geliyor.
- **Yedek veriler kullanılmadı.** FD002, FD003 ve FD004 aynı kuralla etiketlenip `data/processed/yedek/` altına konuldu, ama modellemede **kullanılmadı**. FD002 ve FD004'te 6 farklı çalışma koşulu var, FD003'te iki arıza türü var. Bunlar FD001'den farklı problemler.

## 7. Sunumda sorulabilecek 5 soru

**1. Etiketleri kim koydu?**
Kimse elle koymadı, yapay zekâ da koymadı. Veri setinde her eğitim motoru arızaya kadar çalışıyor, yani arıza anı biliniyor. Etiketi bir kuralla otomatik ürettik: kalan ömür 30 döngü veya daha azsa 1, değilse 0.

**2. Neden 30?**
Proje tanımında ana eşik olarak verildi. Bakımı planlamak için makul bir uyarı süresi, pozitif oranı da %15 ile makul. Eşiğe ne kadar bağlı olduğumuzu görmek için 20 ve 50 için de etiket ürettik. Ama sonuçta bu bizim seçtiğimiz bir çizgi.

**3. Test etiketi nereden geliyor?**
Test motorları arızadan önce kesilmiş. NASA, her test motorunun son kaydında kaç döngü kaldığını `RUL_FD001.txt` dosyasında veriyor. Aynı kuralı (≤ 30 ise 1) bu değere uyguluyoruz. Sonuç 25 pozitif, 75 negatif.

**4. Sınıflar dengeli mi?**
Hayır. Eğitimde 3.100 pozitif, 17.531 negatif var (%15 pozitif). Bu yüzden Accuracy yanıltıcı olur. Modellerde `class_weight="balanced"` kullandık ve F1, Recall gibi ölçütlere baktık.

**5. Etiketleme sızıntı yaratır mı?**
Etiket, RUL'dan ve dolayısıyla döngü numarasından hesaplanıyor. Bu yüzden `cycle`, `RUL` ve etiket sütunları modele **girdi olarak verilmiyor**, model yalnızca 14 sensörü görüyor. Bu kural kodda `assert` ile kontrol ediliyor. Ayrıca veri motor bazlı bölünüyor, yani bir motorun bütün satırları ya eğitimde ya doğrulamada. Etiketler rastgele karıştırılınca modelin başarısı yazı-tura düzeyine iniyor (ROC-AUC 0,5028). Bu da etiketin başka bir yoldan sızmadığını gösteriyor.
