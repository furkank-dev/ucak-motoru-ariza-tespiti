# Proje Anlatımı: Uçak Motoru Arıza Tespiti

Sunumda ve sözlüde işimize yarasın diye hazırladık. Ayrıntılar `YAPILANLAR.md` dosyasında.

## 1. Problem bir cümlede
Bir uçak motorunun sensör ölçümlerine bakarak, motorun **önümüzdeki 30 döngü içinde arızalanıp arızalanmayacağını** tahmin ediyoruz. Bu iki sınıflı bir sınıflandırma problemi: 1 = arızaya yakın, 0 = sağlıklı.

## 2. Veri
- NASA'nın C-MAPSS veri seti (Saxena ve Goebel, 2008). Motorlar gerçek değil, bir **simülasyonla** yıpratılıp arızaya kadar çalıştırılmış.
- Biz en basit alt kümeyi, **FD001**'i kullandık: tek bir çalışma koşulu ve tek bir arıza türü var.
- **Eğitim:** 100 motor, 20.631 satır. Her satır bir motorun bir döngüsü (uçuşu). Her motor arızaya kadar izlenmiş. Motor ömürleri 128 ile 362 döngü arasında, ortalama 206.
- **Test:** 100 farklı motor, 13.096 satır. Bu motorlar arızadan *önce* bir yerde kesilmiş. Her motorun son satırında kaç döngü kaldığı ayrı bir dosyada (`RUL_FD001.txt`) veriliyor.
- Her satırda 3 çalışma ayarı ve 21 sensör var.

## 3. Scriptler ne yapıyor, neden öyle yapıyor?

### `ortak.py`
Bütün scriptlerin ortak kullandığı sabitler (eşik 30, rastgele tohum 42, 14 sensörün listesi) ve küçük fonksiyonlar burada: dosya okuma, etiketleme, metrik hesaplama. **Neden:** aynı sayı iki yerde ayrı ayrı yazılırsa bir gün birbirini tutmaz. Hepsi tek yerde dururken bu olamaz.

### `01_indir_ve_etiketle.py`
1. Veriyi 4 farklı adresten sırayla indirmeyi dener, ilk çalışanı kullanır. Bizde ilk adres (NASA'nın PHM sunucusu) çalıştı.
2. Dosyaların SHA-256 parmak izini yazar. Böylece aynı veriyi kullandığımızı kanıtlayabiliriz.
3. Satır ve motor sayılarını kontrol eder (`assert`). Bir sayı tutmazsa program durur, sessizce devam etmez.
4. **Etiketi kurar:** eğitimde her motorun son döngüsü arıza anıdır. Bu yüzden `RUL = son döngü − şu anki döngü` olur. RUL ≤ 30 ise etiket 1, değilse 0.
5. Testte her motorun yalnızca son döngüsünü alır. RUL değerini NASA'nın verdiği dosyadan okur ve aynı kuralla etiketler.

**Sonuç:** eğitimde 3.100 pozitif, 17.531 negatif (%15,03 pozitif). Testte 25 pozitif, 75 negatif.

### `02_inceleme.py`
Modele geçmeden veriyi tanımak için grafikler ve tablolar üretir:
- **Sınıf dağılımı:** pozitifler yalnızca %15. Veri dengesiz.
- **Motor ömrü:** min 128, ortalama 206,31, medyan 199, maks 362, mod 213.
- **Sabit sütunlar:** 7 sütun (`os3, s1, s5, s10, s16, s18, s19`) hiç değişmiyor. `s6` yalnızca iki değer alıyor. Bunlar bilgi taşımadığı için atıldı ve geriye **14 sensör** kaldı.
- **Örnek motorlar:** s2, s4 ve s11 arızaya yaklaştıkça yükseliyor, s12 düşüyor. Yani sensörlerde yıpranmanın izi var.
- **Korelasyon:** sensörlerin çoğu birbiriyle güçlü ilişkili (ör. s9 ile s14 arasında 0,96).
- **s11 kutu grafiği:** arızaya yakın motorlarda s11 belirgin olarak daha yüksek (medyan 47,97'ye karşı 47,47).

### `03_baz_modeller.py`
- Girdi olarak yalnızca 14 sensörün o döngüdeki değerini kullanır.
- Üç model kurar: **hep 0** diyen aptal model (baz çizgisi), **Lojistik Regresyon** ve **Random Forest**.
- Her model bir `Pipeline` içinde: önce ölçekleme (StandardScaler), sonra model.
- **5 katlı motor bazlı çapraz doğrulama** (GroupKFold): 100 motor 5 gruba bölünür. Her seferinde 80 motorla eğitilir, hiç görülmemiş 20 motorla sınanır.
- Sonra modeli 100 motorun hepsiyle eğitip **resmî test kümesinde** değerlendirir.

## 4. Sonuçlar (sunum tablosu)

| Model | CV F1 | CV Recall | CV ROC-AUC | Test F1 | Test Recall | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Hep 0 (baz çizgisi) | 0,0000 | 0,0000 | 0,5000 | 0,0000 | 0,0000 | 0,5000 |
| Lojistik Regresyon | 0,8229 ± 0,0357 | 0,9471 | 0,9895 | 0,8462 | 0,8800 | 0,9776 |
| Random Forest | 0,8552 ± 0,0258 | 0,8971 | 0,9887 | 0,8261 | 0,7600 | 0,9835 |

Accuracy: "hep 0" CV'de 0,8497 ve testte 0,75. LR CV'de 0,9380 ve testte 0,92. RF CV'de 0,9540 ve testte 0,92.

Anlatırken:
- İki model de baz çizgisini açıkça geçiyor.
- **LR daha çok alarm veriyor.** Arızaların daha fazlasını yakalıyor (yüksek Recall), ama yanlış alarmı da daha çok.
- **RF daha temkinli.** Daha az yanlış alarm veriyor (yüksek Precision), ama birkaç arızayı kaçırıyor.
- CV'de F1'e göre RF biraz önde, testte LR biraz önde. Fark küçük. Testte yalnızca 100 örnek olduğu için bu farka fazla anlam yüklememek gerekiyor.

## 5. Sözlüde sorulabilecek sorular ve kısa cevaplar

**1. Etiketi nasıl kurdunuz?**
Eğitim motorları arızaya kadar çalıştırılmış, yani her motorun son satırı arıza anı. Kalan ömür `RUL = son döngü − şu anki döngü`. RUL ≤ 30 ise "arızaya yakın" (1), değilse "sağlıklı" (0). Testte motorlar arızadan önce kesilmiş. Son satırın RUL değerini NASA ayrı bir dosyada veriyor, biz de aynı kuralı uyguladık.

**2. Eşik neden 30?**
Proje tanımında 30 verildi. Pratikte de mantıklı: 30 döngü, bakımı planlamak için yeterli bir uyarı süresi. Veride pozitiflerin oranı %15, ne çok nadir ne çok sık. 20 ve 50 için de etiket ürettik (pozitif oranları %10,18 ve %24,72). Eşiklerin karşılaştırması bir sonraki adım.

**3. Neden motor bazlı bölme yaptınız? Rastgele bölseydiniz ne olurdu?**
Aynı motorun ardışık döngüleri neredeyse aynı ölçümlere sahip. Rastgele bölseydik model, bir motorun 150. döngüsünü eğitimde görüp 151. döngüsünü doğrulamada "tahmin ederdi". Bu kopya çekmek gibi olur ve sonuç sahte biçimde yüksek çıkar. Gerçek hayatta ise model hiç görmediği yeni bir motor için tahmin yapacak. GroupKFold bir motorun bütün satırlarını aynı tarafta tutuyor. Biz bunu her katta `assert` ile kontrol ettik.

**4. Sızıntı (leakage) nedir, nasıl önlediniz?**
Sızıntı, modelin eğitimde gerçek hayatta elinde olmayacak bir bilgiyi görmesi. Üç önlem aldık: (1) motor bazlı bölme, (2) ölçeklemenin Pipeline içinde yalnızca eğitim verisiyle yapılması, (3) `cycle` ve `unit` sütunlarının özellik olarak kullanılmaması. Ayrıca etiketleri rastgele karıştırıp modeli yeniden eğittik. AUC 0,5028 çıktı, yani rastgele tahmin düzeyi. Bu, boru hattının etiket bilgisini gizlice kullanmadığını gösteriyor.

**5. `cycle` sütununu neden kullanmadınız? Çok faydalı olurdu.**
Eğitimde `cycle` büyüdükçe motor arızaya yaklaşıyor, yani etiketi neredeyse doğrudan veriyor. Ama bir motorun ömrünün kaç döngü olacağını önceden bilemeyiz. 150. döngüdeki bir motor kısa ömürlüyse arızaya yakın, uzun ömürlüyse daha yolun yarısında olabilir. Model sensörlere değil sayaca bakmayı öğrenirdi. `unit` ise yalnızca bir kimlik numarası.

**6. Neden Accuracy yetmiyor?**
Veri dengesiz. Her şeye "sağlıklı" diyen aptal model bile CV'de %84,97 Accuracy alıyor, ama tek bir arızayı yakalamıyor (Recall = 0). Bu yüzden Precision, Recall, F1 ve ROC-AUC'ye bakıyoruz.

**7. Precision ve Recall farkı nedir? Hangisi daha önemli?**
Precision: "arızaya yakın" dediklerimizin ne kadarı gerçekten öyle. Recall: gerçekten arızaya yakın olanların ne kadarını yakaladık. Uçak motorunda bir arızayı kaçırmak, gereksiz bir kontrolden çok daha pahalı. Bu yüzden Recall'u daha önemli sayarız. Ama Precision çok düşerse de sürekli yanlış alarm olur, kimse uyarılara güvenmez.

**8. F1 nedir?**
Precision ile Recall'un harmonik ortalaması. Biri çok düşükse F1 de düşük çıkar. İki ölçütü tek sayıda dengelemek için kullanılır.

**9. ROC-AUC nedir?**
Model her örneğe bir "arıza olasılığı" veriyor. ROC-AUC, rastgele seçilen bir arızalı örneğin rastgele seçilen bir sağlıklı örnekten daha yüksek olasılık alma ihtimali. 0,5 yazı tura, 1 mükemmel sıralama demek. ROC eğrisi, karar eşiğini 0'dan 1'e kaydırdıkça yakalanan arıza oranının (Recall) yanlış alarm oranına karşı nasıl değiştiğini gösterir.

**10. ROC-AUC 0,99'a çok yakın, bu şüpheli değil mi?**
Biz de şüphelendik ve inceledik. CV'de 4. katta iki model de 0,99'u biraz geçti. Nedeni şu: doğrulama satırlarının yarısından fazlasında motor arızadan 100 döngüden daha uzakta ve bunları ayırmak çok kolay. Yalnızca eşiğe yakın satırlara (RUL 15-45) bakınca AUC 0,84 civarına düşüyor. Etiket karıştırma testi 0,50 verdi ve bağımsız test kümesinde AUC 0,98 çıktı. Bu yüzden sızıntı yok, ama bu problemde ROC-AUC iyimser bir ölçüt. F1 ve Recall daha dürüst.

**11. CV ile test sonuçları neden farklı?**
(1) Sınıf oranı farklı: CV'de %15, testte %25 pozitif var. (2) CV bir motorun bütün döngülerini, kolay olanlar dahil, değerlendiriyor. Test ise yalnızca son döngüye bakıyor ve bunlar eşiğe daha yakın, daha zor örnekler. (3) Testte yalnızca 100 örnek ve 25 pozitif var. Tek bir motor Recall'u 0,04 değiştiriyor, bu yüzden sonuç oynak. (4) CV'de model 80 motorla, testte 100 motorla eğitiliyor.

**12. Neden Lojistik Regresyon ve Random Forest?**
İkisi iki uç baz çizgisi. **Lojistik Regresyon** basit, hızlı ve doğrusal. Sensörlerin ağırlıklı toplamına bakar. "Basit bir model ne kadar iyi?" sorusunu cevaplar. **Random Forest** birçok karar ağacının oylamasıdır. Doğrusal olmayan ilişkileri ve sensör etkileşimlerini yakalayabilir, ayar gerektirmeden genelde iyi çalışır. İkisinin yakın sonuç vermesi, sensörler ile etiket arasındaki ilişkinin büyük ölçüde basit olduğunu düşündürüyor.

**13. `class_weight="balanced"` ne işe yarıyor?**
Pozitifler az olduğu için model, her şeye 0 deyip yüksek Accuracy almaya meyilli. `balanced`, az olan sınıftaki hatalara daha büyük ceza veriyor. Böylece model arızaya yakın örnekleri ciddiye alıyor.

**14. Neden ölçekleme yaptınız? Neden Pipeline içinde?**
Sensörlerin birimleri çok farklı. Örneğin s9'un ortalaması yaklaşık 9065, s15'in ortalaması yaklaşık 8,4. Lojistik Regresyon büyük sayılı sensöre haksız ağırlık verebilir. StandardScaler hepsini ortalama 0, std 1 yapıyor. Pipeline içinde olması önemli: ölçekleyici yalnızca eğitim motorlarının ortalamasını öğreniyor, doğrulama verisini hiç görmüyor. Random Forest ölçeklemeye ihtiyaç duymaz, ama tutarlılık için o da aynı yapıda.

**15. Neden 14 sensör? Diğerlerini neden attınız?**
7 sütun hiç değişmiyor (`nunique = 1`). `s6` yalnızca iki değer alıyor. Değişmeyen bir şey arızayı tahmin etmeye yardım edemez. Bunu veriden hesapladık ve beklenen listeyle birebir aynı çıktı.

**16. Sonuçlar tekrarlanabilir mi?**
Evet. Tek bir rastgele tohum (42) kullandık. Scripti iki kez çalıştırdık ve bütün sonuç tabloları birebir aynı çıktı.

**17. Sırada ne var?**
Daha güçlü modeller (XGBoost, SVM, MLP) ve küçük hiperparametre aramaları, 20/30/50 eşik karşılaştırması, son birkaç döngünün ortalaması gibi zaman penceresi özellikleri ve derin modeller (LSTM, 1D-CNN). Bunlar bir sonraki aşamanın konusu.

## 6. Sınırlarımız
- Veri simülasyon. Gerçek motor verisi değil.
- Testte yalnızca 100 örnek var. LR ile RF arasındaki farklar küçük ve anlamlılığı test edilmedi.
- Her satır ayrı bir örnek sayıldı, ama aynı motorun satırları birbirinden bağımsız değil.
- Modeller yalnızca o anki sensör değerine bakıyor, geçmiş döngüleri kullanmıyor.
