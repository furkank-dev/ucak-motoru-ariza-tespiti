# Planlar

Baz modellerden (Aşama 1-3 ve etiket tamamlama) sonra neler yapacağımızı ve her birini nasıl yapacağımızı buraya yazıyoruz. Şimdiye kadar yapılanlar `YAPILANLAR.md` dosyasında. Bu dosyadaki hiçbir şey henüz yapılmadı, sonuç sayısı da yok. Sayılar ancak scriptler çalıştıktan sonra `YAPILANLAR.md`'ye yazılacak.

## Nereden başlıyoruz

- Veri: FD001, 100 eğitim motoru (20.631 satır), 100 test motoru (son döngüde 25 pozitif / 75 negatif).
- Etiket: RUL ≤ 30 ise 1. `etiket_20` ve `etiket_50` sütunları da hazır.
- Özellikler: 14 sensörün o döngüdeki ham değeri.
- Baz sonuçlar (karar eşiği 0,5):

| Model | CV F1 | Test F1 | Test Recall |
|---|---|---|---|
| Lojistik Regresyon | 0,8229 ± 0,0357 | 0,8462 | 0,8800 |
| Random Forest | 0,8552 ± 0,0258 | 0,8261 | 0,7600 |

Yeni her model bu tabloyla karşılaştırılacak. Bir yöntem bunları geçmiyorsa bunu da açıkça yazacağız, "olmadı" da bir sonuç.

## Değişmeyen kurallar

Her yeni adımda bunlara uyuyoruz:
- Bölme hep motor bazlı: `GroupKFold(n_splits=5)`, `groups=unit`. Her katta eğitim ve doğrulama motorlarının kesişmediği `assert` ile kontrol edilir.
- Ölçekleme ve her türlü ön işleme (SMOTE, PCA dahil) `Pipeline` içinde, sadece eğitim kısmında fit edilir.
- `cycle`, `unit`, `RUL` ve `etiket*` sütunları özellik olamaz. Pencere özellikleri `cycle`'ı kullanabilir ama sadece sıralamak için, değer olarak modele girmez.
- Hiperparametre araması CV'nin içinde yapılır. Resmî test kümesi sadece en sonda, bir kez, seçilmiş modellerle kullanılır. Testteki sonuca bakıp ayar değiştirmek yok.
- Rastgelelik için tek tohum: `RASTGELE_TOHUM = 42` (`scripts/ortak.py`).
- F1 veya ROC-AUC 0,99'u geçerse sızıntı şüphesi olarak inceleyip raporluyoruz (03'teki `sizinti_incelemesi` gibi).
- Her adımdan sonra kontrol: çıktıları silip temiz çalıştırma, NaN ve 0-1 aralığı kontrolü, aynı tohumla iki çalıştırmada aynı sonuç, `YAPILANLAR.md` güncelleme ve commit. Kontrol geçmeden sonraki adıma geçmiyoruz.

## Sıra

Kolaydan zora, her adım bir öncekinin üstüne kuruluyor:

1. Karar eşiği ve güven aralığı (05)
2. Güçlü klasik modeller ve hiperparametre araması (06)
3. ESIK 20 / 30 / 50 karşılaştırması (07)
4. Pencere (yuvarlanan) özellikler (08)
5. Dengesiz veri: SMOTE ile class_weight karşılaştırması (09)
6. Derin modeller: LSTM ve 1D-CNN (10)
7. FD003 ile genelleme denemesi (11)
8. Anomali tespiti: Isolation Forest ve PCA (12)

Numaralar script adları için öneri. Zaman yetmezse 1-4 asıl kısım, 5-8 ek.

## 1. Karar eşiği ve güven aralığı (`05_esik_ve_guven.py`)

**Neden:** Baz modellerde karar eşiği 0,5 sabit. LR fazla alarm veriyor, RF arıza kaçırıyor. Eşiği kaydırarak bu dengeyi ayarlayabiliriz. Ayrıca testte sadece 100 örnek var, LR ile RF arasındaki farkın gerçek olup olmadığını bilmiyoruz.

**Nasıl:**
- 03'teki OOF olasılıklarını kullanıp eşiği 0,05 ile 0,95 arasında 0,05 adımla tarıyoruz. Her eşik için Precision, Recall, F1 hesaplıyoruz.
- Eşiği sadece CV'deki OOF tahminlerine göre seçiyoruz (ör. en yüksek F1 veren eşik, ya da Recall ≥ 0,90 şartıyla en yüksek Precision). Seçilen eşik sonra teste aynen uygulanıyor.
- Precision-Recall eğrisi çizip seçilen noktayı işaretliyoruz.
- Test için bootstrap güven aralığı: 100 test motorunu yerine koyarak 1000 kez yeniden örnekleyip F1 ve Recall'un %95 aralığını çıkarıyoruz. LR ile RF'yi aynı örneklerde karşılaştırıp farkın aralığına bakıyoruz (aralık 0'ı içeriyorsa fark anlamlı değil diyoruz).

**Çıktılar:** `esik_taramasi.csv`, `guven_araliklari.csv`, `08_pr_egrisi.png`.

## 2. Güçlü klasik modeller (`06_guclu_modeller.py`)

**Neden:** LR ve RF iki uç baz çizgisi. Daha güçlü modellerin sonucu ne kadar değiştirdiğini görmek istiyoruz.

**Modeller ve aranacak ayarlar (küçük ızgaralar):**
- **XGBoost** (`xgboost` paketi `.venv`'ye kurulacak, `requirements.txt`'ye eklenecek): `n_estimators` 100/300, `max_depth` 3/5, `learning_rate` 0,05/0,1. Dengesizlik için `scale_pos_weight` = negatif / pozitif oranı (eğitim katından hesaplanır).
- **SVM** (`SVC(probability=True)` ya da hız için `LinearSVC` + kalibrasyon): `C` 0,1/1/10, `kernel` linear/rbf, `class_weight="balanced"`. 20.000 satırda RBF SVM yavaş olabilir, çok yavaşsa bunu yazıp örnekleme ya da doğrusal SVM ile devam ederiz.
- **MLP** (`MLPClassifier`): gizli katman (32,) / (64, 32), `alpha` 1e-4/1e-3, `early_stopping=True`. MLP'de `class_weight` yok, bunu not ediyoruz.
- Karşılaştırma için LR ve RF'ye de aynı şekilde küçük bir arama (LR: `C`; RF: `max_depth`, `min_samples_leaf`).

**Nasıl:**
- İç içe CV: dışta 5 katlı GroupKFold, içte her eğitim katında `GridSearchCV(cv=GroupKFold(3), scoring="f1")`. Böylece CV skoru aramadan etkilenmiyor. İç aramaya `groups` parametresi ayrıca verilmeli, yoksa iç bölme motor bazlı olmaz.
- Her model için seçilen en iyi ayarları ve dış kat skorlarını kaydediyoruz.
- Son olarak her modeli bütün eğitim verisinde tek bir aramayla yeniden ayarlayıp resmî testte değerlendiriyoruz.
- Süreyi de yazıyoruz, sunumda "daha iyi ama ne kadar pahalı" sorusu gelebilir.

**Çıktılar:** `sonuclar_guclu.csv`, `sonuclar_guclu_katlar.csv`, `en_iyi_ayarlar.csv`, karışıklık matrisleri ve ROC grafiği.

## 3. ESIK 20 / 30 / 50 karşılaştırması (`07_esik_karsilastirma.py`)

**Neden:** Sonuçların seçtiğimiz 30 eşiğine ne kadar bağlı olduğunu görmek. Etiketler (`etiket_20`, `etiket_50`) zaten hazır.

**Nasıl:**
- Her eşik için aynı modeller (LR, RF ve 06'da en iyi çıkan model), aynı 5 katlı motor bazlı CV ve resmî test.
- Pozitif oranları farklı: eğitimde %10,18 / %15,03 / %24,72, testte 16 / 25 / 33 pozitif. Bu yüzden Accuracy'yi değil F1, Recall ve PR-AUC'yi karşılaştırıyoruz ve her eşikte "hep 0" baz çizgisini de yazıyoruz.
- Beklentimiz: eşik büyüdükçe pozitif artıyor ve problem kolaylaşabilir ama uyarı daha erken, daha belirsiz bir anda veriliyor. Bunu yoruma yazacağız, sonucu önceden varsaymayacağız.

**Çıktılar:** `sonuclar_esik.csv`, eşiklere göre F1 / Recall çubuk grafiği.

## 4. Pencere özellikleri (`08_pencere_ozellikleri.py`)

**Neden:** Modeller şu an sadece o döngüdeki sensör değerine bakıyor. Veri gürültülü ve yıpranma son 50-100 döngüde hızlanıyor. Geçmiş birkaç döngünün ortalaması ve eğimi gürültüyü azaltıp eğilimi yakalayabilir.

**Nasıl:**
- Her motor kendi içinde `cycle`'a göre sıralanıyor, `groupby("unit")` ile hesaplanıyor. Bir motorun penceresi asla başka bir motora taşmıyor.
- Her sensör için pencere boyu w = 5, 10, 20 (ya da sadece biri, önce w=10):
  - yuvarlanan ortalama (`rolling(w, min_periods=1).mean()`)
  - yuvarlanan standart sapma
  - eğim: son w döngüdeki doğrusal eğim (`np.polyfit` ya da fark / w)
- Pencere sadece geçmişe bakıyor (o döngü ve öncesi). Geleceğe bakan pencere sızıntı olur, bunu `assert` ile değil ama kodda `center=False` ile garanti ediyoruz ve küçük bir örnekte elle kontrol ediyoruz.
- Test için önemli: son döngünün pencere özelliğini hesaplamak için test motorlarının önceki döngüleri gerekiyor. Bunun için `test_FD001_tum_donguler.csv` hazır (04'te üretildi). Özellikler bütün test döngülerinde hesaplanıp sonra sadece son döngü değerlendiriliyor.
- Kısa motorlarda ilk döngülerde pencere eksik kalıyor (`min_periods=1`). En kısa test motoru kaç döngü, ona bakıp yazacağız.
- Aynı modellerle (LR, RF, en iyi model) ham özelliklerle pencere özelliklerini karşılaştırıyoruz.

**Çıktılar:** `sonuclar_pencere.csv`, ham ve pencere karşılaştırma grafiği, bir örnek motorda ham ve yumuşatılmış sensör grafiği.

## 5. SMOTE ile dengesiz veri (`09_dengesiz_veri.py`)

**Neden:** Şu an dengesizliği `class_weight="balanced"` ile çözüyoruz. SMOTE (az sınıftan yapay örnek üretme) daha iyi mi, görmek istiyoruz.

**Nasıl:**
- `imbalanced-learn` paketi kurulacak. SMOTE `imblearn.pipeline.Pipeline` içinde olmalı ki sadece eğitim katında uygulansın. Doğrulama ve test verisine asla yapay örnek eklenmiyor. Bu, en sık yapılan sızıntı hatası.
- Karşılaştırma: hiçbir şey / `class_weight="balanced"` / SMOTE / SMOTE + `class_weight`. Aynı CV, aynı modeller.
- Not: SMOTE farklı motorların satırları arasında ara nokta üretebilir. Bu fiziksel olarak anlamsız örnekler doğurabilir, yorumda bunu söyleyeceğiz.

**Çıktılar:** `sonuclar_dengesiz.csv`.

## 6. Derin modeller: LSTM ve 1D-CNN (`10_derin_modeller.py`)

**Neden:** Pencere özelliklerini elle tanımlamak yerine modelin dizi üzerinden kendisinin öğrenmesi.

**Nasıl:**
- Paket: PyTorch (`torch`, CPU sürümü yeterli). Python 3.14 için uygun sürümün olup olmadığına önce bakacağız. Yoksa bunu yazıp ya ayrı bir sanal ortam (daha eski Python) kuracağız ya da bu adımı ertelenmiş sayacağız.
- Girdi: her satır için son L döngünün 14 sensörü, şekil (L, 14). L = 30 ile başlıyoruz. L'den kısa geçmişler başa ilk değer tekrarlanarak dolduruluyor.
- Ölçekleme yine sadece eğitim katının motorlarıyla fit ediliyor, sonra dizilere uygulanıyor.
- LSTM: 1-2 katman, 32-64 gizli birim, sonda sigmoid. 1D-CNN: 2 konvolüsyon katmanı + global ortalama havuzlama.
- Kayıp: pozitif sınıfa ağırlık veren `BCEWithLogitsLoss(pos_weight=...)`.
- Erken durdurma için doğrulama kümesi eğitim motorlarının içinden ayrılıyor (yine motor bazlı), dış kattaki doğrulama motorları erken durdurmada kullanılmıyor.
- Tohum: `torch.manual_seed(42)`, numpy ve random da 42. CPU'da bile tam tekrarlanabilirlik garanti değil, iki çalıştırma arasında fark çıkarsa ne kadar olduğunu yazacağız.
- 5 kat × birkaç epoch CPU'da uzun sürebilir. Süre yazılacak.

**Çıktılar:** `sonuclar_derin.csv`, kayıp eğrileri.

## 7. FD003 ile genelleme (`11_fd003.py`)

**Neden:** FD003 de tek çalışma koşullu ama iki arıza türü var (FD001'de bir tane). Modelimizin farklı bir arıza türüne ne kadar dayanıklı olduğunu görmek.

**Nasıl:**
- Veri hazır: `data/processed/yedek/` altında (24.720 eğitim satırı, 100 motor; test son döngüde ESIK=30 için 20 pozitif).
- Sabit sütunları FD003 için yeniden veriden hesaplıyoruz, FD001 listesi aynen tutmayabilir.
- İki deneme: (a) FD003'ün kendi eğitimiyle eğit, kendi testinde değerlendir; (b) FD001 ile eğitilmiş modeli FD003 testinde dene.
- FD002 ve FD004 altı çalışma koşullu. Onlar için önce koşula göre normalizasyon gerekir, bu planın dışında.

**Çıktılar:** `sonuclar_fd003.csv`.

## 8. Anomali tespiti: Isolation Forest ve PCA (`12_anomali.py`)

**Neden:** Etiket kullanmadan "normalden sapma" ile arızayı yakalayabilir miyiz? Gerçek hayatta arıza etiketi az olur, bu yüzden ilginç bir karşılaştırma.

**Nasıl:**
- Isolation Forest sadece sağlıklı satırlarla (ya da her motorun ilk N döngüsüyle) eğitiliyor. Anomali skoru olasılık yerine kullanılıp ROC-AUC ve en iyi eşikte F1 hesaplanıyor. Eşik yine CV'den seçiliyor.
- PCA: sağlıklı satırlarla fit edilip yeniden kurma hatası anomali skoru olarak kullanılıyor. Ayrıca ilk 2 bileşende satırları RUL'a göre renklendirip bir grafik çiziyoruz, sunumda görsel olarak iyi duruyor.
- Bunlar denetimli modellerden kötü çıkabilir. Amaç onları geçmek değil, etiket olmadan ne kadar yaklaşılabildiğini göstermek.

**Çıktılar:** `sonuclar_anomali.csv`, `pca_2b.png`.

## Her adımda dokümantasyon

- `YAPILANLAR.md`: yeni başlık, gerçek sayılar, kararlar, kontrol sonuçları.
- `ANLATIM.md`: sözlüde sorulabilecek yeni sorular (ör. "XGBoost neden daha iyi/kötü?", "SMOTE neden sızıntıya yol açabilir?", "pencere neden sadece geçmişe bakıyor?").
- `README.md`: yeni scriptler çalıştırma sırasına ve yeni paketler kurulum kısmına eklenecek.
- Bu dosyada yapılan adımı "yapıldı" diye işaretleyip ayrıntıyı `YAPILANLAR.md`'ye bırakacağız.

## Açık sorular

- Teslim tarihi ve sunum süresi: hangi adımlara yetişeceğimizi buna göre seçeceğiz.
- Sunumda ana model hangisi olacak: en yüksek F1 mi, yoksa Recall şartıyla seçilen mi? Ekip olarak karar vermemiz lazım.
- Derin modeller için Python 3.14'te PyTorch desteği (bkz. 6. adım).
- Proje tanımında 30 eşiğinin literatürde yaygın olduğu yazıyor, kaynağına bakmadık. Raporda kullanacaksak bulmamız gerekiyor.
