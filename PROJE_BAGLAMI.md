# Akıllı Arıza Tespit Sistemi — Proje Bağlamı

> **Bu dosya kimin için?** Gruptaki herkesin kendi Claude'una (veya başka bir yapay zekâ asistanına) projenin bütün bağlamını vermesi için hazırlandı. Dosyayı sohbetin başına ekleyip "Bu projede bana yardım edeceksin, önce bu dosyayı oku" demen yeterli.
>
> **Son güncelleme:** 23 Kasım 2026
> **Hazırlayan:** Kaya (projenin uygulama tarafını yürüten kişi)

---

## 0. Bu dosyayı okuyan yapay zekâ için kurallar

1. **Sayı uydurma.** Model sonuçları (accuracy, F1, ROC-AUC vb.) henüz bu dosyada yok. Bir sayı gerekiyorsa "script çıktısından okunmalı" de, tahmin verme.
2. **Kapsamı genişletme.** Bölüm 5'teki "istenmeden yapılmayacaklar" listesindeki şeyleri (derin öğrenme, SMOTE vb.) önerebilirsin ama kendiliğinden ekleme; karar gruba ait.
3. **Rapor metnini öğrenciler yazacak.** Rapor Turnitin'den geçecek ve yapay zekâ tespit oranı **%40'ı geçmemeli**. Yapı, kontrol, açıklama ve geri bildirim konusunda yardım et; raporun paragraflarını baştan sona sen yazma.
4. **Kaynak uydurma.** Literatürde bir makale önerirsen yazar/yıl/DOI bilgisini doğrulanabilir şekilde ver; emin değilsen "doğrulanmalı" diye işaretle.
5. **Bu dosyada "doğrulanmalı" veya "beklenen" diye işaretlenen her şey kesin değildir.** Kesin bilgi; kodun gerçek çıktısı, NASA'nın veri dokümantasyonu ve hocanın yönergesidir.

---

## 1. Proje özeti (tek paragraf)

Yapay Zeka dersi için grupça bir "**Akıllı Arıza Tespit Sistemi**" projesi yapıyoruz. NASA'nın **C-MAPSS turbofan (uçak motoru) veri setinin FD001** alt kümesini kullanıyoruz. Her motorun sensör ölçümlerinden, **motorun 30 döngü (cycle) içinde arızalanıp arızalanmayacağını** tahmin eden bir **ikili sınıflandırma** modeli kuruyoruz (etiket: kalan faydalı ömür **RUL ≤ 30 → 1**, değilse 0). Klasik makine öğrenmesi modelleri (Lojistik Regresyon, Random Forest, XGBoost, SVM, MLP) karşılaştırılacak; motor bazlı bölme ile veri sızıntısı engellenecek; eşik değerinin (20/30/50) sonuçlara etkisi incelenecek. Proje hem rapor hem sunum olarak teslim edilecek ve **final sınavında projeyle ilgili sorular çıkacak**.

---

## 2. Ders yönergesi (hocanın istedikleri)

Tam metin: `Yonerge_Proje_Teslim_ve_Sunum_Kurallari.pdf` (bu klasörde).

**Teslim:** Rapor, kaynak kod ve sunum dosyaları e-postayla gönderilecek. Değerlendirmede rapor içeriği, deneysel çalışmalar, sonuçlar ve **sunumda konuya hâkimiyet** dikkate alınacak.

**Sunumda olması gerekenler:**
1. Araştırma sorusu ve amaç
2. Literatür taraması: **20–30 akademik çalışma**, yöntem ve bulgu özetleri
3. Kullanılan algoritmalar ve tercih nedenleri
4. Veri setinin tanıtımı: kaynak, özellikler, örnek sayısı, sınıf dağılımı, grafikler, tanımlayıcı istatistikler (ortalama, medyan, mod vb.)
5. Model seçimi ve hiperparametreler
6. Deneysel sonuçlar: tablo ve grafiklerle; **Accuracy, Precision, Recall, F1-Score ve ROC-AUC** zorunlu
7. Literatürle karşılaştırma
8. Genel değerlendirme: araştırma sorusunun cevabı, güçlü yönler, sınırlılıklar

**Rapor bölümleri:** Özet → Giriş → Literatür Taraması → Deneysel Yöntemler → Bulgular → Tartışma → Sonuç → Kaynakça (akademik yazım kurallarına uygun).

**Özgünlük:** Turnitin ile benzerlik ve yapay zekâ kontrolü yapılacak; **AI oranı %40'ı aşmamalı**.

**Sunum:** Belirlenen gün ve saatte yapılacak; yöntem, veri seti ve sonuçlarla ilgili sorular sorulabilir.

---

## 3. Karar geçmişi (nasıl buraya geldik)

| Tarih | Ne oldu |
|---|---|
| 14 Kasım 2026 | Yönerge geldi, konu "Akıllı Arıza Tespit Sistemi" olarak seçildi. Sunuma ~2 hafta vardı. Şart: hazır veri seti, aşırı karmaşık olmayan ama başlıkla doğrudan ilgili bir proje (sınavda sorulacağı için). |
| 14 Kasım | İlk ana aday **UCI AI4I 2020** (freze makinesi, sentetik, %3,4 arıza) olarak araştırıldı. Alternatifler de incelendi: Kaggle Electrical Fault (iletim hattı), UCI Steel Plates Faults, CWRU Bearing, Kaggle Machine Predictive Maintenance. |
| 14 Kasım | "Daha niş" adaylar arandı ve veri kalitesi değerlendirildi: UCI MetroPT-3 (metro kompresörü), Engine Health (motor sağlığı; kaynağı tutarsız), UCI Hydraulic Systems (sonuçlar tavana vuruyor), Kaggle Pump Sensor (sadece 7 arıza, eksik sütunlar), Mendeley GPVS-Faults (güneş enerjisi; ağır sinyal işleme). Alternatif veri setleri için ChatGPT'ye verilecek arama promptları da hazırlandı. |
| Sonrasında | Proje **uçak motoru (turbofan) arıza tespiti** yönüne kaydı ve **NASA C-MAPSS FD001** seçildi. |
| 22 Kasım 2026 | Uygulamaya başlandı. Veri indirme, etiketleme ve model denemeleri Kaya'nın bilgisayarında (`~/dev` altında) **Claude Code** ile yapılacak. Kapsam önce genişletildi, sonra sadeleştirildi: **derin modeller ve diğer zor kısımlar çıkarıldı, klasik ML ile devam kararı alındı**. Claude Code için **görev promptu v3** (`claude_code_gorev_v3.md`) yazıldı. |
| Sıradaki | Planın yalnızca **Gün 1–6** kısmını (veri, etiket, inceleme, Lojistik Regresyon + Random Forest, motor bazlı bölme) yapacak daha dar bir Claude Code promptu hazırlanacak. |

> Not: `claude_code_gorev_v3.md` dosyasının tam metni bu klasörde yok; gerekiyorsa Kaya'dan iste. Aşağıdaki bölüm onun özeti.

---

## 4. Veri seti: NASA C-MAPSS FD001

| Özellik | Bilgi |
|---|---|
| Kaynak | NASA Prognostics Center of Excellence, Prognostics Data Repository. Tanıtan çalışma: Saxena vd., 2008 (PHM08 konferansı) |
| Nedir | Simüle edilmiş turbofan motorlar. Eğitim setindeki her motor sağlam başlar, zamanla bozulur ve arızalanana kadar ("run-to-failure") kaydedilir. Test setinde kayıt arızadan önce bir noktada kesilir; gerçek kalan ömür ayrı bir dosyada verilir |
| FD001 | 1 çalışma koşulu, 1 arıza modu (yüksek basınç kompresörü / HPC bozulması). C-MAPSS'in en basit alt kümesi |
| Boyut | Eğitim: 100 motor, 20.631 satır. Test: 100 motor |
| Sütunlar | Motor numarası (unit), döngü (cycle), 3 çalışma ayarı, 21 sensör |
| Veri tipi | Sentetik (simülasyon). Gerçek uçak motoru verisi değil → sınırlılık olarak yazılacak |
| Not | Bazı sensörlerin FD001'de neredeyse sabit olduğu literatürde sık belirtilir. **Doğrulanmalı:** 02 inceleme scriptinin çıktısına bakılacak |

**Etiket:** Her satır için RUL = (o motorun son döngüsü) − (o satırın döngüsü). **RUL ≤ 30 ise 1** ("30 döngü içinde arıza riski var"), değilse 0.

**Beklenen sınıf dağılımı (eğitim):** Her motorun son 31 satırı (RUL 0…30) pozitif olur → 100 × 31 = 3.100 pozitif / 20.631 ≈ **%15**. *Bu bir hesap; 01 scriptinin gerçek çıktısıyla doğrulanacak.*

**Neden regresyon değil sınıflandırma?** C-MAPSS'in asıl görevi RUL tahmini, yani regresyon. Yönerge ise sınıflandırma metrikleri (Accuracy, Precision, Recall, F1, ROC-AUC) istiyor. Bu yüzden RUL bir eşikle ikili sınıfa çevrildi. Eşiğin seçimi keyfi olduğu için 20/30/50 eşikleriyle duyarlılık analizi yapılacak.

---

## 5. Uygulama planı (Claude Code görev promptu v3'ün özeti)

**Çalışma ortamı:** Kaya'nın bilgisayarı, `~/dev` altı, Claude Code. Python + scikit-learn + XGBoost.

**5 script:**

| Script | İş |
|---|---|
| `01` | Veriyi indir, RUL hesapla, RUL ≤ 30 etiketini üret |
| `02` | Veri inceleme (EDA): tanımlayıcı istatistikler, sınıf dağılımı, sensör grafikleri, sabit sensörler |
| `03` | Baz modeller: **Lojistik Regresyon + Random Forest** |
| `04` | Gelişmiş klasik modeller: **XGBoost, SVM, MLP** + küçük bir GridSearchCV |
| `05` | Eşik duyarlılığı (**20 / 30 / 50**) + tüm modellerin karşılaştırması |

**Doğruluk kuralları (pazarlıksız):**
- Veri uydurma yok; raporlanan her sayı gerçek çıktıdan okunacak.
- `cycle` ve `unit` **özellik olarak kullanılamaz** (unit sadece kimlik; cycle ise zamanı doğrudan kodladığı için modelin sensörlere değil "kaçıncı döngüde olduğuna" bakarak kısa yoldan öğrenmesine yol açar).
- Bölme **motor bazlı** yapılacak (**GroupKFold**): aynı motorun satırları hem eğitim hem test setine düşmeyecek.
- Ölçekleme (scaler) **Pipeline içinde** olacak; böylece test verisinin bilgisi eğitime sızmaz.

**Claude Code'un çalışma biçimi:** Durmadan çalışacak; her aşamanın sonunda kendini kontrol edecek (temiz çalıştırma, çıktı ve sayı kontrolü, sızıntı kontrolü, aynı tohumla (seed) tekrar, mantık kontrolü). Yalnızca şu durumlarda duracak: indirme hatası, sayı uyuşmazlığı, sızıntı şüphesi, çözülemeyen hata.

**Raporlama çıktıları:**
- `YAPILANLAR.md`: kronoloji, her dosyanın envanteri, grafik ve tabloların nasıl okunacağı, sonuçlar, doğrulanamayanlar
- `ANLATIM.md`: sunumda veya sınavda sözlü olarak sorulabilecek sorular ve cevapları
- Kod içi yorumlar kısa; uzun açıklamalar bu md dosyalarında

**İstenmeden yapılmayacaklar (kapsam dışı, ancak açık onayla eklenebilir):**
- Derin modeller (CNN, LSTM) ve kayan pencere (sliding window)
- SMOTE
- Yuvarlanan (rolling) özellikler
- FD003 (ikinci alt küme)
- Anomali tespiti baz çizgisi

**Bekleyen karar:** Hocaya **kodda yapay zekâ (Claude Code) kullanımının uygun olup olmadığı** soruldu. **Cevap gelmeden scriptler çalıştırılmayacak.**

---

## 6. Ekip ve iş bölümü

| Kişi | Görev |
|---|---|
| Kaya | Uygulama: veri, etiketleme, modeller, Claude Code yönetimi |
| Bir grup arkadaşı | **Literatür taraması** (20–30 çalışma). Kaya ona veri seti linkini ve proje planını iletecek |
| Diğerleri | *Belirlenecek:* rapor bölümleri, sunum slaytları, prova |

> Ekip içi rol dağılımının geri kalanı netleşmedi; bu tabloyu güncelleyin.

---

## 7. Yapılacaklar ve açık konular

- [ ] Hocadan kodda yapay zekâ kullanımı hakkında cevap
- [ ] Gün 1–6 için dar Claude Code promptu (veri, etiket, inceleme, LR + RF, motor bazlı bölme)
- [ ] 01–05 scriptlerinin çalıştırılması ve çıktıların kontrolü
- [ ] Literatür taraması (20–30 kaynak) + literatürle karşılaştırma tablosu
- [ ] Rapor (öğrencilerin kendi cümleleriyle; AI oranı < %40)
- [ ] Sunum slaytları + prova + sözlü soru hazırlığı (`ANLATIM.md` ve bu dosyadaki Bölüm 8)

**Literatür için başlangıç noktaları** (künyeler doğrulanmalı):
- Saxena, Goebel, Simon, Eklund (2008), "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM 2008 → veri setini tanıtan çalışma
- NASA Prognostics Data Repository → veri setinin resmî kaynağı
- C-MAPSS üzerinde RUL tahmini yapan çalışmalar (ör. CNN tabanlı: Babu vd. 2016; LSTM tabanlı: Zheng vd. 2017). Bunlar regresyon yapıyor. Sonuçlar bizim sınıflandırma sonuçlarımızla birebir kıyaslanamaz; tartışma bölümünde bu fark açıkça yazılmalı
- C-MAPSS'te RUL'u eşikle sınıflandırmaya çeviren çalışmalar → doğrudan karşılaştırma için en değerli kaynaklar, özellikle aranmalı

---

## 8. Hocanın sorabileceği sorular ve cevap taslakları

**Kamera ya da özel donanım gerekiyor mu?**
Hayır. Veri, sensör ölçümlerinden oluşan metin dosyaları (tablo). Normal bir bilgisayarda Python ile çalışıyor; ekran kartı gerekmiyor.

**Veri gerçek mi?**
Hayır. NASA'nın C-MAPSS simülatörüyle üretilmiş sentetik veri. Gerçek motor arıza verisi gizli ve az olduğu için bu alanda standart kıyaslama (benchmark) veri seti olarak kullanılıyor. Sınırlılıklar bölümünde belirtilecek.

**Tek bir motor mu var?**
Hayır. Eğitim setinde 100 farklı motor var. Her biri farklı başlangıç aşınması ve üretim farklılıklarıyla başlıyor ve arızaya kadar izleniyor. Model, motorlar arası bu farklılığa rağmen genel bozulma desenini öğrenmek zorunda.

**Neden yapay zekâ? Sensör belli bir eşiği geçince alarm veren kural tabanlı bir sistem yetmez mi?**
Bozulma yavaş, gürültülü ve tek bir sensörde değil birçok sensöre yayılmış biçimde ortaya çıkıyor. Her motorun başlangıç değerleri de farklı. Bu yüzden "sensör X şu değeri geçerse alarm" gibi tek eşikli bir kural ya geç kalır ya da çok yanlış alarm üretir. Kural yazmak için hangi sensörlerin hangi kombinasyonda, hangi eşikte önemli olduğunu bilmek gerekir; gerçekte bunu kimse bilmiyor. Makine öğrenmesi bu çok değişkenli deseni geçmiş veriden kendisi öğreniyor. Karar ağacı gibi modeller öğrendiği kuralları açıkça gösterebiliyor; yani yapay zekâ kuralların yerine geçmiyor, kuralları veriden buluyor.
*(Opsiyonel fikir: Basit bir eşik sistemini baz çizgisi olarak ML'le karşılaştırmak bu cevabı sayılarla destekler. v3 kapsamında yok; eklenip eklenmeyeceğine grup karar verecek.)*

**Belirsizlik tam olarak nerede?**
Hangi sensörlerin önemli olduğu, hangi eşiklerde arızanın yaklaştığı, ölçüm gürültüsü ve motordan motora değişen başlangıç koşulları. Bunların hiçbiri önceden bilinmiyor.

**Model başka motorlarda da çalışır mı?**
Eğitilen model FD001'e (tek çalışma koşulu, tek arıza modu) özel. Ama kurulan yöntem (etiketleme → motor bazlı bölme → Pipeline → model karşılaştırma) genel. Başka bir motor tipinin etiketli run-to-failure verisi olursa aynı süreç o motor için yeni bir model üretir. Şart: geçmiş arıza verisinin olması.

**Neden RUL ≤ 30?**
Asıl problem regresyon; yönerge sınıflandırma metrikleri istediği için RUL eşikle ikili sınıfa çevrildi. "30 döngü kala bakım planlamak için yeterli bir uyarı süresi" yorumuyla seçildi. Eşik keyfi olduğu için 20 ve 50 de denenip sonuçların eşiğe ne kadar duyarlı olduğu gösterilecek.

**Neden accuracy tek başına yetmez?**
Pozitif sınıf yaklaşık %15 (doğrulanacak). Her satıra "sağlam" diyen işe yaramaz bir model bile yaklaşık %85 accuracy alır. Bu yüzden recall (arızaları kaçırmama), precision (yanlış alarm), F1 ve ROC-AUC birlikte raporlanıyor. Bu problemde arızayı kaçırmak (yanlış negatif) yanlış alarmdan daha pahalı.

**Neden motor bazlı bölme (GroupKFold)?**
Aynı motorun ardışık döngüleri birbirine çok benzer. Satırlar rastgele bölünürse aynı motor hem eğitimde hem testte olur; model motoru "ezberler" ve skor yapay olarak şişer. Motor bazlı bölmede test motorları modelin hiç görmediği motorlardır; gerçek kullanıma daha yakındır.

**Neden `cycle` ve `unit` özellik değil?**
`unit` yalnızca bir kimlik numarası; anlamı yok, ezberlemeye yol açar. `cycle` ise zamanı doğrudan kodluyor; model sensörlere bakmak yerine "kaçıncı döngüdeyim" bilgisinden kısa yoldan tahmin yapabilir. Motorların ömür uzunlukları farklı olduğu için bu yanıltıcıdır ve projenin amacı olan "sensörden arıza tespitini" ölçmez.

**Neden ölçekleme Pipeline içinde?**
Scaler tüm veriye uygulanırsa test setinin ortalama ve standart sapması eğitime sızar. Pipeline içinde, her katmanda yalnızca eğitim verisinden öğrenilip teste uygulanır.

**Neden derin öğrenme ya da SMOTE yok?**
Kapsam bilinçli olarak sade tutuldu: 2 haftalık süre, sınavda açıklanabilirlik ve klasik modellerin bu boyuttaki tabular veride güçlü olması. Sınırlılıklar ve gelecek çalışmalar bölümünde LSTM/CNN ve kayan pencere yöntemleri önerilecek.

**Proje yönergeye uygun mu?**
Evet. Araştırma sorusu var, klasik ML algoritmaları gerekçeleriyle karşılaştırılıyor, veri seti istatistikleriyle tanıtılıyor, hiperparametreler GridSearchCV ile ayarlanıyor, beş zorunlu metrik raporlanıyor, literatür C-MAPSS üzerinde çok geniş ve karşılaştırma yapılabiliyor, sınırlılıklar net.

---

## 9. Kısa sözlük

| Terim | Anlamı |
|---|---|
| C-MAPSS | Commercial Modular Aero-Propulsion System Simulation; NASA'nın turbofan motor simülatörü |
| FD001 | C-MAPSS'in 4 alt kümesinden en basiti (1 koşul, 1 arıza modu) |
| RUL | Remaining Useful Life: arızaya kalan döngü sayısı |
| Cycle (döngü) | Motorun bir çalışma çevrimi (uçuş benzeri) |
| Run-to-failure | Sağlam başlayıp arızaya kadar kaydedilen motor geçmişi |
| GroupKFold | Aynı gruba (motora) ait satırları aynı katmanda tutan çapraz doğrulama |
| Pipeline | Ön işleme ve modeli tek bir nesnede birleştiren scikit-learn yapısı; sızıntıyı önler |
| Veri sızıntısı | Test bilgisinin eğitime karışıp skoru yapay olarak şişirmesi |
| Recall | Gerçek arızaların ne kadarının yakalandığı |
| Precision | Verilen arıza alarmlarının ne kadarının doğru olduğu |
| ROC-AUC | Modelin pozitif ve negatifi tüm eşiklerde ayırma başarısı (0,5 = rastgele, 1 = mükemmel) |
