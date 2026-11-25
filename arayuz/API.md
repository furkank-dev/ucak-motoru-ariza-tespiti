# Arayüz ↔ backend sözleşmesi

Arayüz backend'den yalnızca iki şey ister. Adresler `/api` altında. En kolayı backend'in `arayuz/` klasörünü `/`
adresinden statik olarak sunması. Başka bir portta çalışacaksa `js/api.js` içindeki `API_TABAN` değiştirilir ve
backend'de CORS açılır.

## `GET /api/saglik`

Arayüz açılışta ve her çalıştırmadan önce bunu dener. 1,5 saniyede cevap yoksa "Backend bağlı değil" yazar.

```json
{"durum": "ok"}
```

## `POST /api/tahmin`

Gönderilen:

```json
{
  "motor_id": 31,
  "sensorler": {"s2": 643.31, "s3": 1598.19, "s4": 1450.0, "s7": 551.48, "s8": 2388.18, "s9": 9090.98,
                "s11": 48.9, "s12": 519.55, "s13": 2388.22, "s14": 8157.84, "s15": 8.5015, "s17": 395.0,
                "s20": 38.53, "s21": 23.1105}
}
```

- `motor_id`: formun doldurulduğu test motoru (1-100).
- `sensorler`: formdaki 14 değer, elle değiştirilmiş olabilir.

Beklenen cevap:

```json
{
  "tahminler": [
    {"id": "lojistik", "ad": "Lojistik Regresyon", "olasilik": 0.12, "tahmin": 0},
    {"id": "rf",       "ad": "Random Forest",      "olasilik": 0.08, "tahmin": 0},
    {"id": "xgboost",  "ad": "XGBoost",            "olasilik": 0.05, "tahmin": 0},
    {"id": "svm",      "ad": "SVM (RBF)",          "olasilik": 0.21, "tahmin": 0},
    {"id": "mlp",      "ad": "MLP",                "olasilik": 0.10, "tahmin": 0},
    {"id": "rnn",      "ad": "RNN",                "olasilik": 0.03, "tahmin": 0},
    {"id": "lstm",     "ad": "LSTM",               "olasilik": 0.01, "tahmin": 0}
  ]
}
```

- `olasilik`: "arızaya yakın" sınıfının olasılığı (`predict_proba(X)[:, 1]`, RNN/LSTM'de sigmoid çıktısı).
- `tahmin`: eşik 0,5 ile 0 ya da 1.
- Liste sırası ekrandaki sıradır. Kimlikler yukarıdaki gibi olursa renkler doğru çıkar, başka model eklenirse gri görünür.
- Hata olursa 4xx/5xx ve düz metin hata mesajı dönmesi yeterli, arayüz mesajı gösterir.

## Modellere ne verilecek

- **Klasik modeller** (lojistik, rf, xgboost, svm, mlp): sadece formdaki 14 değer, tek satır. Sütun sırası
  `scripts/ortak.py` içindeki `SENSORLER` ile aynı olmalı. Ölçekleme Pipeline içinde zaten var.
- **RNN / LSTM**: son 30 döngü gerekiyor. `data/processed/test_FD001_tum_donguler.csv` içinden o motorun son 29 döngüsü
  alınır, formdaki değerler 30. döngü olarak sona eklenir. Motorun geçmişi 30'dan kısaysa ilk satır tekrarlanır
  (`07_rnn.py` içindeki `pencereler` fonksiyonu gibi). Ölçekleme eğitimde fit edilen `StandardScaler` ile yapılır.

Form hiç değiştirilmeden gönderilirse bu, resmî testteki durumun aynısıdır. O yüzden 100 motorun hepsi değiştirilmeden
çalıştırılınca doğru/yanlış sayıları raporumuzdaki resmî test karışıklık matrisleriyle aynı çıkmalı. Tutmuyorsa model
farklı ayarla eğitilmiş demektir; bu iyi bir kontrol.

## Dikkat

- Şu anki scriptler modelleri **kaydetmiyor**. Backend için her model tüm eğitim verisiyle (100 motor) bir kez eğitilip
  kaydedilmeli: klasikler `joblib.dump`, RNN/LSTM `torch.save` (ölçekleyiciyle birlikte). Parametreler
  `outputs/tables/en_iyi_parametreler.csv` içindeki "tüm eğitim" satırlarından alınır, tohum 42.
- `xgboost` ve `torch` paketleri gerekiyor (`requirements.txt` içinde var).
