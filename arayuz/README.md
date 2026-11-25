# Arayüz

Sınıfta canlı gösterim için basit bir araç: test verisinden bir motor seçilir, istenirse sensör değerleri elle
değiştirilir, "Tüm modelleri çalıştır" deyince 7 modelin kararı yan yana görünür. Her çalıştırma alttaki turlar
tablosuna eklenir ve hangi modelin kaç turda doğru bildiği sayılır.

Düz HTML, CSS ve JavaScript; kütüphane ya da derleme yok. Tema sunumla aynı (siyah + altın).

## Çalıştırma

Backend `arayuz/` klasörünü sunuyorsa onun adresi açılır. Sadece arayüze bakmak için:

```bash
cd arayuz
python -m http.server 8000
```

Sonra `http://localhost:8000`. Backend yokken form doldurulabilir ama modeller çalışmaz, sağ üstte
"Backend bağlı değil" yazar. Backend'in uyması gereken biçim `API.md` içinde.

## Nasıl kullanılır

1. **Test motoru** listesinden bir motor seçilir (ya da **Rastgele**). Formda o motorun son döngüsünün 14 sensör değeri
   dolar. Gerçek durum çalıştırmadan önce gösterilmez, sınıf önce tahmin edebilsin diye.
2. İstenen değerler değiştirilir. Değişen kutu altın renkli olur; eğitimde görülen aralığın dışına çıkan değer kırmızı.
   **Sıfırla** motorun gerçek değerlerine döner.
3. **Tüm modelleri çalıştır.** Sağda her modelin arıza olasılığı ve kararı çıkar. Değer değiştirilmediyse gerçek durum
   da açıklanır ve her model için doğru / yanlış yazar.
4. Turlar tablosu sayfa yenilense de kalır (tarayıcıda saklanır). **Tabloyu temizle** ile sıfırlanır.

## Dosyalar

- `index.html`, `stil.css`: sayfa ve tema
- `js/api.js`: backend'le konuşan tek yer (`API_TABAN` burada)
- `js/uygulama.js`: form, sonuç kartları ve turlar tablosu
- `veri/veri.json`: 100 test motorunun son döngüsü ve eğitimdeki sensör aralıkları
- `veri_uret.py`: `veri.json`'u `data/processed` içinden üretir (`.venv/bin/python arayuz/veri_uret.py`)
