"""Aşama 1: FD001 verisini indir, oku, doğrula, etiketle."""
import hashlib
import io
import urllib.request
import zipfile

import pandas as pd

from ortak import HAM, ISLENMIS, ESIK, ham_oku, etiketle

DOSYALAR = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]
KAYNAKLAR = [
    ("PHM S3 (zip)", "zip",
     "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"),
    ("data.nasa.gov (zip)", "zip", "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"),
    ("Hugging Face", "dosya",
     "https://huggingface.co/datasets/SoyVitou/NASA-C-MAPSS-Turbofan-Engine/resolve/main/"),
    ("GitHub edwardzjl", "dosya",
     "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master/"),
]
KAYNAK_KAYDI = HAM / "kaynak.txt"


def indir(url):
    """URL'yi indirip bayt olarak döndürür."""
    istek = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(istek, timeout=300) as yanit:
        return yanit.read()


def zipten_cikar(icerik):
    """Zip (iç içe zip dahil) içinden FD001 dosyalarını bulup döndürür."""
    z = zipfile.ZipFile(io.BytesIO(icerik))
    bulunan = {}
    for ad in z.namelist():
        kisa = ad.split("/")[-1]
        if kisa in DOSYALAR:
            bulunan[kisa] = z.read(ad)
        elif kisa.lower().endswith(".zip"):
            bulunan.update(zipten_cikar(z.read(ad)))
    return bulunan


def veriyi_getir():
    """Kaynakları sırayla dener; ilk çalışanla dosyaları data/raw'a yazar."""
    if all((HAM / d).exists() for d in DOSYALAR) and KAYNAK_KAYDI.exists():
        print("Ham dosyalar zaten var, indirme atlandı:", KAYNAK_KAYDI.read_text().strip())
        return
    for ad, tur, url in KAYNAKLAR:
        print(f"Deneniyor: {ad} -> {url}")
        try:
            if tur == "zip":
                icerik = indir(url)
                bulunan = zipten_cikar(icerik)
                (HAM / url.split("/")[-1].replace("+", "_")).write_bytes(icerik)
            else:
                bulunan = {d: indir(url + d) for d in DOSYALAR}
        except Exception as hata:
            print(f"  Başarısız: {hata}")
            continue
        if set(bulunan) != set(DOSYALAR):
            print(f"  Eksik dosya: {set(DOSYALAR) - set(bulunan)}")
            continue
        for d, b in bulunan.items():
            (HAM / d).write_bytes(b)
        KAYNAK_KAYDI.write_text(f"{ad}\t{url}\n")
        print(f"  Çalıştı: {ad}")
        return
    raise SystemExit("HATA: Veri hiçbir kaynaktan indirilemedi.")


def sha256(yol):
    """Dosyanın SHA-256 özetini döndürür."""
    return hashlib.sha256(yol.read_bytes()).hexdigest()


def sinif_dagilimi(df, sutun):
    """Bir etiket sütunu için pozitif/negatif sayısı ve oranını yazdırır."""
    poz = int(df[sutun].sum())
    neg = int(len(df) - poz)
    print(f"  {sutun}: {poz} pozitif / {neg} negatif (%{100 * poz / len(df):.2f})")
    return poz, neg


def main():
    HAM.mkdir(parents=True, exist_ok=True)
    ISLENMIS.mkdir(parents=True, exist_ok=True)
    veriyi_getir()

    print("\nSHA-256:")
    for yol in sorted(HAM.iterdir()):
        if yol.is_file() and yol.name != "kaynak.txt":
            print(f"  {yol.name}  {sha256(yol)}")

    # Okuma ve satır/motor sayısı kontrolü
    egitim = ham_oku(HAM / "train_FD001.txt")
    test = ham_oku(HAM / "test_FD001.txt")
    rul = pd.read_csv(HAM / "RUL_FD001.txt", sep=r"\s+", header=None).dropna(axis=1, how="all")
    assert len(egitim) == 20631 and egitim["unit"].nunique() == 100, (len(egitim), egitim["unit"].nunique())
    assert len(test) == 13096 and test["unit"].nunique() == 100, (len(test), test["unit"].nunique())
    assert rul.shape == (100, 1), rul.shape
    print(f"\nEğitim: {len(egitim)} satır / {egitim['unit'].nunique()} motor")
    print(f"Test:   {len(test)} satır / {test['unit'].nunique()} motor")
    print(f"RUL:    {len(rul)} satır")

    # Eğitim etiketi: RUL = motorun son döngüsü - döngü
    son = egitim.groupby("unit")["cycle"].transform("max")
    egitim = etiketle(egitim, son - egitim["cycle"])

    # Test: her motorun son döngüsü, gerçek RUL dosyadan (sıra = motor no)
    test_son = test.loc[test.groupby("unit")["cycle"].idxmax()].sort_values("unit").reset_index(drop=True)
    assert (test_son["unit"].values == range(1, 101)).all()
    test_son = etiketle(test_son, rul[0].values)

    print(f"\nEğitim sınıf dağılımı (ESIK={ESIK} ana etiket):")
    e30 = sinif_dagilimi(egitim, "etiket")
    sinif_dagilimi(egitim, "etiket_20")
    e50 = sinif_dagilimi(egitim, "etiket_50")
    print("Test (son döngü) sınıf dağılımı:")
    t30 = sinif_dagilimi(test_son, "etiket")
    sinif_dagilimi(test_son, "etiket_20")
    t50 = sinif_dagilimi(test_son, "etiket_50")

    assert e30 == (3100, 17531), e30
    assert e50 == (5100, 15531), e50
    assert t30 == (25, 75), t30
    assert t50 == (33, 67), t50
    print("\nTüm beklenen sayı assert'leri geçti.")

    egitim.to_csv(ISLENMIS / "train_FD001_etiketli.csv", index=False)
    test_son.to_csv(ISLENMIS / "test_FD001_son_dongu.csv", index=False)
    print("Yazıldı:", ISLENMIS / "train_FD001_etiketli.csv", ",", ISLENMIS / "test_FD001_son_dongu.csv")


if __name__ == "__main__":
    main()
