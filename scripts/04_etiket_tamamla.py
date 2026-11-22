"""Etiket tamamlama: test tüm döngüler etiketi ve FD002-FD004 yedek verileri."""
import io
import zipfile

import numpy as np
import pandas as pd

from ortak import HAM, ISLENMIS, ham_oku, etiketle

YEDEK = ISLENMIS / "yedek"
ETIKETLER = {20: "etiket_20", 30: "etiket", 50: "etiket_50"}
# Beklenen satır sayıları (tutmazsa sadece uyarı verir)
BEKLENEN_YEDEK = {"FD002": (53759, 33991), "FD003": (24720, 16596), "FD004": (61249, 41214)}


def zipten_oku(ad):
    """data/raw içindeki orijinal zip'ten (iç içe zip dahil) bir dosyanın baytlarını döndürür."""
    ziplar = sorted(HAM.glob("*.zip"))
    if not ziplar:
        raise SystemExit("DUR: data/raw içinde zip yok.")
    dis = zipfile.ZipFile(ziplar[0])
    ic_ad = next(n for n in dis.namelist() if n.endswith("CMAPSSData.zip"))
    ic = zipfile.ZipFile(io.BytesIO(dis.read(ic_ad)))
    return ic.read(ad)


def dagilim(df):
    """Her eşik için pozitif/negatif sayısını tek satırlık metin olarak döndürür."""
    parcalar = []
    for e, s in ETIKETLER.items():
        poz = int(df[s].sum())
        parcalar.append(f"ESIK={e}: {poz}/{len(df) - poz} (%{100 * poz / len(df):.2f})")
    return " | ".join(parcalar)


def tek_yonlu_mu(df, sutun):
    """Her motorda etiket 0'dan 1'e en fazla bir kez geçiyor ve geri dönmüyor mu."""
    for _, g in df.sort_values(["unit", "cycle"]).groupby("unit"):
        d = np.diff(g[sutun].values)
        if (d < 0).any() or (d > 0).sum() > 1:
            return False
    return True


def test_tum_donguler(test_ham, rul_dosya):
    """Test motorlarının bütün döngülerini RUL_FD00x + (son döngü - döngü) ile etiketler."""
    son = test_ham.groupby("unit")["cycle"].transform("max")
    rul = test_ham["unit"].map(dict(zip(range(1, len(rul_dosya) + 1), rul_dosya))) + (son - test_ham["cycle"])
    return etiketle(test_ham, rul)


def fd001_test_tum_donguler():
    """FD001 test tüm döngüler dosyasını üretir ve kontrol eder."""
    hedef = ISLENMIS / "test_FD001_tum_donguler.csv"
    if hedef.exists():
        print(f"{hedef.name} zaten var, dokunulmadı.")
        return
    test_ham = ham_oku(HAM / "test_FD001.txt")
    rul_dosya = pd.read_csv(HAM / "RUL_FD001.txt", sep=r"\s+", header=None)[0].values
    df = test_tum_donguler(test_ham, rul_dosya)

    assert len(df) == 13096, len(df)
    assert (df["RUL"] >= 0).all()
    son_satir = df.loc[df.groupby("unit")["cycle"].idxmax()].sort_values("unit")
    assert (son_satir["RUL"].values == rul_dosya).all(), "Son döngü RUL'u dosyayla tutmuyor"
    # Son döngü satırları mevcut test_FD001_son_dongu.csv ile birebir aynı olmalı
    mevcut = pd.read_csv(ISLENMIS / "test_FD001_son_dongu.csv")
    pd.testing.assert_frame_equal(son_satir.reset_index(drop=True), mevcut, check_dtype=False)
    for s in ETIKETLER.values():
        assert tek_yonlu_mu(df, s), s

    df.to_csv(hedef, index=False)
    print(f"Yazıldı: {hedef.name}  {len(df)} satır x {df.shape[1]} sütun, {df['unit'].nunique()} motor")
    print(f"  RUL min {df['RUL'].min()} maks {df['RUL'].max()}")
    print("  Dağılım:", dagilim(df))
    print("  Kontroller geçti: 13096 satır, RUL >= 0, son döngü RUL = RUL_FD001.txt, "
          "son döngü satırları test_FD001_son_dongu.csv ile aynı, etiketler tek yönlü.")


def yedek_veriler():
    """FD002-FD004'ü zip'ten okuyup aynı etiket fonksiyonuyla yedek klasörüne yazar."""
    YEDEK.mkdir(parents=True, exist_ok=True)
    print(f"\nYedek veriler -> {YEDEK}")
    for fd, (bek_tr, bek_te) in BEKLENEN_YEDEK.items():
        tr_yol, te_yol = YEDEK / f"train_{fd}_etiketli.csv", YEDEK / f"test_{fd}_son_dongu.csv"
        if tr_yol.exists() and te_yol.exists():
            print(f"{fd}: dosyalar zaten var, dokunulmadı.")
            continue
        tr = ham_oku(io.BytesIO(zipten_oku(f"train_{fd}.txt")))
        te = ham_oku(io.BytesIO(zipten_oku(f"test_{fd}.txt")))
        rul_dosya = pd.read_csv(io.BytesIO(zipten_oku(f"RUL_{fd}.txt")), sep=r"\s+", header=None)[0].values

        # Eğitim: RUL = son döngü - döngü (FD001 ile aynı kural)
        son = tr.groupby("unit")["cycle"].transform("max")
        tr = etiketle(tr, son - tr["cycle"])
        # Test: son döngü, RUL dosyadan
        te_son = te.loc[te.groupby("unit")["cycle"].idxmax()].sort_values("unit").reset_index(drop=True)
        assert len(rul_dosya) == te_son["unit"].nunique() == te_son["unit"].max()
        te_son = etiketle(te_son, rul_dosya)

        tr.to_csv(tr_yol, index=False)
        te_son.to_csv(te_yol, index=False)
        ok = lambda g, b: "tutuyor" if g == b else f"TUTMUYOR (beklenen {b})"
        print(f"\n{fd}:")
        print(f"  train: {len(tr)} satır, {tr['unit'].nunique()} motor -> beklenenle {ok(len(tr), bek_tr)}")
        print(f"  test (ham): {len(te)} satır, {te['unit'].nunique()} motor -> beklenenle {ok(len(te), bek_te)}")
        print(f"  RUL_{fd}.txt: {len(rul_dosya)} satır | NaN train {int(tr.isna().sum().sum())}, test {int(te_son.isna().sum().sum())}")
        print(f"  train dağılım: {dagilim(tr)}")
        print(f"  test son döngü dağılım: {dagilim(te_son)}")
        omur = tr.groupby("unit")["cycle"].max()
        print(f"  motor ömrü min {omur.min()} maks {omur.max()} | train etiketleri tek yönlü: "
              f"{all(tek_yonlu_mu(tr, s) for s in ETIKETLER.values())}")


def main():
    # Ana FD001 sayıları hâlâ tutuyor mu (değişmişse dur)
    tr = pd.read_csv(ISLENMIS / "train_FD001_etiketli.csv")
    te = pd.read_csv(ISLENMIS / "test_FD001_son_dongu.csv")
    assert len(tr) == 20631 and tr["etiket"].sum() == 3100, "DUR: FD001 train sayıları tutmuyor"
    assert len(te) == 100 and te["etiket"].sum() == 25, "DUR: FD001 test sayıları tutmuyor"
    assert len(ham_oku(HAM / "test_FD001.txt")) == 13096, "DUR: FD001 test ham satır sayısı tutmuyor"
    print("FD001 ana sayıları tutuyor: train 20631 / 3100 pozitif, test 13096 satır / son döngü 25 pozitif.\n")

    fd001_test_tum_donguler()
    yedek_veriler()
    print("\nBitti.")


if __name__ == "__main__":
    main()
