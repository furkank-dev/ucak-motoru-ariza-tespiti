"""Aşama 6: etiket eşiği duyarlılığı (RUL <= 20 / 30 / 50) — beş model + hep 0.

Her eşik için 03 ve 05 ile birebir aynı yöntem kullanılır:
  LR, RF, hep 0 : GroupKFold(5) CV + 100 motorla eğitip resmî test
  XGB, SVM, MLP : iç içe CV (her dış katta GroupKFold(3) GridSearchCV) + 100 motorda arama ve resmî test
Hiperparametre araması her eşik için yeniden yapılır (etiket değişince en iyi ayar da değişebilir).
ESIK = 30 satırları sonuclar_baz.csv / sonuclar_gelismis.csv ile aynı çıkmalı; script bunu kontrol eder.
"""
import importlib
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from ortak import SEKILLER, TABLOLAR, SENSORLER, DUYARLILIK_ESIKLERI, egitim_yukle, test_yukle, metrikler

baz = importlib.import_module("03_baz_modeller")
gel = importlib.import_module("05_gelismis_modeller")

METRIKLER = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
YASAK = {"unit", "cycle", "RUL", "etiket", "etiket_20", "etiket_50"}
DIS_KAT = 5


def hedef_sutun(esik):
    return "etiket" if esik == 30 else f"etiket_{esik}"


def tum_modeller():
    """Model adı -> (pipeline, ızgara ya da None, örnek ağırlığı)."""
    m = {ad: (pipe, None, False) for ad, pipe in baz.modeller().items()}
    m.update(gel.modeller())
    return m


def egit(pipe, izgara, agirlikli, X, y, gruplar):
    """Izgara yoksa doğrudan fit, varsa GridSearchCV (05'teki arama_yap). (model, en iyi ayar) döndürür."""
    if izgara is None:
        return clone(pipe).fit(X, y), ""
    arama = gel.arama_yap(pipe, izgara, agirlikli, X, y, gruplar)
    return arama.best_estimator_, gel.parametre_metni(arama.best_params_)


def degerlendir(esik, ad, pipe, izgara, agirlikli, X, y, gruplar, X_test, y_test):
    satirlar, ayarlar, katlar = [], [], []
    for kat, (egt, dog) in enumerate(GroupKFold(n_splits=DIS_KAT).split(X, y, groups=gruplar), start=1):
        assert set(gruplar[egt]).isdisjoint(gruplar[dog]), f"Kat {kat}: motor kesişimi var"
        model, ayar = egit(pipe, izgara, agirlikli, X.iloc[egt], y[egt], gruplar[egt])
        tahmin, skor = gel.skor_ve_tahmin(model, X.iloc[dog])
        katlar.append(metrikler(y[dog], tahmin, skor))
        if ayar:
            ayarlar.append({"esik": esik, "model": ad, "arama": f"dış kat {kat}", "en_iyi_parametre": ayar})
    katlar = pd.DataFrame(katlar)
    satir = {"esik": esik, "model": ad, "degerlendirme": "CV 5 kat (ortalama ± std)",
             "n": len(y), "pozitif": int(y.sum())}
    for m in METRIKLER:
        satir[m], satir[m + "_std"] = katlar[m].mean(), katlar[m].std(ddof=1)
    satirlar.append(satir)

    model, ayar = egit(pipe, izgara, agirlikli, X, y, gruplar)
    if ayar:
        ayarlar.append({"esik": esik, "model": ad, "arama": "tüm eğitim (100 motor)", "en_iyi_parametre": ayar})
    tahmin, skor = gel.skor_ve_tahmin(model, X_test)
    test_metrik = metrikler(y_test, tahmin, skor)
    satir = {"esik": esik, "model": ad, "degerlendirme": "Resmî test (son döngü)",
             "n": len(y_test), "pozitif": int(y_test.sum())}
    for m in METRIKLER:
        satir[m], satir[m + "_std"] = test_metrik[m], np.nan
    satirlar.append(satir)
    return satirlar, ayarlar


def ciz(tablo):
    """Satırlar: F1 ve Recall; sütunlar: CV ve test. x ekseni eşik."""
    tablo = tablo[tablo["model"] != "Çoğunluk sınıfı (hep 0)"]
    fig, eksenler = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for j, (on_ek, baslik) in enumerate([("CV", "Çapraz doğrulama (5 kat, ortalama ± std)"),
                                         ("Resmî test", "Resmî test (100 motorun son döngüsü)")]):
        alt = tablo[tablo["degerlendirme"].str.startswith(on_ek)]
        for i, metrik in enumerate(["F1", "Recall"]):
            ax = eksenler[i, j]
            for k, (ad, renk) in enumerate(gel.RENKLER.items()):
                d = alt[alt["model"] == ad].sort_values("esik")
                x = d["esik"].values + (k - 2) * 0.6  # çakışmasın diye küçük kaydırma
                hata = None if d[metrik + "_std"].isna().all() else d[metrik + "_std"].values
                ax.errorbar(x, d[metrik].values, yerr=hata, color=renk, marker="o", linewidth=1.8,
                            capsize=3, label=ad)
            ax.set_title(f"{metrik} — {baslik}", fontsize=10, fontweight="bold")
            ax.set_xticks(DUYARLILIK_ESIKLERI, [f"RUL ≤ {e}" for e in DUYARLILIK_ESIKLERI])
            ax.grid(color="#e4e3df", linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            if j == 0:
                ax.set_ylabel(metrik)
    eksenler[0, 1].legend(frameon=False, fontsize=9, loc="lower right")
    eksenler[1, 0].set_xlabel("Etiket eşiği (ESIK)")
    eksenler[1, 1].set_xlabel("Etiket eşiği (ESIK)")
    fig.suptitle("Etiket eşiğinin etkisi (karar eşiği 0,5; SVM'de 0)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(SEKILLER / "11_esik_duyarliligi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    egitim, test = egitim_yukle(), test_yukle()
    ozellikler = list(SENSORLER)
    assert not YASAK & set(ozellikler)
    X, gruplar, X_test = egitim[ozellikler], egitim["unit"].values, test[ozellikler]

    sonuclar, ayarlar = [], []
    for esik in DUYARLILIK_ESIKLERI:
        y, y_test = egitim[hedef_sutun(esik)].values, test[hedef_sutun(esik)].values
        print(f"\n##### ESIK = {esik}: eğitim {y.sum()} pozitif (%{100 * y.mean():.2f}), test {y_test.sum()} pozitif")
        for ad, (pipe, izgara, agirlikli) in tum_modeller().items():
            t0 = time.time()
            s, a = degerlendir(esik, ad, pipe, izgara, agirlikli, X, y, gruplar, X_test, y_test)
            sonuclar += s
            ayarlar += a
            print(f"{ad:25s} CV F1 {s[0]['F1']:.4f} ± {s[0]['F1_std']:.4f} | test F1 {s[1]['F1']:.4f} "
                  f"Recall {s[1]['Recall']:.4f} | {time.time() - t0:.0f} sn")

    tablo = pd.DataFrame(sonuclar)
    tablo.to_csv(TABLOLAR / "esik_duyarliligi.csv", index=False)
    pd.DataFrame(ayarlar).to_csv(TABLOLAR / "esik_duyarliligi_parametreler.csv", index=False)
    ciz(tablo)

    # Kontrol: ESIK = 30 satırları 03 ve 05'in çıktılarıyla aynı olmalı
    onceki = pd.concat([pd.read_csv(TABLOLAR / "sonuclar_baz.csv"), pd.read_csv(TABLOLAR / "sonuclar_gelismis.csv")])
    otuz = tablo[tablo["esik"] == 30].drop(columns="esik")
    birlesik = otuz.merge(onceki, on=["model", "degerlendirme"], suffixes=("", "_once"))
    assert len(birlesik) == len(otuz) == len(onceki), "ESIK=30 satırları eşleşmedi"
    for m in METRIKLER:
        assert np.allclose(birlesik[m], birlesik[m + "_once"]), f"ESIK=30 {m} 03/05 ile farklı"
    print("\nKontrol: ESIK = 30 sonuçları sonuclar_baz.csv ve sonuclar_gelismis.csv ile birebir aynı.")
    degerler = tablo[METRIKLER].to_numpy().ravel()
    assert np.all((degerler >= 0) & (degerler <= 1)), "Metrik 0-1 dışında"

    goster = tablo[["esik", "model", "degerlendirme", "pozitif"]].copy()
    for m in METRIKLER:
        goster[m] = [f"{a:.4f}" if np.isnan(s) else f"{a:.4f} ± {s:.4f}" for a, s in zip(tablo[m], tablo[m + "_std"])]
    print(goster.to_markdown(index=False))
    print("\n" + pd.DataFrame(ayarlar).to_markdown(index=False))


if __name__ == "__main__":
    main()
