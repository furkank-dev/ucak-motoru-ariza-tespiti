"""Aşama 3: baz modeller (Lojistik Regresyon, Random Forest) ve çoğunluk sınıfı baz çizgisi."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ortak import (SEKILLER, TABLOLAR, RASTGELE_TOHUM, SENSORLER, ESIK,
                   egitim_yukle, test_yukle, metrikler)

HEDEF = "etiket"
YASAK = {"unit", "cycle", "RUL", "etiket", "etiket_20", "etiket_50"}
METRIKLER = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
KAT_SAYISI = 5
MAVI, TURUNCU, GRI, METIN = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b"


def modeller():
    """Değerlendirilecek pipeline'ları döndürür (ölçekleme her zaman pipeline içinde)."""
    return {
        "Çoğunluk sınıfı (hep 0)": Pipeline([
            ("olcek", StandardScaler()),
            ("model", DummyClassifier(strategy="most_frequent"))]),
        "Lojistik Regresyon": Pipeline([
            ("olcek", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000))]),
        "Random Forest": Pipeline([
            ("olcek", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                             random_state=RASTGELE_TOHUM, n_jobs=-1))]),
    }


def tahmin_et(pipe, X):
    """Eşik 0,5 ile sınıf tahmini ve pozitif sınıf olasılığını döndürür."""
    olasilik = pipe.predict_proba(X)[:, 1]
    return (olasilik >= 0.5).astype(int), olasilik


def capraz_dogrulama(ad, pipe, X, y, gruplar):
    """GroupKFold(5) ile kat bazlı metrikleri ve katlar dışı (OOF) tahminleri döndürür."""
    gkf = GroupKFold(n_splits=KAT_SAYISI)
    kat_satirlari = []
    oof_tahmin = np.zeros(len(y), dtype=int)
    oof_olasilik = np.zeros(len(y))
    for kat, (egt, dog) in enumerate(gkf.split(X, y, groups=gruplar), start=1):
        egt_motor, dog_motor = set(gruplar[egt]), set(gruplar[dog])
        assert egt_motor.isdisjoint(dog_motor), f"Kat {kat}: motor kesişimi var"
        model = clone(pipe).fit(X.iloc[egt], y[egt])
        tahmin, olasilik = tahmin_et(model, X.iloc[dog])
        oof_tahmin[dog], oof_olasilik[dog] = tahmin, olasilik
        satir = {"model": ad, "kat": kat, "egitim_motor": len(egt_motor), "dogrulama_motor": len(dog_motor),
                 "dogrulama_satir": len(dog), "dogrulama_pozitif": int(y[dog].sum())}
        satir.update(metrikler(y[dog], tahmin, olasilik))
        kat_satirlari.append(satir)
    return kat_satirlari, oof_tahmin, oof_olasilik


def karisiklik_ciz(matrisler):
    """Model x değerlendirme karışıklık matrislerini tek şekilde çizer."""
    fig, eksenler = plt.subplots(2, 2, figsize=(10, 8.5))
    for ax, (baslik, cm) in zip(eksenler.flat, matrisler):
        ax.imshow(cm, cmap="Blues", vmin=0)
        oran = cm / cm.sum()
        for i in range(2):
            for j in range(2):
                renk = "white" if cm[i, j] > cm.max() / 2 else METIN
                ax.text(j, i, f"{cm[i, j]}\n(%{100 * oran[i, j]:.1f})", ha="center", va="center", color=renk)
        ax.set_xticks([0, 1], ["0: Sağlıklı", "1: Arızaya yakın"])
        ax.set_yticks([0, 1], ["0: Sağlıklı", "1: Arızaya yakın"])
        ax.set_xlabel("Tahmin edilen sınıf")
        ax.set_ylabel("Gerçek sınıf")
        ax.set_title(baslik, fontsize=10, fontweight="bold")
    fig.suptitle(f"Karışıklık matrisleri (ESIK = {ESIK}, karar eşiği 0,5)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(SEKILLER / "06_karisiklik_matrisleri.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def roc_ciz(egriler):
    """CV (OOF) ve resmî test ROC eğrilerini tek şekilde iki panel olarak çizer."""
    fig, eksenler = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    renkler = {"Lojistik Regresyon": MAVI, "Random Forest": TURUNCU}
    for ax, (panel, panel_egrileri) in zip(eksenler, egriler.items()):
        for ad, (y, olasilik, auc) in panel_egrileri.items():
            fpr, tpr, _ = roc_curve(y, olasilik)
            ax.plot(fpr, tpr, color=renkler[ad], linewidth=2, label=f"{ad} (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], color=GRI, linestyle="--", linewidth=1.2, label="Rastgele / hep 0 (AUC = 0,5)")
        ax.set_title(panel, fontsize=11, fontweight="bold")
        ax.set_xlabel("Yanlış pozitif oranı (1 - özgüllük)")
        ax.grid(color="#e4e3df", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="lower right", fontsize=9)
    eksenler[0].set_ylabel("Doğru pozitif oranı (Recall)")
    fig.suptitle("ROC eğrileri", fontweight="bold")
    fig.tight_layout()
    fig.savefig(SEKILLER / "07_roc_egrileri.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    SEKILLER.mkdir(parents=True, exist_ok=True)
    TABLOLAR.mkdir(parents=True, exist_ok=True)
    egitim, test = egitim_yukle(), test_yukle()

    # Özellikler: yalnızca 14 sensörün ham değeri
    ozellikler = list(SENSORLER)
    assert not YASAK & set(ozellikler), "Yasak sütun özelliklerde"
    print("Özellikler:", ozellikler)
    X, y, gruplar = egitim[ozellikler], egitim[HEDEF].values, egitim["unit"].values
    X_test, y_test = test[ozellikler], test[HEDEF].values
    print(f"Eğitim: {len(X)} satır, {len(set(gruplar))} motor, {y.sum()} pozitif")
    print(f"Test (son döngü): {len(X_test)} satır, {y_test.sum()} pozitif")

    tum_katlar, ozet, matrisler = [], [], []
    egriler = {"Çapraz doğrulama (5 kat, katlar dışı tahminler)": {},
               "Resmî test (100 motorun son döngüsü)": {}}
    for ad, pipe in modeller().items():
        print(f"\n== {ad} ==")
        kat_satirlari, oof_tahmin, oof_olasilik = capraz_dogrulama(ad, pipe, X, y, gruplar)
        tum_katlar += kat_satirlari
        katlar = pd.DataFrame(kat_satirlari)
        satir = {"model": ad, "degerlendirme": "CV 5 kat (ortalama ± std)", "n": len(y), "pozitif": int(y.sum())}
        for m in METRIKLER:
            satir[m], satir[m + "_std"] = katlar[m].mean(), katlar[m].std(ddof=1)
        ozet.append(satir)
        print(katlar[["kat", "dogrulama_motor", "dogrulama_satir"] + METRIKLER].round(4).to_markdown(index=False))

        # Resmî test: tüm eğitim motorlarıyla eğit, test son döngüde değerlendir
        model = clone(pipe).fit(X, y)
        tahmin, olasilik = tahmin_et(model, X_test)
        test_metrik = metrikler(y_test, tahmin, olasilik)
        satir = {"model": ad, "degerlendirme": "Resmî test (son döngü)", "n": len(y_test), "pozitif": int(y_test.sum())}
        for m in METRIKLER:
            satir[m], satir[m + "_std"] = test_metrik[m], np.nan
        ozet.append(satir)
        print("Test:", {k: round(v, 4) for k, v in test_metrik.items()})

        if ad != "Çoğunluk sınıfı (hep 0)":
            matrisler.append((f"{ad}: CV katlar dışı ({len(y)} satır)", confusion_matrix(y, oof_tahmin, labels=[0, 1])))
            matrisler.append((f"{ad}: resmî test ({len(y_test)} motor)", confusion_matrix(y_test, tahmin, labels=[0, 1])))
            egriler["Çapraz doğrulama (5 kat, katlar dışı tahminler)"][ad] = (y, oof_olasilik, roc_auc_score(y, oof_olasilik))
            egriler["Resmî test (100 motorun son döngüsü)"][ad] = (y_test, olasilik, test_metrik["ROC-AUC"])

    # Tablolar
    sonuc = pd.DataFrame(ozet)
    sonuc.to_csv(TABLOLAR / "sonuclar_baz.csv", index=False)
    pd.DataFrame(tum_katlar).to_csv(TABLOLAR / "sonuclar_baz_katlar.csv", index=False)
    cm_satirlari = [{"baslik": b, "TN": cm[0, 0], "FP": cm[0, 1], "FN": cm[1, 0], "TP": cm[1, 1]} for b, cm in matrisler]
    pd.DataFrame(cm_satirlari).to_csv(TABLOLAR / "karisiklik_matrisleri.csv", index=False)

    karisiklik_ciz(matrisler)
    roc_ciz(egriler)

    print("\n== Özet (sonuclar_baz.csv) ==")
    goster = sonuc[["model", "degerlendirme"]].copy()
    for m in METRIKLER:
        goster[m] = [f"{a:.4f}" if np.isnan(s) else f"{a:.4f} ± {s:.4f}" for a, s in zip(sonuc[m], sonuc[m + "_std"])]
    print(goster.to_markdown(index=False))
    print("\nKarışıklık matrisleri:")
    print(pd.DataFrame(cm_satirlari).to_markdown(index=False))

    # Kontroller: metrik aralığı ve sızıntı şüphesi (özet ve kat düzeyinde)
    katlar = pd.DataFrame(tum_katlar)
    degerler = np.concatenate([sonuc[METRIKLER].to_numpy().ravel(), katlar[METRIKLER].to_numpy().ravel()])
    assert np.all((degerler >= 0) & (degerler <= 1)), "Metrik 0-1 dışında"
    supheli = pd.concat([
        sonuc.assign(kat="özet")[["model", "degerlendirme", "kat", "F1", "ROC-AUC"]],
        katlar.assign(degerlendirme="CV kat")[["model", "degerlendirme", "kat", "F1", "ROC-AUC"]],
    ])
    supheli = supheli[(supheli["F1"] > 0.99) | (supheli["ROC-AUC"] > 0.99)]
    if len(supheli):
        print("\nUYARI: F1 veya ROC-AUC > 0,99 (sızıntı şüphesi):")
        print(supheli.to_markdown(index=False, floatfmt=".4f"))
        sizinti_incelemesi(X, y, gruplar, egitim["RUL"].values, egriler)
    else:
        print("\nF1 / ROC-AUC 0,99'u aşan sonuç yok.")


def sizinti_incelemesi(X, y, gruplar, rul, egriler):
    """Yüksek ROC-AUC'nin nedenini inceler: eşiğe yakın bölgede AUC ve etiket karıştırma testi."""
    satirlar = []
    cv = egriler["Çapraz doğrulama (5 kat, katlar dışı tahminler)"]
    # Eşiğe yakın satırlarda (RUL 0-60) ayırt etmek daha zordur; AUC düşmeli
    for ad, (y_oof, olasilik, _) in cv.items():
        for alt, ust in [(0, 10_000), (0, 60), (15, 45)]:
            m = (rul >= alt) & (rul <= ust)
            satirlar.append({"inceleme": f"OOF AUC, RUL {alt}-{ust if ust < 10_000 else 'tümü'}",
                             "model": ad, "satir": int(m.sum()), "deger": roc_auc_score(y_oof[m], olasilik[m])})
    # Etiketler rastgele karıştırılır; boru hattında sızıntı yoksa AUC ~0,5 olmalı
    y_karisik = np.random.default_rng(RASTGELE_TOHUM).permutation(y)
    lr = modeller()["Lojistik Regresyon"]
    olasilik = np.zeros(len(y))
    for egt, dog in GroupKFold(n_splits=KAT_SAYISI).split(X, y_karisik, groups=gruplar):
        assert set(gruplar[egt]).isdisjoint(gruplar[dog])
        olasilik[dog] = clone(lr).fit(X.iloc[egt], y_karisik[egt]).predict_proba(X.iloc[dog])[:, 1]
    satirlar.append({"inceleme": "Karıştırılmış etiketle OOF AUC", "model": "Lojistik Regresyon",
                     "satir": len(y), "deger": roc_auc_score(y_karisik, olasilik)})
    tablo = pd.DataFrame(satirlar)
    tablo.to_csv(TABLOLAR / "sizinti_incelemesi.csv", index=False)
    print("\n== Sızıntı incelemesi (sizinti_incelemesi.csv) ==")
    print(tablo.to_markdown(index=False, floatfmt=".4f"))


if __name__ == "__main__":
    main()
