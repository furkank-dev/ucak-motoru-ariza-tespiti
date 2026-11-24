"""Aşama 5: gelişmiş klasik modeller (XGBoost, SVM, MLP) + küçük GridSearchCV.

Değerlendirme iç içe (nested) motor bazlı CV ile yapılır:
  dış döngü: GroupKFold(5), 03 ile aynı katlar -> raporlanan CV metrikleri
  iç döngü : her dış katın 80 eğitim motorunda GroupKFold(3) ile GridSearchCV (skor: F1)
Resmî test için GridSearchCV 100 eğitim motorunun tamamında tekrar çalıştırılır,
en iyi ayar tüm eğitimle fit edilip test son döngüde değerlendirilir.
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ortak import (SEKILLER, TABLOLAR, RASTGELE_TOHUM, SENSORLER, ESIK,
                   egitim_yukle, test_yukle, metrikler)

HEDEF = "etiket"
YASAK = {"unit", "cycle", "RUL", "etiket", "etiket_20", "etiket_50"}
METRIKLER = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
DIS_KAT, IC_KAT = 5, 3
RENKLER = {"Lojistik Regresyon": "#2a78d6", "Random Forest": "#eb6834",
           "XGBoost": "#1f9e6e", "SVM (RBF)": "#8e4fc4", "MLP": "#c9a227"}
GRI, METIN = "#8a8984", "#0b0b0b"


def modeller():
    """Model adı -> (pipeline, parametre ızgarası, örnek ağırlığı kullanılsın mı).

    Dengesizlik: SVM'de class_weight="balanced"; XGBoost ve MLP'de aynı etkiyi
    veren "balanced" örnek ağırlıkları (MLP class_weight desteklemiyor).
    """
    return {
        "XGBoost": (
            Pipeline([("olcek", StandardScaler()),
                      ("model", XGBClassifier(subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                              tree_method="hist", random_state=RASTGELE_TOHUM, n_jobs=1))]),
            {"model__max_depth": [3, 5],
             "model__n_estimators": [200, 400],
             "model__learning_rate": [0.05, 0.1]},
            True),
        "SVM (RBF)": (
            Pipeline([("olcek", StandardScaler()),
                      ("model", SVC(kernel="rbf", class_weight="balanced"))]),
            {"model__C": [0.1, 1, 10],
             "model__gamma": ["scale", 0.01, 0.1]},
            False),
        "MLP": (
            Pipeline([("olcek", StandardScaler()),
                      ("model", MLPClassifier(max_iter=500, early_stopping=True, n_iter_no_change=20,
                                              random_state=RASTGELE_TOHUM))]),
            {"model__hidden_layer_sizes": [(32,), (64, 32)],
             "model__alpha": [1e-4, 1e-2],
             "model__learning_rate_init": [1e-3, 1e-2]},
            True),
    }


def skor_ve_tahmin(model, X):
    """Sınıf tahmini ve sürekli skor. predict_proba varsa eşik 0,5;
    SVM'de decision_function (eşik 0, predict() ile aynı). ROC-AUC için ikisi de geçerli."""
    if hasattr(model, "predict_proba"):
        skor = model.predict_proba(X)[:, 1]
        return (skor >= 0.5).astype(int), skor
    skor = model.decision_function(X)
    return (skor >= 0).astype(int), skor


def arama_yap(pipe, izgara, agirlikli, X, y, gruplar):
    """Verilen motorlarda GroupKFold(3) ile GridSearchCV yapar, en iyi ayarla tüm veride fit eder."""
    arama = GridSearchCV(clone(pipe), izgara, scoring="f1", cv=GroupKFold(n_splits=IC_KAT),
                         n_jobs=-1, refit=True)
    ek = {"model__sample_weight": compute_sample_weight("balanced", y)} if agirlikli else {}
    arama.fit(X, y, groups=gruplar, **ek)
    return arama


def parametre_metni(p):
    return json.dumps({k.replace("model__", ""): (list(v) if isinstance(v, tuple) else v) for k, v in p.items()})


def karisiklik_ciz(matrisler):
    fig, eksenler = plt.subplots(3, 2, figsize=(10, 12.5))
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
    fig.suptitle(f"Karışıklık matrisleri: XGBoost, SVM, MLP (ESIK = {ESIK})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(SEKILLER / "08_karisiklik_matrisleri_gelismis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def roc_ciz(egriler):
    fig, eksenler = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (panel, panel_egrileri) in zip(eksenler, egriler.items()):
        for ad, (y, skor, auc) in panel_egrileri.items():
            fpr, tpr, _ = roc_curve(y, skor)
            ax.plot(fpr, tpr, color=RENKLER[ad], linewidth=2, label=f"{ad} (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], color=GRI, linestyle="--", linewidth=1.2, label="Rastgele (AUC = 0,5)")
        ax.set_title(panel, fontsize=11, fontweight="bold")
        ax.set_xlabel("Yanlış pozitif oranı (1 - özgüllük)")
        ax.grid(color="#e4e3df", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="lower right", fontsize=9)
    eksenler[0].set_ylabel("Doğru pozitif oranı (Recall)")
    fig.suptitle("ROC eğrileri: XGBoost, SVM, MLP", fontweight="bold")
    fig.tight_layout()
    fig.savefig(SEKILLER / "09_roc_egrileri_gelismis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def karsilastirma_ciz(tablo):
    """Beş modelin CV (± std) ve test F1 / Recall / Precision değerleri, gruplanmış çubuk grafik."""
    tablo = tablo[tablo["model"] != "Çoğunluk sınıfı (hep 0)"]
    adlar = list(RENKLER)
    gosterilen = ["Precision", "Recall", "F1", "ROC-AUC"]
    fig, eksenler = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (degerlendirme, baslik) in zip(eksenler, [("CV", "Çapraz doğrulama (5 kat, ortalama ± std)"),
                                                      ("Resmî test", "Resmî test (100 motorun son döngüsü)")]):
        alt = tablo[tablo["degerlendirme"].str.startswith(degerlendirme)].set_index("model").loc[adlar]
        x = np.arange(len(gosterilen))
        genislik = 0.16
        for i, ad in enumerate(adlar):
            degerler = alt.loc[ad, gosterilen].astype(float).values
            hatalar = alt.loc[ad, [m + "_std" for m in gosterilen]].astype(float).values
            hatalar = None if np.all(np.isnan(hatalar)) else hatalar
            ax.bar(x + (i - 2) * genislik, degerler, genislik, yerr=hatalar, capsize=2,
                   color=RENKLER[ad], label=ad, error_kw={"elinewidth": 0.8, "ecolor": "#444"})
        ax.set_xticks(x, gosterilen)
        ax.set_ylim(0.5, 1.02)
        ax.set_title(baslik, fontsize=11, fontweight="bold")
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    eksenler[0].set_ylabel("Değer (y ekseni 0,5'ten başlıyor)")
    tut, etiket = eksenler[0].get_legend_handles_labels()
    fig.legend(tut, etiket, frameon=False, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Beş modelin karşılaştırması (ESIK = {ESIK})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(SEKILLER / "10_model_karsilastirma.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    egitim, test = egitim_yukle(), test_yukle()
    ozellikler = list(SENSORLER)
    assert not YASAK & set(ozellikler), "Yasak sütun özelliklerde"
    X, y, gruplar = egitim[ozellikler], egitim[HEDEF].values, egitim["unit"].values
    X_test, y_test = test[ozellikler], test[HEDEF].values
    rul = egitim["RUL"].values
    print(f"Eğitim: {len(X)} satır, {len(set(gruplar))} motor, {y.sum()} pozitif | Test: {len(X_test)} satır, {y_test.sum()} pozitif")

    tum_katlar, ozet, matrisler, parametreler, zor_bolge = [], [], [], [], []
    egriler = {"Çapraz doğrulama (5 kat, katlar dışı tahminler)": {},
               "Resmî test (100 motorun son döngüsü)": {}}
    for ad, (pipe, izgara, agirlikli) in modeller().items():
        print(f"\n== {ad} ({np.prod([len(v) for v in izgara.values()])} aday ayar) ==")
        t0 = time.time()
        oof_tahmin, oof_skor = np.zeros(len(y), dtype=int), np.zeros(len(y))
        kat_satirlari = []
        for kat, (egt, dog) in enumerate(GroupKFold(n_splits=DIS_KAT).split(X, y, groups=gruplar), start=1):
            assert set(gruplar[egt]).isdisjoint(gruplar[dog]), f"Kat {kat}: motor kesişimi var"
            arama = arama_yap(pipe, izgara, agirlikli, X.iloc[egt], y[egt], gruplar[egt])
            tahmin, skor = skor_ve_tahmin(arama.best_estimator_, X.iloc[dog])
            oof_tahmin[dog], oof_skor[dog] = tahmin, skor
            satir = {"model": ad, "kat": kat, "egitim_motor": len(set(gruplar[egt])),
                     "dogrulama_motor": len(set(gruplar[dog])), "dogrulama_satir": len(dog),
                     "dogrulama_pozitif": int(y[dog].sum())}
            satir.update(metrikler(y[dog], tahmin, skor))
            kat_satirlari.append(satir)
            parametreler.append({"model": ad, "arama": f"dış kat {kat} (80 motor)",
                                 "en_iyi_parametre": parametre_metni(arama.best_params_),
                                 "ic_cv_F1": arama.best_score_})
        tum_katlar += kat_satirlari
        katlar = pd.DataFrame(kat_satirlari)
        satir = {"model": ad, "degerlendirme": "CV 5 kat (ortalama ± std)", "n": len(y), "pozitif": int(y.sum())}
        for m in METRIKLER:
            satir[m], satir[m + "_std"] = katlar[m].mean(), katlar[m].std(ddof=1)
        ozet.append(satir)
        print(katlar[["kat"] + METRIKLER].round(4).to_markdown(index=False))

        # Resmî test: aramayı 100 motorda tekrarla, en iyi ayarla test son döngüyü değerlendir
        arama = arama_yap(pipe, izgara, agirlikli, X, y, gruplar)
        parametreler.append({"model": ad, "arama": "tüm eğitim (100 motor) -> resmî test",
                             "en_iyi_parametre": parametre_metni(arama.best_params_), "ic_cv_F1": arama.best_score_})
        tahmin, skor = skor_ve_tahmin(arama.best_estimator_, X_test)
        test_metrik = metrikler(y_test, tahmin, skor)
        satir = {"model": ad, "degerlendirme": "Resmî test (son döngü)", "n": len(y_test), "pozitif": int(y_test.sum())}
        for m in METRIKLER:
            satir[m], satir[m + "_std"] = test_metrik[m], np.nan
        ozet.append(satir)
        print("En iyi ayar (100 motor):", parametre_metni(arama.best_params_))
        print("Test:", {k: round(v, 4) for k, v in test_metrik.items()}, f"| süre {time.time() - t0:.0f} sn")

        matrisler.append((f"{ad}: CV katlar dışı ({len(y)} satır)", confusion_matrix(y, oof_tahmin, labels=[0, 1])))
        matrisler.append((f"{ad}: resmî test ({len(y_test)} motor)", confusion_matrix(y_test, tahmin, labels=[0, 1])))
        oof_auc = roc_auc_score(y, oof_skor)
        egriler["Çapraz doğrulama (5 kat, katlar dışı tahminler)"][ad] = (y, oof_skor, oof_auc)
        egriler["Resmî test (100 motorun son döngüsü)"][ad] = (y_test, skor, test_metrik["ROC-AUC"])
        # 03'teki sızıntı incelemesiyle aynı bantlar: eşiğe yakın zor bölgede AUC
        for alt, ust in [(0, 10_000), (0, 60), (15, 45)]:
            m = (rul >= alt) & (rul <= ust)
            zor_bolge.append({"model": ad, "bant": f"RUL {alt}-{ust if ust < 10_000 else 'tümü'}",
                              "satir": int(m.sum()), "OOF_AUC": roc_auc_score(y[m], oof_skor[m])})

    # Tablolar
    sonuc = pd.DataFrame(ozet)
    sonuc.to_csv(TABLOLAR / "sonuclar_gelismis.csv", index=False)
    katlar = pd.DataFrame(tum_katlar)
    katlar.to_csv(TABLOLAR / "sonuclar_gelismis_katlar.csv", index=False)
    pd.DataFrame(parametreler).to_csv(TABLOLAR / "en_iyi_parametreler.csv", index=False)
    cm_satirlari = [{"baslik": b, "TN": cm[0, 0], "FP": cm[0, 1], "FN": cm[1, 0], "TP": cm[1, 1]} for b, cm in matrisler]
    pd.DataFrame(cm_satirlari).to_csv(TABLOLAR / "karisiklik_matrisleri_gelismis.csv", index=False)
    pd.DataFrame(zor_bolge).to_csv(TABLOLAR / "zor_bolge_auc_gelismis.csv", index=False)

    # Beş modelin karşılaştırması (baz sonuçlar 03'ün çıktısından okunur)
    baz = pd.read_csv(TABLOLAR / "sonuclar_baz.csv")
    tum = pd.concat([baz, sonuc], ignore_index=True)
    tum.to_csv(TABLOLAR / "tum_modeller_karsilastirma.csv", index=False)
    baz_katlar = pd.read_csv(TABLOLAR / "sonuclar_baz_katlar.csv")
    tum_kat = pd.concat([baz_katlar, katlar], ignore_index=True)
    kat_f1 = tum_kat[tum_kat["model"] != "Çoğunluk sınıfı (hep 0)"].pivot(index="kat", columns="model", values="F1")
    kat_f1.to_csv(TABLOLAR / "kat_bazli_f1_tum_modeller.csv")

    karisiklik_ciz(matrisler)
    roc_ciz(egriler)
    karsilastirma_ciz(tum)

    print("\n== Beş model (tum_modeller_karsilastirma.csv) ==")
    goster = tum[["model", "degerlendirme"]].copy()
    for m in METRIKLER:
        goster[m] = [f"{a:.4f}" if np.isnan(s) else f"{a:.4f} ± {s:.4f}" for a, s in zip(tum[m], tum[m + "_std"])]
    print(goster.to_markdown(index=False))
    print("\nKat bazlı F1:\n" + kat_f1.round(4).to_markdown())
    print("\nEn iyi parametreler:\n" + pd.DataFrame(parametreler).to_markdown(index=False, floatfmt=".4f"))
    print("\nKarışıklık matrisleri:\n" + pd.DataFrame(cm_satirlari).to_markdown(index=False))
    print("\nZor bölge AUC:\n" + pd.DataFrame(zor_bolge).to_markdown(index=False, floatfmt=".4f"))

    # Kontroller
    degerler = np.concatenate([sonuc[METRIKLER].to_numpy().ravel(), katlar[METRIKLER].to_numpy().ravel()])
    assert np.all((degerler >= 0) & (degerler <= 1)), "Metrik 0-1 dışında"
    supheli = katlar[(katlar["F1"] > 0.99) | (katlar["ROC-AUC"] > 0.995)]
    if len(supheli):
        print("\nUYARI: F1 > 0,99 veya ROC-AUC > 0,995 olan kat var:\n" + supheli.to_markdown(index=False))


if __name__ == "__main__":
    main()
