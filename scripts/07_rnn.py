"""Aşama 7: basit tekrarlayan sinir ağı (RNN) ve LSTM, ESIK = 30.

Klasik modellerden tek farkı girdinin biçimi: her satır için o motorun son PENCERE döngüsünün
14 sensör değeri sıra halinde verilir (motorun ilk döngülerinde pencere ilk satır tekrarlanarak doldurulur).
Değerlendirme klasik modellerle birebir aynı: aynı GroupKFold(5) katları, aynı 20.631 satır,
resmî testte aynı 100 motorun son döngüsü (pencere test motorunun kendi geçmişinden alınır).

Eğitim: her katta eğitim motorlarının %10'u (motor bazlı) erken durdurma için ayrılır.
Resmî test için model 100 motorun tamamıyla, CV'de bulunan en iyi epoch sayılarının
ortalaması kadar eğitilir (doğrulama payı ayrılmaz).
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn

from ortak import (SEKILLER, TABLOLAR, ISLENMIS, RASTGELE_TOHUM, SENSORLER, ESIK,
                   egitim_yukle, test_yukle, metrikler)

HEDEF = "etiket"
METRIKLER = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
DIS_KAT = 5
PENCERE = 30          # kaç geçmiş döngü
GIZLI = 32            # gizli durum boyutu
DROPOUT = 0.2
OGRENME_ORANI = 1e-3
GRUP_BOYU = 256
MAKS_EPOCH = 40
SABIR = 5             # erken durdurma: doğrulama kaybı bu kadar epoch iyileşmezse dur
DOGRULAMA_PAYI = 0.1  # eğitim motorlarının erken durdurmaya ayrılan oranı
HUCRELER = {"RNN": nn.RNN, "LSTM": nn.LSTM}
RENKLER = {"RNN": "#d1495b", "LSTM": "#00798c"}

torch.set_num_threads(8)


class TekrarlayanAg(nn.Module):
    """Tek katmanlı RNN/LSTM -> son zaman adımının gizli durumu -> dropout -> tek çıktı (logit)."""

    def __init__(self, hucre, girdi=len(SENSORLER)):
        super().__init__()
        self.rnn = HUCRELER[hucre](girdi, GIZLI, batch_first=True)
        self.cikis = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(GIZLI, 1))

    def forward(self, x):
        cikti, _ = self.rnn(x)
        return self.cikis(cikti[:, -1, :]).squeeze(-1)


def pencereler(degerler, motorlar, pencere=PENCERE):
    """Her satır için (pencere, özellik) boyutlu geçmiş dizisi. Satırlar motor ve döngüye göre sıralı olmalı."""
    sonuc = np.empty((len(degerler), pencere, degerler.shape[1]), dtype=np.float32)
    for motor in np.unique(motorlar):
        idx = np.flatnonzero(motorlar == motor)
        blok = degerler[idx]
        dolgulu = np.vstack([np.repeat(blok[:1], pencere - 1, axis=0), blok])
        for k in range(len(idx)):
            sonuc[idx[k]] = dolgulu[k:k + pencere]
    return sonuc


def tohumla(tohum):
    torch.manual_seed(tohum)
    np.random.seed(tohum)


def olasilik(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(torch.cat([model(torch.from_numpy(X[i:i + 4096]))
                                        for i in range(0, len(X), 4096)])).numpy()


def egit(hucre, X, y, X_dog=None, y_dog=None, epoch_sayisi=MAKS_EPOCH):
    """Ağı eğitir. Doğrulama verisi verilirse erken durdurur ve en iyi epoch'un ağırlıklarını döndürür."""
    tohumla(RASTGELE_TOHUM)
    model = TekrarlayanAg(hucre)
    pozitif_agirlik = torch.tensor((y == 0).sum() / (y == 1).sum(), dtype=torch.float32)  # dengesizlik
    kayip_fn = nn.BCEWithLogitsLoss(pos_weight=pozitif_agirlik)
    opt = torch.optim.Adam(model.parameters(), lr=OGRENME_ORANI)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    uretec = torch.Generator().manual_seed(RASTGELE_TOHUM)
    gecmis, en_iyi, en_iyi_epoch, en_iyi_durum = [], np.inf, epoch_sayisi, None
    for epoch in range(1, epoch_sayisi + 1):
        model.train()
        toplam = 0.0
        for idx in torch.randperm(len(Xt), generator=uretec).split(GRUP_BOYU):
            opt.zero_grad()
            kayip = kayip_fn(model(Xt[idx]), yt[idx])
            kayip.backward()
            opt.step()
            toplam += kayip.item() * len(idx)
        satir = {"epoch": epoch, "egitim_kaybi": toplam / len(Xt)}
        if X_dog is not None:
            model.eval()
            with torch.no_grad():
                satir["dogrulama_kaybi"] = kayip_fn(model(torch.from_numpy(X_dog)),
                                                    torch.from_numpy(y_dog.astype(np.float32))).item()
            if satir["dogrulama_kaybi"] < en_iyi - 1e-4:
                en_iyi, en_iyi_epoch = satir["dogrulama_kaybi"], epoch
                en_iyi_durum = {k: v.clone() for k, v in model.state_dict().items()}
            elif epoch - en_iyi_epoch >= SABIR:
                gecmis.append(satir)
                break
        gecmis.append(satir)
    if en_iyi_durum is not None:
        model.load_state_dict(en_iyi_durum)
    return model, en_iyi_epoch, gecmis


def hazirla(egitim_df, uygula_df_listesi, pencere=PENCERE):
    """Ölçekleyiciyi yalnızca eğitim satırlarında fit eder, pencereleri üretir."""
    olcek = StandardScaler().fit(egitim_df[SENSORLER])
    return [pencereler(olcek.transform(d[SENSORLER]).astype(np.float32), d["unit"].values, pencere)
            for d in [egitim_df] + uygula_df_listesi]


def test_hazirla():
    """Test motorlarının tüm döngüleri (pencere için geçmiş gerekiyor) ve her motorun son döngü indeksi."""
    tum = pd.read_csv(ISLENMIS / "test_FD001_tum_donguler.csv").sort_values(["unit", "cycle"]).reset_index(drop=True)
    son = tum.groupby("unit").tail(1).index.values
    resmi = test_yukle().sort_values("unit").reset_index(drop=True)
    assert np.array_equal(tum.loc[son, "unit"].values, resmi["unit"].values)
    assert np.array_equal(tum.loc[son, HEDEF].values, resmi[HEDEF].values), "Test son döngü etiketleri tutmuyor"
    assert np.allclose(tum.loc[son, SENSORLER].values, resmi[SENSORLER].values)
    return tum, son, resmi[HEDEF].values


def capraz_dogrulama(hucre, egitim, y, gruplar, rng, pencere=PENCERE, yazdir=True):
    """GroupKFold(5) ile OOF olasılıkları, katların en iyi epoch'ları, kat metrikleri ve eğitim geçmişi."""
    oof = np.zeros(len(y))
    en_iyi_epochlar, kat_satirlari, gecmisler = [], [], []
    for kat, (egt, dog) in enumerate(GroupKFold(n_splits=DIS_KAT).split(egitim, y, groups=gruplar), start=1):
        assert set(gruplar[egt]).isdisjoint(gruplar[dog])
        egt_motor = np.unique(gruplar[egt])
        erken = rng.choice(egt_motor, size=int(round(DOGRULAMA_PAYI * len(egt_motor))), replace=False)
        m_erken = np.isin(gruplar, erken)
        asil = egt[~m_erken[egt]]
        ayrik = egt[m_erken[egt]]
        assert set(gruplar[asil]).isdisjoint(gruplar[ayrik]) and set(gruplar[ayrik]).isdisjoint(gruplar[dog])
        X_asil, X_ayrik, X_dog = hazirla(egitim.iloc[asil], [egitim.iloc[ayrik], egitim.iloc[dog]], pencere)
        model, en_iyi_epoch, gecmis = egit(hucre, X_asil, y[asil], X_ayrik, y[ayrik])
        en_iyi_epochlar.append(en_iyi_epoch)
        gecmisler += [{"model": hucre, "kat": kat, **g} for g in gecmis]
        oof[dog] = olasilik(model, X_dog)
        tahmin = (oof[dog] >= 0.5).astype(int)
        satir = {"model": hucre, "kat": kat, "egitim_motor": len(np.unique(gruplar[asil])),
                 "erken_durdurma_motor": len(erken), "dogrulama_motor": len(np.unique(gruplar[dog])),
                 "en_iyi_epoch": en_iyi_epoch}
        satir.update(metrikler(y[dog], tahmin, oof[dog]))
        kat_satirlari.append(satir)
        if yazdir:
            print(f"kat {kat}: epoch {en_iyi_epoch:2d} | F1 {satir['F1']:.4f} Recall {satir['Recall']:.4f} "
                  f"AUC {satir['ROC-AUC']:.4f}")
    return oof, en_iyi_epochlar, kat_satirlari, gecmisler


def kontroller(egitim, y, gruplar):
    """(1) Karıştırılmış etiketle LSTM: sızıntı yoksa OOF AUC ~0,5.
    (2) Pencere = 1 ile LSTM: geçmiş yokken klasik modellerle aynı bilgiyi görür; kazancın geçmişten mi
        mimariden mi geldiğini ayırır."""
    satirlar = []
    y_karisik = np.random.default_rng(RASTGELE_TOHUM).permutation(y)
    for ad, hedef, pencere in [("LSTM, karıştırılmış etiket", y_karisik, PENCERE),
                               ("LSTM, pencere = 1 (geçmiş yok)", y, 1),
                               ("RNN, pencere = 1 (geçmiş yok)", y, 1)]:
        hucre = ad.split(",")[0]
        oof, _, katlar, _ = capraz_dogrulama(hucre, egitim, hedef, gruplar, np.random.default_rng(RASTGELE_TOHUM),
                                             pencere, yazdir=False)
        katlar = pd.DataFrame(katlar)
        satirlar.append({"kontrol": ad, "pencere": pencere, "OOF_AUC": roc_auc_score(hedef, oof),
                         "F1_ortalama": katlar["F1"].mean(), "F1_std": katlar["F1"].std(ddof=1),
                         "Recall_ortalama": katlar["Recall"].mean()})
        print(f"{ad}: OOF AUC {satirlar[-1]['OOF_AUC']:.4f}, F1 {satirlar[-1]['F1_ortalama']:.4f}")
    tablo = pd.DataFrame(satirlar)
    tablo.to_csv(TABLOLAR / "rnn_kontroller.csv", index=False)
    return tablo


def main():
    egitim = egitim_yukle()
    assert egitim.equals(egitim.sort_values(["unit", "cycle"]).reset_index(drop=True)), "Eğitim sıralı değil"
    y, gruplar, rul = egitim[HEDEF].values, egitim["unit"].values, egitim["RUL"].values
    test_tum, test_son, y_test = test_hazirla()
    print(f"Eğitim {len(y)} satır ({y.sum()} pozitif), test {len(y_test)} motor ({y_test.sum()} pozitif), "
          f"pencere {PENCERE} döngü")

    ozet, kat_satirlari, gecmisler, matrisler, zor_bolge, egriler = [], [], [], [], [], {}
    rng = np.random.default_rng(RASTGELE_TOHUM)
    for hucre in HUCRELER:
        print(f"\n== {hucre} ==")
        t0 = time.time()
        oof, en_iyi_epochlar, k_satir, k_gecmis = capraz_dogrulama(hucre, egitim, y, gruplar, rng)
        kat_satirlari += k_satir
        gecmisler += k_gecmis
        katlar = pd.DataFrame([k for k in kat_satirlari if k["model"] == hucre])
        satir = {"model": hucre, "degerlendirme": "CV 5 kat (ortalama ± std)", "n": len(y), "pozitif": int(y.sum())}
        for m in METRIKLER:
            satir[m], satir[m + "_std"] = katlar[m].mean(), katlar[m].std(ddof=1)
        ozet.append(satir)

        # Resmî test: 100 motorun tamamı, sabit epoch sayısı
        son_epoch = int(round(np.mean(en_iyi_epochlar)))
        X_tum, X_test_tum = hazirla(egitim, [test_tum])
        model, _, gecmis = egit(hucre, X_tum, y, epoch_sayisi=son_epoch)
        gecmisler += [{"model": hucre, "kat": "tüm eğitim", **g} for g in gecmis]
        skor = olasilik(model, X_test_tum[test_son])
        tahmin = (skor >= 0.5).astype(int)
        test_metrik = metrikler(y_test, tahmin, skor)
        satir = {"model": hucre, "degerlendirme": "Resmî test (son döngü)", "n": len(y_test), "pozitif": int(y_test.sum())}
        for m in METRIKLER:
            satir[m], satir[m + "_std"] = test_metrik[m], np.nan
        ozet.append(satir)
        print(f"Test ({son_epoch} epoch): " + str({k: round(v, 4) for k, v in test_metrik.items()})
              + f" | süre {time.time() - t0:.0f} sn")

        oof_tahmin = (oof >= 0.5).astype(int)
        matrisler.append((f"{hucre}: CV katlar dışı", confusion_matrix(y, oof_tahmin, labels=[0, 1])))
        matrisler.append((f"{hucre}: resmî test", confusion_matrix(y_test, tahmin, labels=[0, 1])))
        egriler[hucre] = pd.DataFrame([g for g in gecmisler if g["model"] == hucre and g["kat"] != "tüm eğitim"])
        for alt, ust in [(0, 10_000), (0, 60), (15, 45)]:
            m = (rul >= alt) & (rul <= ust)
            zor_bolge.append({"model": hucre, "bant": f"RUL {alt}-{ust if ust < 10_000 else 'tümü'}",
                              "satir": int(m.sum()), "OOF_AUC": roc_auc_score(y[m], oof[m])})

    sonuc = pd.DataFrame(ozet)
    sonuc.to_csv(TABLOLAR / "sonuclar_rnn.csv", index=False)
    pd.DataFrame(kat_satirlari).to_csv(TABLOLAR / "sonuclar_rnn_katlar.csv", index=False)
    pd.DataFrame(gecmisler).to_csv(TABLOLAR / "rnn_egitim_gecmisi.csv", index=False)
    cm = [{"baslik": b, "TN": c[0, 0], "FP": c[0, 1], "FN": c[1, 0], "TP": c[1, 1]} for b, c in matrisler]
    pd.DataFrame(cm).to_csv(TABLOLAR / "karisiklik_matrisleri_rnn.csv", index=False)
    pd.DataFrame(zor_bolge).to_csv(TABLOLAR / "zor_bolge_auc_rnn.csv", index=False)

    # Klasik modellerle birleşik tablo
    klasik = pd.read_csv(TABLOLAR / "tum_modeller_karsilastirma.csv")
    tum = pd.concat([klasik, sonuc], ignore_index=True)
    tum.to_csv(TABLOLAR / "tum_modeller_rnn_dahil.csv", index=False)

    ogrenme_egrisi_ciz(egriler)
    karsilastirma_ciz(tum)

    goster = tum[["model", "degerlendirme"]].copy()
    for m in METRIKLER:
        goster[m] = [f"{a:.4f}" if np.isnan(s) else f"{a:.4f} ± {s:.4f}" for a, s in zip(tum[m], tum[m + "_std"])]
    print("\n" + goster.to_markdown(index=False))
    print("\n" + pd.DataFrame(kat_satirlari).round(4).to_markdown(index=False))
    print("\n" + pd.DataFrame(cm).to_markdown(index=False))
    print("\n" + pd.DataFrame(zor_bolge).to_markdown(index=False, floatfmt=".4f"))
    print("\n== Kontroller (rnn_kontroller.csv) ==")
    print(kontroller(egitim, y, gruplar).to_markdown(index=False, floatfmt=".4f"))
    degerler = np.concatenate([sonuc[METRIKLER].to_numpy().ravel(), pd.DataFrame(kat_satirlari)[METRIKLER].to_numpy().ravel()])
    assert np.all((degerler >= 0) & (degerler <= 1)), "Metrik 0-1 dışında"


def ogrenme_egrisi_ciz(egriler):
    fig, eksenler = plt.subplots(1, len(egriler), figsize=(12, 4.5), sharey=True)
    for ax, (hucre, g) in zip(eksenler, egriler.items()):
        for kat, d in g.groupby("kat"):
            ax.plot(d["epoch"], d["egitim_kaybi"], color=RENKLER[hucre], alpha=0.35, linewidth=1)
            ax.plot(d["epoch"], d["dogrulama_kaybi"], color=RENKLER[hucre], linewidth=1.8)
        ax.plot([], [], color=RENKLER[hucre], alpha=0.35, label="Eğitim kaybı (5 kat)")
        ax.plot([], [], color=RENKLER[hucre], linewidth=1.8, label="Erken durdurma motorlarında kayıp")
        ax.set_title(hucre, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.grid(color="#e4e3df", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    eksenler[0].set_ylabel("Ağırlıklı ikili çapraz entropi")
    fig.suptitle(f"Öğrenme eğrileri (pencere {PENCERE} döngü, erken durdurma sabrı {SABIR})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(SEKILLER / "12_rnn_ogrenme_egrileri.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def karsilastirma_ciz(tum):
    tum = tum[tum["model"] != "Çoğunluk sınıfı (hep 0)"]
    renkler = {"Lojistik Regresyon": "#2a78d6", "Random Forest": "#eb6834", "XGBoost": "#1f9e6e",
               "SVM (RBF)": "#8e4fc4", "MLP": "#c9a227", **RENKLER}
    adlar = list(renkler)
    gosterilen = ["Precision", "Recall", "F1", "ROC-AUC"]
    fig, eksenler = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, (on_ek, baslik) in zip(eksenler, [("CV", "Çapraz doğrulama (5 kat, ortalama ± std)"),
                                              ("Resmî test", "Resmî test (100 motorun son döngüsü)")]):
        alt = tum[tum["degerlendirme"].str.startswith(on_ek)].set_index("model").loc[adlar]
        x, gen = np.arange(len(gosterilen)), 0.12
        for i, ad in enumerate(adlar):
            hata = alt.loc[ad, [m + "_std" for m in gosterilen]].astype(float).values
            ax.bar(x + (i - 3) * gen, alt.loc[ad, gosterilen].astype(float).values, gen,
                   yerr=None if np.all(np.isnan(hata)) else hata, capsize=2, color=renkler[ad], label=ad,
                   hatch="//" if ad in RENKLER else None, edgecolor="white", linewidth=0.5,
                   error_kw={"elinewidth": 0.8, "ecolor": "#444"})
        ax.set_xticks(x, gosterilen)
        ax.set_ylim(0.5, 1.02)
        ax.set_title(baslik, fontsize=11, fontweight="bold")
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    eksenler[0].set_ylabel("Değer (y ekseni 0,5'ten başlıyor)")
    tut, etiket = eksenler[0].get_legend_handles_labels()
    fig.legend(tut, etiket, frameon=False, loc="lower center", ncol=7, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Klasik modeller (tek döngü) ve RNN / LSTM (son {PENCERE} döngü), ESIK = {ESIK}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(SEKILLER / "13_tum_modeller_rnn_dahil.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
