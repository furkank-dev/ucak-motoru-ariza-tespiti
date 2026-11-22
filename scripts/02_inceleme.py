"""Aşama 2: keşifsel veri incelemesi (grafikler ve tablolar)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from ortak import (SEKILLER, TABLOLAR, ESIK, DUYARLILIK_ESIKLERI, SENSORLER,
                   BEKLENEN_SABITLER, egitim_yukle)

# Renkler (doğrulanmış kategorik palet; 0 = sağlıklı mavi, 1 = arızaya yakın turuncu)
MAVI, TURUNCU = "#2a78d6", "#eb6834"
MOTOR_RENKLERI = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
METIN, IKINCIL, IZGARA = "#0b0b0b", "#52514e", "#e4e3df"
DIVERJAN = LinearSegmentedColormap.from_list(
    "mavi_gri_kirmizi", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a8302f"])

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": IKINCIL, "axes.labelcolor": METIN, "text.color": METIN,
    "xtick.color": IKINCIL, "ytick.color": IKINCIL,
    "axes.grid": True, "grid.color": IZGARA, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
})


def kaydet(fig, ad):
    """Şekli outputs/figures altına png olarak kaydeder."""
    yol = SEKILLER / ad
    fig.savefig(yol)
    plt.close(fig)
    print("  Şekil:", yol.name)


def tablo_kaydet(df, ad, index=False):
    """Tabloyu csv olarak kaydeder ve terminale yazdırır."""
    df.to_csv(TABLOLAR / ad, index=index)
    print(f"\n[{ad}]")
    print(df.to_markdown(index=index, floatfmt=".4f"))


def sinif_dagilimi(df):
    """ESIK=30 sınıf dağılımı grafiği ve 20/30/50 oran tablosu."""
    satirlar = []
    for e in DUYARLILIK_ESIKLERI:
        sutun = "etiket" if e == ESIK else f"etiket_{e}"
        poz = int(df[sutun].sum())
        satirlar.append({"esik": e, "sutun": sutun, "pozitif": poz, "negatif": len(df) - poz,
                         "toplam": len(df), "pozitif_orani": poz / len(df)})
    tablo_kaydet(pd.DataFrame(satirlar), "sinif_oranlari.csv")

    sayilar = df["etiket"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    cubuklar = ax.bar(["0: Sağlıklı (RUL > 30)", "1: Arızaya yakın (RUL ≤ 30)"],
                      sayilar.values, color=[MAVI, TURUNCU], width=0.55)
    for c, s in zip(cubuklar, sayilar.values):
        ax.text(c.get_x() + c.get_width() / 2, s, f"{s:,} (%{100 * s / len(df):.2f})".replace(",", "."),
                ha="center", va="bottom", color=METIN)
    ax.set_title(f"Eğitim verisi sınıf dağılımı (ESIK = {ESIK})")
    ax.set_xlabel("Sınıf")
    ax.set_ylabel("Satır (döngü) sayısı")
    ax.grid(axis="x", visible=False)
    kaydet(fig, "01_sinif_dagilimi.png")


def motor_omru(df):
    """Motor ömrü histogramı ve min/ortalama/medyan/maks/mod tablosu."""
    omur = df.groupby("unit")["cycle"].max()
    modlar = omur.mode().tolist()
    ozet = pd.DataFrame([{
        "motor_sayisi": len(omur), "min": omur.min(), "ortalama": omur.mean(),
        "medyan": omur.median(), "maks": omur.max(),
        "mod": ";".join(str(m) for m in modlar), "mod_frekansi": int((omur == modlar[0]).sum()),
    }])
    tablo_kaydet(ozet, "motor_omru_ozet.csv")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(omur, bins=20, color=MAVI, edgecolor="white", linewidth=2)
    cizgiler = [("Ortalama", omur.mean(), "-"), ("Medyan", omur.median(), "--")]
    for ad, deger, stil in cizgiler:
        ax.axvline(deger, color=METIN, linestyle=stil, linewidth=1.5, label=f"{ad} = {deger:.1f}")
    ax.set_title("Eğitim motorlarının ömrü (arızaya kadar döngü sayısı)")
    ax.set_xlabel("Motor ömrü (döngü)")
    ax.set_ylabel("Motor sayısı")
    ax.legend(title=f"min = {omur.min()}, maks = {omur.max()}, mod = {', '.join(map(str, modlar))}",
              frameon=False, loc="upper right")
    kaydet(fig, "02_motor_omru_histogram.png")
    return omur


def betimsel(df, omur):
    """14 sensör için betimsel istatistik; mod yalnızca ayrık sütunlarda."""
    ist = df[SENSORLER].agg(["mean", "median", "std", "min", "max"]).T
    ist.columns = ["ortalama", "medyan", "std", "min", "maks"]
    ist.index.name = "sensor"
    tablo_kaydet(ist, "betimsel_istatistik.csv", index=True)

    ayrik = pd.DataFrame([
        {"sutun": "motor_omru", "mod": ";".join(map(str, omur.mode())), "mod_frekansi": int((omur == omur.mode()[0]).sum()), "n": len(omur)},
        {"sutun": "etiket", "mod": str(int(df["etiket"].mode()[0])), "mod_frekansi": int((df["etiket"] == df["etiket"].mode()[0]).sum()), "n": len(df)},
    ])
    tablo_kaydet(ayrik, "mod_ayrik_sutunlar.csv")
    print("Not: sürekli sensörlerde mod anlamsızdır; değerler neredeyse hiç tam tekrar etmez, bu yüzden hesaplanmadı.")


def sabit_sutunlar(df):
    """Her ölçüm sütunu için nunique; sabit sütunları veriden bulur ve beklenenle karşılaştırır."""
    olcumler = ["os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
    nu = df[olcumler].nunique().rename("nunique").to_frame()
    nu["sabit"] = nu["nunique"] == 1
    nu.index.name = "sutun"
    tablo_kaydet(nu, "sabit_sutunlar.csv", index=True)

    bulunan = nu.index[nu["sabit"]].tolist()
    print("\nVeriden bulunan sabit sütunlar:", bulunan)
    print("Beklenen sabit sütunlar:        ", BEKLENEN_SABITLER)
    if set(bulunan) == set(BEKLENEN_SABITLER):
        print("Fark yok.")
    else:
        print("FARK: fazladan", sorted(set(bulunan) - set(BEKLENEN_SABITLER)),
              "eksik", sorted(set(BEKLENEN_SABITLER) - set(bulunan)))
    print("s6 benzersiz değer sayısı:", int(nu.loc["s6", "nunique"]), "değerler:", sorted(df["s6"].unique()))
    ikili = nu.index[nu["nunique"] == 2].tolist()
    print("Yalnızca iki değer alan sütunlar:", ikili)
    kalan = [c for c in olcumler if c.startswith("s") and c not in bulunan and c not in ikili]
    print("Sabit ve ikili olmayan sensörler:", kalan)
    print("Kullanılacak 14 sensör ile aynı mı:", kalan == SENSORLER)
    return bulunan


def ornek_motorlar(df):
    """4 örnek motor x 4 sensör çizgi grafiği, RUL ≤ ESIK bölgesi gölgeli."""
    motorlar = [1, 2, 3, 4]
    sensorler = ["s2", "s4", "s11", "s12"]
    fig, eksenler = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    for ax, s in zip(eksenler.flat, sensorler):
        for m, renk in zip(motorlar, MOTOR_RENKLERI):
            d = df[df["unit"] == m]
            x = d["RUL"].values
            ax.plot(x, d[s].values, color=renk, linewidth=1.5, label=f"Motor {m}")
        ax.axvspan(0, ESIK, color=TURUNCU, alpha=0.12, linewidth=0, label=f"RUL ≤ {ESIK} (arızaya yakın)")
        ax.invert_xaxis()
        ax.set_title(f"Sensör {s}")
        ax.set_ylabel(f"{s} değeri")
    for ax in eksenler[1]:
        ax.set_xlabel("Kalan ömür, RUL (döngü) → arıza")
    eksenler[0, 0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Örnek 4 motorda sensör değerlerinin arızaya doğru değişimi", fontweight="bold")
    fig.tight_layout()
    kaydet(fig, "03_ornek_motorlar_sensorler.png")


def korelasyon(df):
    """14 sensörün Pearson korelasyon ısı haritası."""
    kor = df[SENSORLER].corr()
    kor.index.name = "sensor"
    kor.to_csv(TABLOLAR / "korelasyon_14_sensor.csv")
    print("\n[korelasyon_14_sensor.csv] kaydedildi")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(kor, ax=ax, cmap=DIVERJAN, vmin=-1, vmax=1, center=0, annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=1, linecolor="white",
                cbar_kws={"label": "Pearson korelasyon katsayısı"})
    ax.set_title("14 sensör arasındaki korelasyon (eğitim verisi)")
    ax.set_xlabel("Sensör")
    ax.set_ylabel("Sensör")
    ax.grid(False)
    kaydet(fig, "04_korelasyon_isi_haritasi.png")


def kutu_s11(df):
    """s11 sensörünün etikete göre kutu grafiği."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    gruplar = [df.loc[df["etiket"] == k, "s11"] for k in (0, 1)]
    kutu = ax.boxplot(gruplar, patch_artist=True, widths=0.5,
                      medianprops={"color": METIN, "linewidth": 1.5},
                      flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
    for yama, renk in zip(kutu["boxes"], [MAVI, TURUNCU]):
        yama.set_facecolor(renk)
        yama.set_alpha(0.75)
    ax.set_xticks([1, 2], [f"0: Sağlıklı\n(RUL > {ESIK})", f"1: Arızaya yakın\n(RUL ≤ {ESIK})"])
    ax.set_title("s11 sensörünün sınıfa göre dağılımı")
    ax.set_xlabel("Sınıf")
    ax.set_ylabel("s11 değeri")
    ax.grid(axis="x", visible=False)
    kaydet(fig, "05_s11_kutu_grafigi.png")
    ozet = df.groupby("etiket")["s11"].describe()
    tablo_kaydet(ozet, "s11_sinifa_gore.csv", index=True)


def main():
    SEKILLER.mkdir(parents=True, exist_ok=True)
    TABLOLAR.mkdir(parents=True, exist_ok=True)
    df = egitim_yukle()
    print(f"Eğitim verisi: {df.shape[0]} satır x {df.shape[1]} sütun")

    sinif_dagilimi(df)
    omur = motor_omru(df)
    betimsel(df, omur)
    sabit_sutunlar(df)
    print("\nKullanılacak 14 sensör:", SENSORLER)
    ornek_motorlar(df)
    korelasyon(df)
    kutu_s11(df)
    print("\nAşama 2 bitti.")


if __name__ == "__main__":
    main()
