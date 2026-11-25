"""Arayüzün giriş formu için test motorlarının son döngüsünü ve sensör aralıklarını veri/veri.json dosyasına yazar.

Formdaki "test verisinden doldur" bu dosyadan okur. Model tahmini burada yok, onu backend yapar.
"""
import json
from pathlib import Path

import pandas as pd

KOK = Path(__file__).resolve().parent.parent
ISLENMIS = KOK / "data" / "processed"
CIKTI = Path(__file__).resolve().parent / "veri" / "veri.json"

# C-MAPSS belgesindeki sensör adları (Saxena ve ark., 2008)
SENSORLER = [
    ("s2", "LPC çıkış sıcaklığı", "°R"),
    ("s3", "HPC çıkış sıcaklığı", "°R"),
    ("s4", "LPT çıkış sıcaklığı", "°R"),
    ("s7", "HPC çıkış basıncı", "psia"),
    ("s8", "Fan devri", "rpm"),
    ("s9", "Çekirdek devri", "rpm"),
    ("s11", "HPC çıkış statik basıncı", "psia"),
    ("s12", "Yakıt akışı / Ps30 oranı", "pps/psi"),
    ("s13", "Düzeltilmiş fan devri", "rpm"),
    ("s14", "Düzeltilmiş çekirdek devri", "rpm"),
    ("s15", "Bypass oranı", ""),
    ("s17", "Tahliye entalpisi", ""),
    ("s20", "HPT soğutma havası", "lbm/s"),
    ("s21", "LPT soğutma havası", "lbm/s"),
]


def main():
    egitim = pd.read_csv(ISLENMIS / "train_FD001_etiketli.csv")
    test = pd.read_csv(ISLENMIS / "test_FD001_son_dongu.csv").sort_values("unit")
    sensorler = [{"ad": ad, "aciklama": aciklama, "birim": birim,
                  "min": float(egitim[ad].min()), "max": float(egitim[ad].max())}
                 for ad, aciklama, birim in SENSORLER]
    motorlar = [{"id": int(s["unit"]), "son_dongu": int(s["cycle"]), "gercek_rul": int(s["RUL"]),
                 "gercek_etiket": int(s["etiket"]),
                 "sensorler": {ad: float(s[ad]) for ad, *_ in SENSORLER}}
                for _, s in test.iterrows()]
    # Kendini kontrol: 100 motor, 25'i arızaya yakın, eksik değer yok
    assert len(motorlar) == 100 and sum(m["gercek_etiket"] for m in motorlar) == 25
    assert not test[[ad for ad, *_ in SENSORLER]].isna().any().any()
    CIKTI.parent.mkdir(exist_ok=True)
    CIKTI.write_text(json.dumps({"sensorler": sensorler, "motorlar": motorlar}, ensure_ascii=False, indent=1),
                     encoding="utf-8")
    print("yazıldı:", CIKTI)


if __name__ == "__main__":
    main()
