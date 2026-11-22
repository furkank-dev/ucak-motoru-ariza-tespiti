"""Ortak sabitler ve yardımcı fonksiyonlar."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

# Klasörler
KOK = Path(__file__).resolve().parent.parent
HAM = KOK / "data" / "raw"
ISLENMIS = KOK / "data" / "processed"
SEKILLER = KOK / "outputs" / "figures"
TABLOLAR = KOK / "outputs" / "tables"

# Sabitler
RASTGELE_TOHUM = 42
ESIK = 30
DUYARLILIK_ESIKLERI = [20, 30, 50]

SUTUNLAR = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
SENSORLER = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12",
             "s13", "s14", "s15", "s17", "s20", "s21"]
BEKLENEN_SABITLER = ["os3", "s1", "s5", "s10", "s16", "s18", "s19"]


def ham_oku(yol):
    """Boşlukla ayrılmış, başlıksız C-MAPSS dosyasını 26 sütunla okur."""
    df = pd.read_csv(yol, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")  # sondaki boş sütun
    assert df.shape[1] == 26, f"{yol}: {df.shape[1]} sütun"
    df.columns = SUTUNLAR
    return df


def etiketle(df, rul):
    """RUL sütunundan etiket, etiket_20 ve etiket_50 sütunlarını üretir."""
    df = df.copy()
    df["RUL"] = rul
    df["etiket"] = (df["RUL"] <= ESIK).astype(int)
    df["etiket_20"] = (df["RUL"] <= 20).astype(int)
    df["etiket_50"] = (df["RUL"] <= 50).astype(int)
    return df


def egitim_yukle():
    """İşlenmiş eğitim csv'sini okur."""
    return pd.read_csv(ISLENMIS / "train_FD001_etiketli.csv")


def test_yukle():
    """İşlenmiş test (son döngü) csv'sini okur."""
    return pd.read_csv(ISLENMIS / "test_FD001_son_dongu.csv")


def metrikler(y_gercek, y_tahmin, y_olasilik):
    """Accuracy, Precision, Recall, F1 ve ROC-AUC sözlüğü döndürür (pozitif = 1)."""
    tek_sinif = len(np.unique(y_gercek)) < 2
    return {
        "Accuracy": accuracy_score(y_gercek, y_tahmin),
        "Precision": precision_score(y_gercek, y_tahmin, zero_division=0),
        "Recall": recall_score(y_gercek, y_tahmin, zero_division=0),
        "F1": f1_score(y_gercek, y_tahmin, zero_division=0),
        "ROC-AUC": np.nan if tek_sinif else roc_auc_score(y_gercek, y_olasilik),
    }
