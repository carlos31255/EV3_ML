"""Gráficos del EDA de EV3, derivados del módulo de EV2. Solo recibe train."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib

# En scripts: backend sin pantalla. En Jupyter: se deja el backend inline.
EN_NOTEBOOK = "ipykernel" in sys.modules
if not EN_NOTEBOOK:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "axes.facecolor":    "#ffffff",
    "figure.facecolor":  "#f8f9fa",
    "text.color":        "#212529",
    "axes.labelcolor":   "#212529",
    "xtick.color":       "#212529",
    "ytick.color":       "#212529",
    "axes.edgecolor":    "#dee2e6",
    "grid.color":        "#e9ecef",
    "grid.alpha":        0.8,
})

SPOTIFY_GREEN = "#1DB954"
RED           = "#FF6B6B"
PALETTE       = [SPOTIFY_GREEN, "#1E90FF", "#FFD700", RED, "#B19CD9", "#FF8C00"]
OUTPUT_DIR    = os.path.join("output", "graficos")

NUM_COLS  = ["danceability", "energy", "loudness", "valence", "tempo",
             "acousticness", "speechiness", "instrumentalness", "liveness", "duration_ms"]
CORR_COLS = NUM_COLS + ["explicit", "mode", "key", "time_signature"]
VARS_BOX  = ["danceability", "energy", "loudness", "valence",
             "acousticness", "instrumentalness"]

# Dos recortes de la matriz de correlacion (grafico_correlaciones):
# candidatas = lo de CORR_COLS + n_generos (variable descartada, para ver por que)
# finales    = lo de CORR_COLS + las 5 variables de texto (las que SI entran al modelo), sin n_generos
CORR_COLS_CANDIDATAS = CORR_COLS + ["n_generos"]
CORR_COLS_FINALES    = CORR_COLS + ["n_artistas", "t_palabras", "t_remix", "t_live", "t_feat"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig, nombre: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ruta = os.path.join(OUTPUT_DIR, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    if EN_NOTEBOOK:
        plt.show()
    plt.close(fig)
    print(f"[graficos] Guardado: {ruta}")


def _explotar_generos(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (cancion, genero); track_genre viene como 'a|b|c'."""
    return df.assign(genero=df["track_genre"].str.split("|")).explode("genero")


# ── EDA ───────────────────────────────────────────────────────────────────────

def grafico_generos_por_cancion(dist: pd.Series) -> None:
    """dist: canciones por numero de generos distintos (value_counts ordenado por indice)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(dist.index.astype(str), dist.values, color=SPOTIFY_GREEN, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("Numero de generos distintos por cancion")
    ax.set_ylabel("Canciones (escala log)")
    ax.set_title("Una misma cancion aparece en varios generos")
    fig.tight_layout()
    _save(fig, "00_generos_por_cancion.png")


def grafico_distribucion_popularidad(df: pd.DataFrame, umbral: float) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(df["popularity"], bins=50, color=SPOTIFY_GREEN, alpha=0.85)
    ax[0].axvline(umbral, color="#FFD700", ls="--", label=f"P80 (train) = {umbral:.0f}")
    ax[0].legend()
    ax[0].set_title("Distribucion de popularidad (train)")
    ax[1].hist(df.loc[df["popularity"] > 0, "popularity"], bins=50,
               color=SPOTIFY_GREEN, alpha=0.85)
    ax[1].axvline(umbral, color="#FFD700", ls="--")
    ax[1].set_title("Sin los casos con popularidad = 0")
    for a in ax:
        a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    _save(fig, "01_distribucion_popularidad.png")


def grafico_distribuciones_numericas(df: pd.DataFrame, cols: list = None) -> None:
    cols = cols or NUM_COLS
    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    for ax, c in zip(axes.ravel(), cols):
        ax.hist(df[c].dropna(), bins=40, color=SPOTIFY_GREEN, alpha=0.85)
        ax.set_title(c, fontsize=10)
    fig.tight_layout()
    _save(fig, "02_distribuciones_numericas.png")


def tabla_hit_rate_genero(df: pd.DataFrame) -> pd.DataFrame:
    g = _explotar_generos(df)
    return (g.groupby("genero")["hit"].agg(["mean", "size"])
              .rename(columns={"mean": "hit_rate", "size": "n"})
              .sort_values("hit_rate"))


def grafico_hit_rate_genero(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    tasa = tabla_hit_rate_genero(df)
    global_rate = df["hit"].mean()
    extremos = pd.concat([tasa.head(n), tasa.tail(n)])
    extremos = extremos.loc[~extremos.index.duplicated()]
    fig, ax = plt.subplots(figsize=(8, 9))
    colores = [RED if v < global_rate else SPOTIFY_GREEN for v in extremos["hit_rate"]]
    ax.barh(extremos.index, extremos["hit_rate"], color=colores, alpha=0.85)
    ax.axvline(global_rate, color="black", ls="--", label=f"Global = {global_rate:.2f}")
    ax.set_title(f"Hit rate por genero (train): {n} mas bajos y {n} mas altos")
    ax.legend()
    fig.tight_layout()
    _save(fig, "03_hit_rate_por_genero.png")
    return tasa


def grafico_hit_rate_n_generos(df: pd.DataFrame) -> pd.DataFrame:
    t = (df.groupby("n_generos")["hit"].agg(["mean", "size"])
           .rename(columns={"mean": "hit_rate", "size": "n"}))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(t.index.astype(str), t["hit_rate"], color=SPOTIFY_GREEN, alpha=0.85)
    ax.axhline(df["hit"].mean(), color="black", ls="--")
    ax.set_xlabel("Numero de generos")
    ax.set_ylabel("Hit rate")
    ax.set_title("Hit rate segun numero de generos de la cancion")
    fig.tight_layout()
    _save(fig, "04_hit_rate_n_generos.png")
    return t.round(3)


def grafico_correlaciones(df: pd.DataFrame, cols: list = None, nombre: str = "05_correlaciones.png",
                          titulo: str = "Correlaciones (Pearson)") -> pd.DataFrame:
    cols = cols or CORR_COLS
    corr = df[cols + ["popularity", "hit"]].astype(float).corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                linewidths=0.5, linecolor="#0f0f1a", ax=ax,
                annot_kws={"size": 7}, cbar_kws={"shrink": .8})
    ax.set_title(titulo)
    fig.tight_layout()
    _save(fig, nombre)
    return corr


def grafico_features_por_clase(df: pd.DataFrame, cols: list = None) -> None:
    cols = cols or VARS_BOX
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, c in zip(axes.ravel(), cols):
        sns.boxplot(data=df, x="hit", y=c, hue="hit", legend=False, ax=ax, showfliers=False,
                    palette=[RED, SPOTIFY_GREEN])
        ax.set_xticks([0, 1], ["No hit", "Hit"])
        ax.set_xlabel("")
        ax.set_title(c)
    fig.tight_layout()
    _save(fig, "06_features_por_clase.png")


def grafico_popularidad_cero(df: pd.DataFrame) -> pd.Series:
    g = _explotar_generos(df)
    cero = (g.assign(cero=(g["popularity"] == 0))
              .groupby("genero")["cero"].mean().sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cero.values, bins=20, color=SPOTIFY_GREEN, alpha=0.85)
    ax.set_xlabel("Proporcion de canciones con popularidad 0 en el genero")
    ax.set_ylabel("Generos")
    ax.set_title("Popularidad 0: ruido uniforme o concentrado?")
    fig.tight_layout()
    _save(fig, "07_popularidad_cero.png")
    return cero


