"""Generación de figuras estadísticas y diagramas conceptuales (300 DPI)."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.ticker import FuncFormatter
from scipy import stats as st

from .config import DIR_IMG, DIR_RESULTADOS, PARAMETROS

INK = "#1F242B"
SLATE = "#4C5B61"
PEWTER = "#9CA3AF"
GRID_COLOR = "#E5E7EB"
BURGUNDY = "#7A1F2B"
CRIMSON = "#A63A32"
OCHRE = "#C47D2C"
FOREST = "#2B6148"
PLUM = "#5B5470"

COLOR_ESC = {
    "E1_Base": CRIMSON,
    "E2_Conservador": OCHRE,
    "E3_Optimizado": FOREST,
    "E4_Periodico": PLUM,
}
COLOR_COMP = {
    "costo_mantenimiento": SLATE,
    "costo_ordenamiento": PEWTER,
    "costo_faltante": CRIMSON,
}
ETIQ_COMP = {
    "costo_mantenimiento": "Mantenimiento",
    "costo_ordenamiento": "Ordenamiento",
    "costo_faltante": "Faltantes",
}
NOMBRE_ESC = {
    "E1_Base": "E1: Base (s, Q)\nsin SS",
    "E2_Conservador": "E2: Conservador\n(s, Q) con SS",
    "E3_Optimizado": "E3: Optimizado\n(s*, Q*)",
    "E4_Periodico": "E4: Periódico\n(R*, S*)",
}

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "academic_warm",
    ["#FAF7F2", "#EFE2D0", "#D99B78", "#B34B3E", "#6E1D27"],
)


def _aplicar_estilo() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#D1D5DB",
        "axes.linewidth": 0.85,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.65,
        "grid.linewidth": 0.6,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK,
        "axes.titlesize": 12.0,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlepad": 12,
        "legend.frameon": True,
        "legend.facecolor": "#FFFFFF",
        "legend.edgecolor": "#E5E7EB",
        "legend.fontsize": 9.2,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


def _fmt_es(x: float, decimales: int = 0) -> str:
    s = f"{x:,.{decimales}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def _formatter_es(decimales: int = 0) -> FuncFormatter:
    return FuncFormatter(lambda val, _: _fmt_es(val, decimales))


def _limpiar(ax, grid_y: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_y:
        ax.yaxis.grid(True, alpha=0.65)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)


def _guardar(fig, nombre: str) -> None:
    DIR_IMG.mkdir(parents=True, exist_ok=True)
    ruta = DIR_IMG / nombre
    fig.savefig(ruta, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"  OK {ruta.name} ({ruta.stat().st_size / 1024:.0f} KB)")


def _cargar(nombre: str) -> pd.DataFrame:
    return pd.read_csv(DIR_RESULTADOS / nombre)


def fig_distribuciones() -> None:
    rng = np.random.default_rng(0)
    n = 20_000
    dem = rng.poisson(PARAMETROS.lambda_demanda, size=n)
    lt = rng.integers(PARAMETROS.lt_a, PARAMETROS.lt_b + 1, size=n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.7), gridspec_kw={"wspace": 0.24})

    ks = np.arange(0, dem.max() + 1)
    pmf = st.poisson.pmf(ks, PARAMETROS.lambda_demanda) * n
    ax1.bar(ks, np.bincount(dem, minlength=len(ks)), color=SLATE, alpha=0.88,
            label="Empírica (n = 20.000)", zorder=3)
    ax1.plot(ks, pmf, "o", color=BURGUNDY, markersize=4.5, markerfacecolor="white",
             markeredgewidth=1.4, label=r"Poisson($\lambda = 8$) teórica", zorder=4)
    ax1.set_xlabel(r"Demanda diaria $D_t$ (unidades)")
    ax1.set_ylabel("Frecuencia absoluta")
    ax1.yaxis.set_major_formatter(_formatter_es(0))
    ax1.set_title(r"Demanda diaria $\sim$ Poisson($\lambda = 8$)", loc="left")
    ax1.legend(loc="upper right", framealpha=0.95)
    _limpiar(ax1)

    vals = np.arange(PARAMETROS.lt_a, PARAMETROS.lt_b + 1)
    p_teo = n / len(vals)
    ax2.bar(vals, np.bincount(lt)[vals], width=0.52, color=FOREST, alpha=0.88,
            label="Empírica (n = 20.000)", zorder=3)
    ax2.axhline(p_teo, color=BURGUNDY, linestyle="--", linewidth=1.6,
                label="Uniforme discreta teórica (5.000)", zorder=4)
    ax2.set_xticks(vals)
    ax2.set_xlabel(r"Lead time $LT$ (días)")
    ax2.set_ylabel("Frecuencia absoluta")
    ax2.yaxis.set_major_formatter(_formatter_es(0))
    ax2.set_ylim(0, p_teo * 1.28)
    ax2.set_title(r"Lead time $\sim$ UniformeDiscreta$\{2, 3, 4, 5\}$", loc="left")
    ax2.legend(loc="upper right", framealpha=0.95)
    _limpiar(ax2)

    fig.suptitle(
        "Validación visual de los generadores de entrada (20.000 muestras aleatorias)",
        fontsize=13, fontweight="bold", color=INK, y=1.02,
    )
    _guardar(fig, "fig01_distribuciones_entrada.png")


def fig_warmup() -> None:
    df = _cargar("warmup_costos.csv")
    fig, ax = plt.subplots(figsize=(10.8, 4.8))

    colores = {
        "I0=0": (CRIMSON, r"$I_0 = 0$ (arranque vacío)"),
        "I0=28": (OCHRE, r"$I_0 = 28$ ($s_0 = \mu_D \cdot \mu_{LT}$)"),
        "I0=200": (FOREST, r"$I_0 = 200$ (régimen: lote EOQ)"),
    }
    for ini, g in df.groupby("inicial"):
        g = g.sort_values("dia")
        serie = g["costo_diario_prom"].rolling(10, min_periods=1).mean()
        color, etiq = colores[ini]
        ax.plot(g["dia"], serie, color=color, linewidth=1.8, label=etiq)

    ax.axvspan(0, PARAMETROS.warmup_dias, color="#F3F4F6", zorder=0)
    ax.axvline(PARAMETROS.warmup_dias, color=INK, linestyle=":", linewidth=1.35)
    y_top = ax.get_ylim()[1]
    ax.text(
        PARAMETROS.warmup_dias + 6, y_top * 0.90,
        "Fin del calentamiento\n(t = 60 días ≈ 2,4 ciclos EOQ)",
        fontsize=9.5, color=INK, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#E5E7EB", alpha=0.92),
    )
    ax.set_xlim(0, PARAMETROS.dias_sim)
    ax.set_xlabel("Día de simulación (t)")
    ax.set_ylabel("Costo diario promedio (USD/día)")
    ax.set_title(
        "Análisis del período de calentamiento: convergencia antes del día 60 (media móvil 10 días)",
        loc="left",
    )
    ax.legend(loc="upper right", framealpha=0.95)
    _limpiar(ax)
    _guardar(fig, "fig02_analisis_warmup.png")


def fig_traza_inventario() -> None:
    df = _cargar("traza_ejemplo.csv")
    s = PARAMETROS.s_con_ss
    dias = df["dia"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11.8, 7.6), sharex=True, gridspec_kw={"hspace": 0.36}
    )

    ax1.axvspan(0, PARAMETROS.warmup_dias, color="#F3F4F6", zorder=0, label="Warm-up (60 días)")
    ax1.step(dias, df["inv_fin"], where="post", color=SLATE, linewidth=1.45,
             label=r"Inventario físico $I_t$ (cierre)")
    quiebres = df["perdidas"] > 0
    if quiebres.any():
        ax1.fill_between(dias, 0, df["inv_fin"], where=quiebres, color=CRIMSON,
                         alpha=0.35, step="post", label="Días con venta perdida")
    arribos = df[df["arribos"] > 0]
    ax1.scatter(arribos["dia"] - 0.5, arribos["inv_ini"], marker="v", s=46,
                color=FOREST, zorder=5, label=r"Arribo de orden ($+Q$)")
    ax1.axhline(s, color=OCHRE, linestyle="--", linewidth=1.45, label=f"Punto de reorden s = {s}")
    ax1.annotate(
        f"s = {s}", xy=(1.01, s), xycoords=("axes fraction", "data"),
        va="center", ha="left", fontsize=9.2, fontweight="bold", color=OCHRE,
    )
    ax1.set_ylim(-6, max(df["inv_ini"].max(), 230) * 1.10)
    ax1.set_ylabel("Unidades en stock")
    ax1.set_title(
        "Evolución del inventario físico — Política (s, Q) con SS (E2, réplica de ejemplo)",
        loc="left", pad=28,
    )
    ax1.legend(
        loc="lower left", bbox_to_anchor=(0.0, 1.01, 1.0, 0.12),
        ncols=5, frameon=False, fontsize=8.8, handlelength=1.8, columnspacing=1.2,
    )
    _limpiar(ax1)

    ax2.axvspan(0, PARAMETROS.warmup_dias, color="#F3F4F6", zorder=0)
    ax2.step(dias, df["ip_cierre"], where="post", color=PLUM, linewidth=1.4,
             label=r"Posición de inventario $IP_t = I_t + \mathrm{en\ tr\acute{a}nsito}$")
    pedidos = df[df["pedido"] > 0]
    ax2.scatter(pedidos["dia"], pedidos["ip_cierre"], marker="o", s=28,
                color=BURGUNDY, zorder=5, label="Emisión de orden")
    ax2.axhline(s, color=OCHRE, linestyle="--", linewidth=1.45, label=f"Umbral s = {s}")
    ax2.annotate(
        f"s = {s}", xy=(1.01, s), xycoords=("axes fraction", "data"),
        va="center", ha="left", fontsize=9.2, fontweight="bold", color=OCHRE,
    )
    ax2.set_ylim(28, df["ip_cierre"].max() * 1.08)
    ax2.set_xlim(0, dias[-1] + 2)
    ax2.set_ylabel("Unidades (IP)")
    ax2.set_xlabel("Día de simulación (t)")
    ax2.set_title(
        r"Posición de inventario: el disparador de reposición es $IP_t \leq s$ al cierre del día",
        loc="left", pad=28,
    )
    ax2.legend(
        loc="lower left", bbox_to_anchor=(0.0, 1.01, 1.0, 0.12),
        ncols=3, frameon=False, fontsize=8.8, handlelength=1.8, columnspacing=1.4,
    )
    _limpiar(ax2)

    _guardar(fig, "fig03_evolucion_inventario.png")


def fig_costos_composicion() -> None:
    res = _cargar("resumen_escenarios.csv")
    escenarios = ["E1_Base", "E2_Conservador", "E3_Optimizado", "E4_Periodico"]
    comps = ["costo_mantenimiento", "costo_ordenamiento", "costo_faltante"]

    medias = {c: [] for c in comps}
    totales, err = [], []
    for e in escenarios:
        sub = res[res["escenario"] == e]
        for c in comps:
            medias[c].append(float(sub[sub["metrica"] == c]["media"].iloc[0]))
        tot = sub[sub["metrica"] == "costo_total"].iloc[0]
        totales.append(float(tot["media"]))
        err.append(float(tot["ic95_sup"] - tot["media"]))

    fig, ax = plt.subplots(figsize=(10.6, 5.7))
    x = np.arange(len(escenarios))
    base = np.zeros(len(escenarios))
    for c in comps:
        vals = np.array(medias[c])
        ax.bar(x, vals, bottom=base, width=0.54, color=COLOR_COMP[c],
               label=ETIQ_COMP[c], zorder=3, edgecolor="white", linewidth=0.7)
        for xi, (v, b) in enumerate(zip(vals, base)):
            if v > totales[0] * 0.045:
                ax.text(xi, b + v / 2, _fmt_es(v, 0), ha="center", va="center",
                        color="white", fontsize=9.2, fontweight="bold")
        base += vals

    ax.errorbar(x, totales, yerr=err, fmt="none", ecolor=INK, elinewidth=1.4,
                capsize=5, capthick=1.3, zorder=5, label="IC 95 % del costo total")
    for xi, (t, e_val) in enumerate(zip(totales, err)):
        ax.text(xi, t + e_val + 95, f"{_fmt_es(t, 0)} USD", ha="center", va="bottom",
                fontsize=9.8, fontweight="bold", color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([NOMBRE_ESC[e] for e in escenarios], fontsize=9.6)
    ax.set_ylabel("Costo anual promedio (USD/año)")
    ax.yaxis.set_major_formatter(_formatter_es(0))
    ax.set_ylim(0, max(totales) * 1.20)
    pct_falt = 100.0 * medias["costo_faltante"][0] / totales[0]
    ax.set_title(
        "Composición del costo anual promedio por escenario\n"
        f"Sin stock de seguridad, los faltantes representan el {_fmt_es(pct_falt, 1)} % del costo total",
        loc="left",
    )
    ax.legend(loc="upper right", framealpha=0.95)
    _limpiar(ax)
    _guardar(fig, "fig04_composicion_costos.png")


def fig_boxplots() -> None:
    rep = _cargar("replicas_metricas.csv")
    escenarios = ["E1_Base", "E2_Conservador", "E3_Optimizado", "E4_Periodico"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.4), gridspec_kw={"wspace": 0.24})

    data_tc = [rep[rep["escenario"] == e]["costo_total"] for e in escenarios]
    bp = ax1.boxplot(
        data_tc, patch_artist=True, widths=0.50, showfliers=True,
        flierprops=dict(marker="o", markersize=3.5, markerfacecolor=PEWTER, markeredgecolor="none", alpha=0.65),
        medianprops=dict(color="white", linewidth=1.8),
        whiskerprops=dict(color=INK, linewidth=1.1),
        capprops=dict(color=INK, linewidth=1.1),
    )
    for parche, e in zip(bp["boxes"], escenarios):
        parche.set_facecolor(COLOR_ESC[e])
        parche.set_edgecolor("white")
    ax1.set_xticks(range(1, 5))
    ax1.set_xticklabels([NOMBRE_ESC[e] for e in escenarios], fontsize=8.8)
    ax1.set_ylabel("Costo total anual (USD/año)")
    ax1.yaxis.set_major_formatter(_formatter_es(0))
    ax1.set_title("Distribución del costo total anual (100 réplicas)", loc="left")
    _limpiar(ax1)

    data_fr = [rep[rep["escenario"] == e]["fill_rate_pct"] for e in escenarios]
    bp2 = ax2.boxplot(
        data_fr, patch_artist=True, widths=0.50, showfliers=True,
        flierprops=dict(marker="o", markersize=3.5, markerfacecolor=PEWTER, markeredgecolor="none", alpha=0.65),
        medianprops=dict(color="white", linewidth=1.8),
        whiskerprops=dict(color=INK, linewidth=1.1),
        capprops=dict(color=INK, linewidth=1.1),
    )
    for parche, e in zip(bp2["boxes"], escenarios):
        parche.set_facecolor(COLOR_ESC[e])
        parche.set_edgecolor("white")
    ax2.axhline(PARAMETROS.fill_rate_objetivo, color=BURGUNDY, linestyle="--", linewidth=1.45, zorder=4)
    ax2.text(
        1.10, PARAMETROS.fill_rate_objetivo - 0.16, "Umbral 95 %",
        color=BURGUNDY, fontsize=9.0, ha="left", va="top", fontweight="bold", zorder=5,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.92),
    )
    ax2.set_xticks(range(1, 5))
    ax2.set_xticklabels([NOMBRE_ESC[e] for e in escenarios], fontsize=8.8)
    ax2.set_ylabel("Fill Rate (%)")
    ax2.set_ylim(93.4, 100.35)
    ax2.set_title("Distribución del Fill Rate por escenario", loc="left")
    _limpiar(ax2)

    ax_ins = ax2.inset_axes([0.45, 0.16, 0.52, 0.43])
    ax_ins.patch.set_facecolor("white")
    ax_ins.patch.set_alpha(1.0)
    ax_ins.set_zorder(6)
    bp_ins = ax_ins.boxplot(
        data_fr[1:], patch_artist=True, widths=0.45, showfliers=True,
        flierprops=dict(marker="o", markersize=2.8, markerfacecolor=PEWTER, markeredgecolor="none", alpha=0.65),
        medianprops=dict(color="white", linewidth=1.5),
        whiskerprops=dict(color=INK, linewidth=0.9),
        capprops=dict(color=INK, linewidth=0.9),
    )
    for parche, e in zip(bp_ins["boxes"], escenarios[1:]):
        parche.set_facecolor(COLOR_ESC[e])
        parche.set_edgecolor("white")
    ax_ins.set_ylim(98.95, 100.08)
    ax_ins.set_xticks([1, 2, 3])
    ax_ins.set_xticklabels(["E2", "E3", "E4"], fontsize=8.5)
    ax_ins.tick_params(axis="y", labelsize=8.0)
    ax_ins.set_title("Detalle 99 %–100 % (E2–E4)", fontsize=8.5, fontweight="bold", pad=4)
    ax_ins.yaxis.grid(True, alpha=0.5)
    ax_ins.spines["top"].set_visible(False)
    ax_ins.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax_ins.spines[spine].set_color("#9CA3AF")

    _guardar(fig, "fig05_boxplots_costo_servicio.png")


def fig_ic() -> None:
    res = _cargar("resumen_escenarios.csv")
    escenarios = ["E1_Base", "E2_Conservador", "E3_Optimizado", "E4_Periodico"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.2), gridspec_kw={"wspace": 0.24})

    for ax, met, ylab in [
        (ax1, "costo_total", "Costo total anual (USD/año)"),
        (ax2, "fill_rate_pct", "Fill Rate (%)"),
    ]:
        sub = res[res["metrica"] == met].set_index("escenario").loc[escenarios]
        x = np.arange(len(escenarios))
        medio = sub["media"].to_numpy()
        medio_err = medio - sub["ic95_inf"].to_numpy()

        for xi, (m, e, esc) in enumerate(zip(medio, medio_err, escenarios)):
            ax.errorbar(
                xi, m, yerr=e, fmt="o", markersize=8.5,
                markerfacecolor="white", markeredgewidth=2.1, capsize=6,
                capthick=1.5, elinewidth=1.8, color=COLOR_ESC[esc], zorder=4,
            )
            if met == "costo_total":
                ax.annotate(
                    f"{_fmt_es(m, 0)} ± {_fmt_es(e, 0)}",
                    xy=(xi, m + e), xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.8, fontweight="bold", color=COLOR_ESC[esc],
                )
            else:
                offset_y = 10 if xi != 2 else -16
                va = "bottom" if xi != 2 else "top"
                ax.annotate(
                    f"{_fmt_es(m, 2)} %",
                    xy=(xi, m), xytext=(0, offset_y), textcoords="offset points",
                    ha="center", va=va, fontsize=8.8, fontweight="bold", color=COLOR_ESC[esc],
                )

        if met == "costo_total":
            ref = PARAMETROS.costo_total_eoq()
            ax.axhline(ref, color=INK, linestyle=":", linewidth=1.35)
            ax.text(2.0, ref - 80, f"Referencia analítica EOQ = {_fmt_es(ref, 0)} USD/año",
                    fontsize=9.0, color=INK, ha="center", va="top")
            ax.set_ylim(1150, 4750)
            ax.yaxis.set_major_formatter(_formatter_es(0))
        else:
            ax.axhline(95.0, color=BURGUNDY, linestyle="--", linewidth=1.35)
            ax.text(3.35, 95.18, "Objetivo mínimo: 95 %", fontsize=9.0,
                    color=BURGUNDY, ha="right", fontweight="bold")
            ax.set_ylim(94.4, 100.65)

        ax.set_xlim(-0.5, 3.5)
        ax.set_xticks(x)
        ax.set_xticklabels([NOMBRE_ESC[e] for e in escenarios], fontsize=8.8)
        ax.set_ylabel(ylab)
        _limpiar(ax)

    min_fr_e24 = float(res[(res["metrica"] == "fill_rate_pct") & (res["escenario"] != "E1_Base")]["media"].min())
    min_fr_e1 = float(res[(res["metrica"] == "fill_rate_pct") & (res["escenario"] == "E1_Base")]["minimo"].iloc[0])
    ax1.set_title("Costo total anual (IC 95 %)\nE3 logra el menor costo y la menor variabilidad", loc="left")
    ax2.set_title(
        f"Fill Rate promedio (IC 95 %)\nE2–E4 superan {_fmt_es(min_fr_e24, 1)} %; E1 cae hasta {_fmt_es(min_fr_e1, 1)} % en réplicas",
        loc="left",
    )
    _guardar(fig, "fig06_intervalos_confianza.png")


def fig_heatmap_sQ() -> None:
    grid1 = _cargar("grid_sQ.csv")
    grid2 = _cargar("grid_sQ_fino.csv")
    with open(DIR_RESULTADOS / "optimos.json", encoding="utf-8") as f:
        opt = json.load(f)

    s1 = sorted(grid1["s"].unique())
    q1 = sorted(grid1["Q"].unique())
    c1 = grid1.pivot(index="s", columns="Q", values="costo_medio").loc[s1, q1]
    fr1 = grid1.pivot(index="s", columns="Q", values="fill_rate_medio").loc[s1, q1]

    s2 = sorted(grid2["s"].unique())
    q2 = sorted(grid2["Q"].unique())
    c2 = grid2.pivot(index="s", columns="Q", values="costo_medio").loc[s2, q2]
    fr2 = grid2.pivot(index="s", columns="Q", values="fill_rate_medio").loc[s2, q2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.8), gridspec_kw={"wspace": 0.30})
    caja = dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D1D5DB", alpha=0.95)

    vmax1 = float(np.nanpercentile(c1.to_numpy(), 92))
    pc1 = ax1.pcolormesh(q1, s1, c1.to_numpy(), shading="nearest",
                         cmap=HEATMAP_CMAP, vmin=c1.to_numpy().min(), vmax=vmax1)
    cb1 = fig.colorbar(pc1, ax=ax1, shrink=0.88, pad=0.03)
    cb1.set_label("Costo medio (USD/año, 30 rep.)", fontsize=8.8, color=INK)
    cb1.ax.yaxis.set_major_formatter(_formatter_es(0))
    cb1.outline.set_visible(False)

    cs1 = ax1.contour(q1, s1, fr1.to_numpy(), levels=[95.0], colors=INK, linewidths=1.7, linestyles="--")
    for txt in ax1.clabel(cs1, fmt={95.0: "FR = 95 %"}, fontsize=8.5, inline=True, inline_spacing=5):
        txt.set_bbox(dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=INK, alpha=0.92))

    rect_zoom = FancyBboxPatch(
        (min(q2) - 4, min(s2) - 0.5),
        (max(q2) - min(q2)) + 8,
        (max(s2) - min(s2)) + 1.0,
        boxstyle="round,pad=0.0",
        facecolor="none", edgecolor=INK, linewidth=1.35, linestyle=":", zorder=5,
    )
    ax1.add_patch(rect_zoom)
    ax1.text(
        0.5 * (min(q2) + max(q2)), max(s2) + 1.8, "Región Etapa 2",
        fontsize=8.2, color=INK, fontweight="bold", ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.90),
    )

    ax1.scatter(PARAMETROS.eoq, PARAMETROS.s_sin_ss, marker="D", s=72, color=BURGUNDY, edgecolor="white", linewidth=1.2, zorder=6)
    ax1.annotate(
        f"E1 sin SS ({PARAMETROS.s_sin_ss}, 200)",
        (PARAMETROS.eoq, PARAMETROS.s_sin_ss), textcoords="offset points", xytext=(16, -2),
        ha="left", va="center", fontsize=8.6, fontweight="bold", color=BURGUNDY, bbox=caja, zorder=7,
    )

    ax1.set_xlabel("Tamaño de lote Q (unidades)")
    ax1.set_ylabel("Punto de reorden s (unidades)")
    ax1.set_title("Etapa 1: exploración global (Δs = 4, ΔQ = 20)", loc="left")
    ax1.set_xticks(q1[::2])
    ax1.set_yticks(s1[::2])
    _limpiar(ax1, grid_y=False)

    c2_arr = c2.to_numpy()
    c2_min = float(c2_arr.min())
    vmax2 = float(np.nanpercentile(c2_arr, 90))
    pc2 = ax2.pcolormesh(q2, s2, c2_arr, shading="nearest",
                         cmap=HEATMAP_CMAP, vmin=c2_min, vmax=vmax2)
    cb2 = fig.colorbar(pc2, ax=ax2, shrink=0.88, pad=0.03)
    cb2.set_label("Costo medio (USD/año, 100 rep.)", fontsize=8.8, color=INK)
    cb2.ax.yaxis.set_major_formatter(_formatter_es(0))
    cb2.outline.set_visible(False)

    ax2.contour(q2, s2, c2_arr, levels=[c2_min + 10.0], colors=FOREST, linewidths=1.5, linestyles="-.")
    if float(fr2.to_numpy().min()) < 95.0 < float(fr2.to_numpy().max()):
        cs2 = ax2.contour(q2, s2, fr2.to_numpy(), levels=[95.0], colors=INK, linewidths=1.5, linestyles="--")
        for txt in ax2.clabel(cs2, fmt={95.0: "FR = 95 %"}, fontsize=8.3, inline=True, inline_spacing=4):
            txt.set_bbox(dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=INK, alpha=0.92))

    ax2.scatter(opt["Q_opt"], opt["s_opt"], marker="*", s=340, color=FOREST, edgecolor="white", linewidth=0.9, zorder=7)
    ax2.annotate(
        f"E3 óptimo ({opt['s_opt']}, {opt['Q_opt']})\nContorno ≤ +10 USD",
        xy=(opt["Q_opt"], opt["s_opt"]),
        xytext=(max(q2) - 42, max(s2) - 2.3), textcoords="data",
        ha="center", va="center", fontsize=8.8, fontweight="bold", color=FOREST,
        bbox=caja, arrowprops=dict(arrowstyle="-", color=FOREST, lw=1.1), zorder=8,
    )

    ax2.scatter(PARAMETROS.eoq, PARAMETROS.s_con_ss, marker="o", s=85, color=OCHRE, edgecolor="white", linewidth=1.2, zorder=7)
    ax2.annotate(
        f"E2 analítico ({PARAMETROS.s_con_ss}, 200)",
        xy=(PARAMETROS.eoq, PARAMETROS.s_con_ss),
        xytext=(min(q2) + 28, min(s2) + 2.8), textcoords="data",
        ha="center", va="center", fontsize=8.6, fontweight="bold", color="#8F5718",
        bbox=caja, arrowprops=dict(arrowstyle="-", color="#8F5718", lw=1.0), zorder=8,
    )

    ax2.set_xlabel("Tamaño de lote Q (unidades)")
    ax2.set_ylabel("Punto de reorden s (unidades)")
    ax2.set_title("Etapa 2: refinamiento fino (Δs = 1, ΔQ = 10)", loc="left")
    ax2.set_xticks(q2[::2])
    ax2.set_yticks(s2[::3])
    _limpiar(ax2, grid_y=False)

    fig.suptitle(
        "Superficie de costo total en el plano (s, Q): exploración global y meseta cuasi-óptima",
        fontsize=12.6, fontweight="bold", color=INK, y=0.99,
    )
    _guardar(fig, "fig07_heatmap_sQ.png")


def _caja_diagrama(ax, x, y, w, h, texto, facecolor="#FFFFFF", edgecolor=INK,
                   lw=1.3, fontsize=9.2, bold=False, rounding=0.02):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={rounding}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, texto, ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal", color=INK, zorder=4)


def _rombo_diagrama(ax, x, y, w, h, texto, facecolor="#F5F3EF", edgecolor=BURGUNDY,
                    lw=1.35, fontsize=9.0):
    pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    poly = Polygon(pts, closed=True, facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=3)
    ax.add_patch(poly)
    ax.text(x, y, texto, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=INK, zorder=4)


def _flecha(ax, p1, p2, texto="", tx_off=(0, 0), rad=0.0, fontsize=8.4):
    arr = FancyArrowPatch(
        p1, p2, arrowstyle="-|>,head_length=6,head_width=3.5",
        connectionstyle=f"arc3,rad={rad}", color=INK, linewidth=1.2, zorder=2,
    )
    ax.add_patch(arr)
    if texto:
        mx = (p1[0] + p2[0]) / 2 + tx_off[0]
        my = (p1[1] + p2[1]) / 2 + tx_off[1]
        ax.text(mx, my, texto, ha="center", va="center", fontsize=fontsize,
                color=INK, bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"),
                zorder=5)


def fig_diagrama_flujo() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _caja_diagrama(ax, 0.23, 0.90, 0.30, 0.08, "Inicio de simulación\nt = 0,  I = I₀,  en_camino = 0",
                   facecolor="#EEF4F0", edgecolor=FOREST, bold=True, rounding=0.035)
    _caja_diagrama(ax, 0.23, 0.75, 0.36, 0.09, "1. Recepción al inicio del día t\nI ← I + arribos[t] ;  en_camino −= arribos[t]")
    _caja_diagrama(ax, 0.23, 0.59, 0.36, 0.10, "2. Ocurrencia de demanda D_t\nventas = min(I, D_t) ;  pérdidas = D_t − ventas\nI ← I − ventas")
    _caja_diagrama(ax, 0.23, 0.42, 0.36, 0.10, "3. Costeo diario\nC_mant = h·(I_ini + I_fin)/2\nC_falt = p·pérdidas")

    _rombo_diagrama(ax, 0.72, 0.72, 0.34, 0.16, "4. Revisión al cierre\n¿Política dispara pedido?")
    _caja_diagrama(ax, 0.72, 0.51, 0.38, 0.10, "Emitir orden de Q_t unidades\nMuestrear LT ;  arribo en t + LT + 1\nen_camino += Q_t ;  C_ord += K",
                   facecolor="#FAF5F5", edgecolor=BURGUNDY)
    _caja_diagrama(ax, 0.72, 0.33, 0.30, 0.08, "5. Avance de reloj\nt ← t + 1")
    _rombo_diagrama(ax, 0.72, 0.17, 0.32, 0.14, "¿t < 425?\n(60 warm-up + 365 reg.)")
    _caja_diagrama(ax, 0.23, 0.17, 0.34, 0.10, "Fin de réplica\nCalcular métricas sobre días 60–424",
                   facecolor="#EEF4F0", edgecolor=FOREST, bold=True, rounding=0.035)

    _flecha(ax, (0.23, 0.855), (0.23, 0.80))
    _flecha(ax, (0.23, 0.70), (0.23, 0.645))
    _flecha(ax, (0.23, 0.535), (0.23, 0.475))
    # Paso 3 -> Paso 4
    ax.plot([0.415, 0.48, 0.48, 0.545], [0.42, 0.42, 0.72, 0.72], color=INK, linewidth=1.2, zorder=2)
    _flecha(ax, (0.48, 0.72), (0.545, 0.72))

    _flecha(ax, (0.72, 0.635), (0.72, 0.565), texto="Sí", tx_off=(-0.03, 0))
    _flecha(ax, (0.72, 0.455), (0.72, 0.375))
    # Rama No del paso 4 directo al paso 5
    ax.plot([0.895, 0.95, 0.95, 0.875], [0.72, 0.72, 0.33, 0.33], color=INK, linewidth=1.2, zorder=2)
    _flecha(ax, (0.95, 0.33), (0.875, 0.33), texto="No", tx_off=(0.04, 0.20))

    _flecha(ax, (0.72, 0.285), (0.72, 0.245))
    _flecha(ax, (0.555, 0.17), (0.405, 0.17), texto="No", tx_off=(0, 0.03))

    # Bucle siguiente día: t < 425 (Sí) -> Paso 1
    ax.plot([0.72, 0.72, 0.45, 0.45, 0.415], [0.095, 0.04, 0.04, 0.75, 0.75], color=INK, linewidth=1.2, zorder=2)
    _flecha(ax, (0.45, 0.75), (0.415, 0.75))
    ax.text(0.585, 0.055, "Sí (siguiente día)", ha="center", va="center", fontsize=8.6,
            color=INK, bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))

    ax.set_title("Diagrama de flujo del motor de simulación por paso diario",
                 fontsize=12.5, fontweight="bold", color=INK, pad=6)
    _guardar(fig, "fig08_diagrama_flujo_sistema.png")


def fig_diagrama_politicas() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _caja_diagrama(ax, 0.50, 0.91, 0.28, 0.075, "Cierre del día t\nIP = I_fin + en_camino",
                   facecolor="#EEF4F0", edgecolor=FOREST, bold=True, rounding=0.035)
    _rombo_diagrama(ax, 0.50, 0.74, 0.28, 0.13, "¿Tipo de política?")

    _rombo_diagrama(ax, 0.24, 0.53, 0.26, 0.13, "¿IP ≤ s ?", facecolor="#F8FAFC", edgecolor=SLATE)
    _rombo_diagrama(ax, 0.76, 0.58, 0.28, 0.13, "¿(t + 1) mod R = 0?", facecolor="#F8FAFC", edgecolor=SLATE)
    _rombo_diagrama(ax, 0.84, 0.395, 0.24, 0.115, "¿S − IP > 0?", facecolor="#F8FAFC", edgecolor=SLATE)

    _caja_diagrama(ax, 0.14, 0.31, 0.20, 0.08, "Pedido = Q", facecolor="#FAF5F5", edgecolor=BURGUNDY, bold=True)
    _caja_diagrama(ax, 0.50, 0.31, 0.22, 0.08, "Pedido = 0\n(no se ordena)", facecolor="#F9FAFB", edgecolor=SLATE)
    _caja_diagrama(ax, 0.84, 0.23, 0.22, 0.08, "Pedido = S − IP", facecolor="#FAF5F5", edgecolor=BURGUNDY, bold=True)

    _caja_diagrama(
        ax, 0.50, 0.07, 0.72, 0.095,
        "Si Pedido > 0: muestrear LT ~ U{2, 3, 4, 5},\nprogramar arribo en t + LT + 1 y sumar costo K",
        facecolor="#F3F4F6", edgecolor=INK, bold=True,
    )

    _flecha(ax, (0.50, 0.87), (0.50, 0.81))
    _flecha(ax, (0.355, 0.74), (0.24, 0.60), texto="Continua (s, Q) · E1–E3", tx_off=(-0.065, 0.04))
    _flecha(ax, (0.645, 0.74), (0.76, 0.65), texto="Periódica (R, S) · E4", tx_off=(0.075, 0.04))

    _flecha(ax, (0.17, 0.47), (0.14, 0.355), texto="Sí", tx_off=(-0.025, 0.01))
    _flecha(ax, (0.31, 0.47), (0.43, 0.355), texto="No", tx_off=(0.015, 0.02))
    _flecha(ax, (0.67, 0.525), (0.53, 0.355), texto="No", tx_off=(-0.025, 0.01))
    _flecha(ax, (0.83, 0.525), (0.84, 0.455), texto="Sí", tx_off=(0.025, 0.01))
    _flecha(ax, (0.72, 0.395), (0.615, 0.33), texto="No", tx_off=(0, 0.025))
    _flecha(ax, (0.84, 0.335), (0.84, 0.275), texto="Sí", tx_off=(0.025, 0.0))

    _flecha(ax, (0.18, 0.265), (0.24, 0.13))
    _flecha(ax, (0.50, 0.265), (0.50, 0.13))
    _flecha(ax, (0.81, 0.185), (0.76, 0.13))

    ax.set_title("Lógica comparada de decisión: revisión continua (s, Q) frente a periódica (R, S)",
                 fontsize=12.5, fontweight="bold", color=INK, pad=6)
    _guardar(fig, "fig09_diagrama_logica_politicas.png")


def fig_sensibilidad_penalizacion() -> None:
    df = _cargar("sensibilidad_penalizacion.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.1), gridspec_kw={"wspace": 0.44})

    p_vals = df["p_falta"].to_numpy()
    ax1.plot(p_vals, df["s_opt"], "o-", color=BURGUNDY, linewidth=2.0, markersize=7,
             label="Punto de reorden s*")
    ax1_r = ax1.twinx()
    ax1_r.plot(p_vals, df["Q_opt"], "s--", color=SLATE, linewidth=1.8, markersize=6.5,
               label="Lote óptimo Q*")
    ax1_r.spines["top"].set_visible(False)
    ax1_r.set_ylabel("Tamaño de lote Q* (unidades)", color=SLATE, labelpad=8)
    ax1_r.tick_params(axis="y", colors=SLATE)
    q_min, q_max = int(df["Q_opt"].min()), int(df["Q_opt"].max())
    ax1_r.set_ylim(min(150, q_min - 35), max(255, q_max + 35))

    s_min, s_max = int(df["s_opt"].min()), int(df["s_opt"].max())
    s_lo, s_hi = max(25, s_min - 9), s_max + 7
    ax1.axvline(PARAMETROS.p_falta, color=OCHRE, linestyle=":", linewidth=1.45, zorder=2)
    ax1.text(
        PARAMETROS.p_falta + 1.5, s_lo + 1.2, "Caso base\np = 30 USD",
        fontsize=8.6, color="#8F5718", fontweight="bold", va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.92),
    )
    ax1.set_xlabel("Penalización por unidad faltante p (USD/unidad)")
    ax1.set_ylabel("Punto de reorden s* (unidades)", color=BURGUNDY)
    ax1.tick_params(axis="y", colors=BURGUNDY)
    ax1.set_ylim(s_lo, s_hi)
    ax1.set_title("Sensibilidad de parámetros óptimos (s*, Q*)", loc="left", pad=26)
    _limpiar(ax1)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(
        h1 + h2, l1 + l2, loc="lower left", bbox_to_anchor=(0.0, 1.01, 1.0, 0.12),
        ncols=2, frameon=False, fontsize=8.6,
    )

    x = np.arange(len(df))
    base = np.zeros(len(df))
    for col, etiq, col_hex in [
        ("mant_medio", "Mantenimiento", SLATE),
        ("ord_medio", "Ordenamiento", PEWTER),
        ("falt_medio", "Faltantes", CRIMSON),
    ]:
        vals = df[col].to_numpy()
        ax2.bar(x, vals, bottom=base, width=0.56, color=col_hex, label=etiq, edgecolor="white", linewidth=0.6)
        base += vals
    for xi, tot in enumerate(base):
        ax2.text(xi, tot + 35, _fmt_es(tot, 0), ha="center", va="bottom",
                 fontsize=8.8, fontweight="bold", color=INK)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{int(v)}" for v in p_vals])
    ax2.set_xlabel("Penalización por unidad faltante p (USD/unidad)")
    ax2.set_ylabel("Costo anual mínimo medio (USD/año)")
    ax2.yaxis.set_major_formatter(_formatter_es(0))
    ax2.set_ylim(0, base.max() * 1.18)
    ax2.set_title("Descomposición del costo óptimo frente a p", loc="left", pad=26)
    ax2.legend(
        loc="lower left", bbox_to_anchor=(0.0, 1.01, 1.0, 0.12),
        ncols=3, frameon=False, fontsize=8.6, columnspacing=1.2,
    )
    _limpiar(ax2)

    _guardar(fig, "fig10_sensibilidad_penalizacion.png")


def fig_sensibilidad_K_LT() -> None:
    df_k = _cargar("sensibilidad_K.csv")
    df_lt = _cargar("sensibilidad_LT.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.1), gridspec_kw={"wspace": 0.36})

    k_vals = df_k["K"].to_numpy()
    ax1.plot(k_vals, df_k["Q_opt"], "s-", color=FOREST, linewidth=2.1, markersize=7, label="Lote óptimo simulado Q*")
    ax1.plot(k_vals, df_k["eoq_analitico"], "o--", color=SLATE, linewidth=1.8, markersize=6.5, label="EOQ analítico de Wilson")
    bbox_k = dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.88)
    for i, (k, q_val, eoq_val, gap) in enumerate(zip(k_vals, df_k["Q_opt"], df_k["eoq_analitico"], df_k["brecha_RS_vs_sQ"])):
        y_top = max(float(q_val), float(eoq_val))
        ox = -8 if i < 2 else 0
        ax1.annotate(
            f"+{_fmt_es(gap, 0)} USD\nen (R, S)",
            xy=(k, y_top), xytext=(ox, 13), textcoords="offset points",
            ha="center", va="bottom", fontsize=7.8, color=PLUM, fontweight="bold", bbox=bbox_k,
        )
    ax1.axvline(PARAMETROS.k_orden, color=OCHRE, linestyle=":", linewidth=1.35)
    ax1.set_xlabel("Costo fijo por orden K (USD/orden)")
    ax1.set_ylabel("Tamaño de lote (unidades)")
    ax1.set_xlim(10, 168)
    ax1.set_ylim(95, max(df_k["Q_opt"].max(), df_k["eoq_analitico"].max()) + 65)
    ax1.set_title("Respuesta de Q* y brecha (R, S) frente a K", loc="left", pad=26)
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01, 1.0, 0.12), ncols=2, frameon=False, fontsize=8.4)
    _limpiar(ax1)

    sig_lt = df_lt["sigma_lt_obs"].to_numpy()
    ax2.plot(sig_lt, df_lt["s_opt"], "o-", color=BURGUNDY, linewidth=2.1, markersize=7, label="Punto de reorden óptimo s*")
    ax2_r = ax2.twinx()
    ax2_r.plot(sig_lt, df_lt["costo_medio"], "^--", color=SLATE, linewidth=1.8, markersize=7, label="Costo óptimo (USD/año)")
    ax2_r.spines["top"].set_visible(False)
    ax2_r.set_ylabel("Costo anual mínimo medio (USD/año)", color=SLATE, labelpad=8)
    ax2_r.tick_params(axis="y", colors=SLATE)
    ax2_r.yaxis.set_major_formatter(_formatter_es(0))
    ax2_r.set_ylim(df_lt["costo_medio"].min() - 95, df_lt["costo_medio"].max() + 140)

    etiquetas_lt = ["U{3,4}", "U{2..5}\n(base)", "U{1..6}", "Gamma\ndiscreta"]
    bbox_lt = dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.88)
    for i, (sx, sy, etiq) in enumerate(zip(sig_lt, df_lt["s_opt"], etiquetas_lt)):
        ox = -14 if i == 3 else (10 if i == 0 else 0)
        ax2.annotate(
            etiq, xy=(sx, sy), xytext=(ox, 11), textcoords="offset points",
            ha="center", va="bottom", fontsize=7.9, color=INK, fontweight="bold", bbox=bbox_lt,
        )
    ax2.set_xlabel("Desvío estándar observado del lead time $\\sigma_{LT}$ (días)")
    ax2.set_ylabel("Punto de reorden óptimo s* (unidades)", color=BURGUNDY)
    ax2.tick_params(axis="y", colors=BURGUNDY)
    ax2.set_ylim(df_lt["s_opt"].min() - 14, df_lt["s_opt"].max() + 14)
    ax2.set_title("Impacto de la variabilidad del lead time ($\\mu_{LT} \\approx 3{,}5$ d)", loc="left", pad=26)
    _limpiar(ax2)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower left", bbox_to_anchor=(0.0, 1.01, 1.0, 0.12), ncols=2, frameon=False, fontsize=8.3)

    _guardar(fig, "fig11_sensibilidad_K_LT.png")


def main() -> None:
    _aplicar_estilo()
    print("Generando figuras académicas (300 DPI) en results/img/ ...")
    fig_distribuciones()
    fig_warmup()
    fig_traza_inventario()
    fig_costos_composicion()
    fig_boxplots()
    fig_ic()
    fig_heatmap_sQ()
    fig_diagrama_flujo()
    fig_diagrama_politicas()
    fig_sensibilidad_penalizacion()
    fig_sensibilidad_K_LT()
    print("Figuras completadas.")


if __name__ == "__main__":
    main()
