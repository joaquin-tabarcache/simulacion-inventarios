"""Diseño experimental, optimización por simulación, análisis estadístico y sensibilidad."""

from __future__ import annotations

import copy
from dataclasses import replace
import json
import time

import numpy as np
import pandas as pd
from scipy import stats

from .config import (
    DIR_RESULTADOS,
    ESCENARIOS,
    GRIDS,
    K_SENSIBILIDAD,
    N_REPLICAS,
    N_REPLICAS_BUSQUEDA,
    N_REPLICAS_REFINAMIENTO,
    PARAMETROS,
    PENALIZACIONES_SENSIBILIDAD,
    WARMUP_VERIFICACION,
    Parametros,
)
from .simulator import METRICAS, calcular_metricas, caminos_de_semilla, simular
from .validation import traza_a_dataframe


def ic95(x) -> dict[str, float]:
    """Estadísticos descriptivos e IC del 95% mediante distribución t de Student."""
    raw = np.asarray(x, dtype=float)
    arr = raw[np.isfinite(raw)]
    n = len(arr)
    if n == 0:
        return {
            "n": 0, "media": float("nan"), "desv_std": float("nan"),
            "minimo": float("nan"), "maximo": float("nan"),
            "ic95_inf": float("nan"), "ic95_sup": float("nan"),
        }
    media = float(arr.mean())
    if n == 1:
        return {
            "n": 1, "media": media, "desv_std": 0.0,
            "minimo": media, "maximo": media, "ic95_inf": media, "ic95_sup": media,
        }
    desv = float(arr.std(ddof=1))
    semi_ancho = float(stats.t.ppf(0.975, n - 1) * desv / np.sqrt(n))
    return {
        "n": n,
        "media": media,
        "desv_std": desv,
        "minimo": float(arr.min()),
        "maximo": float(arr.max()),
        "ic95_inf": media - semi_ancho,
        "ic95_sup": media + semi_ancho,
    }


def semillas(n: int, base: int) -> list[np.random.SeedSequence]:
    return np.random.SeedSequence(base).spawn(n)


def _cfg_politica(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k in ("s", "Q", "R", "S", "inv_inicial") and v is not None}


def preparar_caminos(
    lista_semillas: list[np.random.SeedSequence],
    p: Parametros = PARAMETROS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pre-genera los pares (demanda, lead times) compartidos bajo CRN."""
    return [caminos_de_semilla(ss, p.dias_sim, p=p) for ss in lista_semillas]


def _asegurar_caminos(
    origen: list[np.random.SeedSequence] | list[tuple[np.ndarray, np.ndarray]],
    p: Parametros = PARAMETROS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if origen and isinstance(origen[0], tuple):
        return origen  # type: ignore[return-value]
    return preparar_caminos(origen, p=p)  # type: ignore[arg-type]


def _evaluar_caminos(
    caminos: list[tuple[np.ndarray, np.ndarray]],
    politica: str,
    params_pol: dict,
    p: Parametros = PARAMETROS,
    p_falta: float | None = None,
    k_orden: float | None = None,
) -> list[dict[str, float]]:
    inv_ini = params_pol.get("inv_inicial", params_pol.get("Q", 0))
    args_pol = {k: v for k, v in params_pol.items() if k != "inv_inicial"}
    return [
        calcular_metricas(
            simular(
                dem, lts, politica=politica, inventario_inicial=inv_ini,
                K=k_orden, p_falta=p_falta, p=p, **args_pol,
            ),
            p.warmup_dias,
            p.dias_sim,
            fr_objetivo=p.fill_rate_objetivo,
        )
        for dem, lts in caminos
    ]


def buscar_sQ(
    semillas_busq: list[np.random.SeedSequence] | list[tuple[np.ndarray, np.ndarray]],
    p: Parametros = PARAMETROS,
    p_falta: float | None = None,
    k_orden: float | None = None,
    s_valores: list[int] | None = None,
    Q_valores: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Búsqueda en grilla (s, Q) con CRN sujeta a restricción de Fill Rate."""
    caminos = _asegurar_caminos(semillas_busq, p=p)
    s_list = GRIDS["s_valores"] if s_valores is None else s_valores
    q_list = GRIDS["Q_valores"] if Q_valores is None else Q_valores
    filas = []
    for s in s_list:
        for q in q_list:
            mets = _evaluar_caminos(
                caminos, "sQ", dict(s=s, Q=q, inv_inicial=q), p=p, p_falta=p_falta, k_orden=k_orden,
            )
            df = pd.DataFrame(mets)
            cost_ic = ic95(df["costo_total"])
            filas.append({
                "s": s,
                "Q": q,
                "costo_medio": cost_ic["media"],
                "costo_std": cost_ic["desv_std"],
                "costo_ic95_half": cost_ic["ic95_sup"] - cost_ic["media"],
                "fill_rate_medio": float(df["fill_rate_pct"].mean()),
                "csl_medio": float(df["csl_pct"].mean()),
                "dias_quiebre_medio": float(df["dias_quiebre"].mean()),
                "mant_medio": float(df["costo_mantenimiento"].mean()),
                "ord_medio": float(df["costo_ordenamiento"].mean()),
                "falt_medio": float(df["costo_faltante"].mean()),
                "unidades_perdidas_medio": float(df["unidades_perdidas"].mean()),
            })
    df_grid = pd.DataFrame(filas)
    factibles = df_grid[df_grid["fill_rate_medio"] >= p.fill_rate_objetivo]
    mejor = factibles.loc[factibles["costo_medio"].idxmin()]
    return df_grid, mejor


def buscar_RS(
    semillas_busq: list[np.random.SeedSequence] | list[tuple[np.ndarray, np.ndarray]],
    p: Parametros = PARAMETROS,
    k_orden: float | None = None,
    r_valores: list[int] | None = None,
    s_margenes: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Búsqueda en grilla (R, S) con CRN sujeta a restricción de Fill Rate."""
    caminos = _asegurar_caminos(semillas_busq, p=p)
    r_list = GRIDS["R_valores"] if r_valores is None else r_valores
    marg_list = GRIDS["S_margenes"] if s_margenes is None else s_margenes
    filas = []
    for r in r_list:
        s_base = p.s_analitico_rs(r)
        for margen in marg_list:
            s_obj = s_base + margen
            mets = _evaluar_caminos(
                caminos, "RS", dict(R=r, S=s_obj, inv_inicial=s_obj), p=p, k_orden=k_orden,
            )
            df = pd.DataFrame(mets)
            cost_ic = ic95(df["costo_total"])
            filas.append({
                "R": r,
                "S": s_obj,
                "S_analitico": s_base,
                "margen_sobre_analitico": margen,
                "costo_medio": cost_ic["media"],
                "costo_std": cost_ic["desv_std"],
                "costo_ic95_half": cost_ic["ic95_sup"] - cost_ic["media"],
                "fill_rate_medio": float(df["fill_rate_pct"].mean()),
                "csl_medio": float(df["csl_pct"].mean()),
                "dias_quiebre_medio": float(df["dias_quiebre"].mean()),
                "mant_medio": float(df["costo_mantenimiento"].mean()),
                "ord_medio": float(df["costo_ordenamiento"].mean()),
                "falt_medio": float(df["costo_faltante"].mean()),
            })
    df_grid = pd.DataFrame(filas)
    factibles = df_grid[df_grid["fill_rate_medio"] >= p.fill_rate_objetivo]
    mejor = factibles.loc[factibles["costo_medio"].idxmin()]
    return df_grid, mejor


def sensibilidad_penalizacion(
    semillas_busq: list[np.random.SeedSequence] | list[tuple[np.ndarray, np.ndarray]],
    p: Parametros = PARAMETROS,
    df_sQ_base: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Evalúa cómo cambian (s*, Q*) y costos según p sobre la grilla fina (paso 1 en s, 10 en Q)."""
    if df_sQ_base is None:
        df_sQ_base, _ = buscar_sQ(
            semillas_busq, p=p, s_valores=GRIDS["s_fino"], Q_valores=GRIDS["Q_fino"],
        )
    factibles = df_sQ_base[df_sQ_base["fill_rate_medio"] >= p.fill_rate_objetivo].copy()
    u_perd = (
        factibles["unidades_perdidas_medio"]
        if "unidades_perdidas_medio" in factibles
        else factibles["falt_medio"] / p.p_falta
    )
    filas = []
    for p_val in PENALIZACIONES_SENSIBILIDAD:
        falt_val = p_val * u_perd
        costo_val = factibles["mant_medio"] + factibles["ord_medio"] + falt_val
        idx = costo_val.idxmin()
        mejor = factibles.loc[idx]
        filas.append({
            "p_falta": p_val,
            "s_opt": int(mejor["s"]),
            "Q_opt": int(mejor["Q"]),
            "costo_medio": float(costo_val.loc[idx]),
            "fill_rate_medio": float(mejor["fill_rate_medio"]),
            "csl_medio": float(mejor["csl_medio"]),
            "mant_medio": float(mejor["mant_medio"]),
            "ord_medio": float(mejor["ord_medio"]),
            "falt_medio": float(falt_val.loc[idx]),
        })
    return pd.DataFrame(filas)


def sensibilidad_K(
    caminos_busq: list[tuple[np.ndarray, np.ndarray]],
    p: Parametros = PARAMETROS,
) -> pd.DataFrame:
    """Evalúa cómo cambian Q*, EOQ y la brecha (R, S) vs (s, Q) al variar el costo fijo K."""
    filas = []
    s_grid = list(range(42, 62, 2))
    q_grid = list(range(100, 420, 20))
    r_grid = [7, 14, 21, 24, 28, 35, 42, 49, 56]
    marg_grid = list(range(-16, 28, 4))
    for k_val in K_SENSIBILIDAD:
        pk = replace(p, k_orden=k_val)
        _, best_sq = buscar_sQ(caminos_busq, p=pk, k_orden=k_val, s_valores=s_grid, Q_valores=q_grid)
        _, best_rs = buscar_RS(caminos_busq, p=pk, k_orden=k_val, r_valores=r_grid, s_margenes=marg_grid)
        filas.append({
            "K": float(k_val),
            "eoq_analitico": round(pk.eoq, 1),
            "s_opt": int(best_sq["s"]),
            "Q_opt": int(best_sq["Q"]),
            "costo_sQ": float(best_sq["costo_medio"]),
            "fr_sQ": float(best_sq["fill_rate_medio"]),
            "R_opt": int(best_rs["R"]),
            "S_opt": int(best_rs["S"]),
            "costo_RS": float(best_rs["costo_medio"]),
            "brecha_RS_vs_sQ": float(best_rs["costo_medio"] - best_sq["costo_medio"]),
        })
    return pd.DataFrame(filas)


def sensibilidad_lt(n_reps: int = N_REPLICAS_REFINAMIENTO, p: Parametros = PARAMETROS) -> pd.DataFrame:
    """Evalúa sensibilidad de s* y costo óptimo al aumentar la dispersión del lead time con media ~3,5 d."""
    regimenes = [
        ("Uniforme {3, 4} (baja)", dict(dist_lt="uniforme_discreta", lt_a=3, lt_b=4)),
        ("Uniforme {2, 3, 4, 5} (base)", dict(dist_lt="uniforme_discreta", lt_a=2, lt_b=5)),
        ("Uniforme {1..6} (alta)", dict(dist_lt="uniforme_discreta", lt_a=1, lt_b=6)),
        ("Gamma discreta (cola pesada)", dict(dist_lt="gamma_discreta", gamma_shape=2.0, gamma_scale=1.5)),
    ]
    filas = []
    s_grid = list(range(36, 106, 2))
    q_grid = list(range(180, 250, 20))
    for etiqueta, overrides in regimenes:
        pr = replace(p, **overrides)
        caminos = preparar_caminos(semillas(n_reps, p.seed_busqueda), p=pr)
        lts_todos = np.concatenate([lt for _, lt in caminos])
        _, best = buscar_sQ(caminos, p=pr, s_valores=s_grid, Q_valores=q_grid)
        filas.append({
            "regimen_lt": etiqueta,
            "mu_lt_obs": float(lts_todos.mean()),
            "sigma_lt_obs": float(lts_todos.std(ddof=1)),
            "s_opt": int(best["s"]),
            "Q_opt": int(best["Q"]),
            "costo_medio": float(best["costo_medio"]),
            "fill_rate_medio": float(best["fill_rate_medio"]),
            "csl_medio": float(best["csl_medio"]),
        })
    return pd.DataFrame(filas)


def correr_escenario(
    nombre: str,
    cfg: dict,
    lista_semillas: list[np.random.SeedSequence] | list[tuple[np.ndarray, np.ndarray]],
    p: Parametros = PARAMETROS,
) -> pd.DataFrame:
    caminos = _asegurar_caminos(lista_semillas, p=p)
    mets = _evaluar_caminos(caminos, cfg["politica"], _cfg_politica(cfg), p=p)
    for i, m in enumerate(mets):
        m["escenario"] = nombre
        m["replica"] = i
    return pd.DataFrame(mets)


def pruebas_hipotesis(replicas: dict[str, pd.DataFrame], p: Parametros = PARAMETROS) -> pd.DataFrame:
    """Pruebas t pareadas, Wilcoxon no paramétrico e IC 95% sobre diferencias pareadas."""
    filas = []

    def pareada(a: str, b: str, var: str = "costo_total") -> None:
        xa = replicas[a][var].to_numpy(float)
        xb = replicas[b][var].to_numpy(float)
        t_stat, pv_t = stats.ttest_rel(xa, xb)
        _, pv_w = stats.wilcoxon(xa, xb)
        diff = xa - xb
        ic_diff = ic95(diff)
        filas.append({
            "comparacion": f"{a} vs {b}",
            "variable": var,
            "tipo": "t pareada (CRN)",
            "media_a": float(xa.mean()),
            "media_b": float(xb.mean()),
            "diff_media": float(diff.mean()),
            "ic95_diff_inf": ic_diff["ic95_inf"],
            "ic95_diff_sup": ic_diff["ic95_sup"],
            "t_stat": float(t_stat),
            "gl": len(xa) - 1,
            "p_valor": float(pv_t),
            "p_wilcoxon": float(pv_w),
            "cohens_d_z": float(diff.mean() / diff.std(ddof=1)),
            "significativo_05": bool(pv_t < 0.05),
        })

    pareada("E1_Base", "E3_Optimizado")
    pareada("E2_Conservador", "E3_Optimizado")
    pareada("E4_Periodico", "E3_Optimizado")
    pareada("E1_Base", "E2_Conservador")
    return pd.DataFrame(filas)


def correlaciones_crn(replicas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Evalúa correlación de Pearson entre réplicas y reducción efectiva de varianza bajo CRN."""
    pares = [
        ("E2_Conservador", "E3_Optimizado"),
        ("E1_Base", "E3_Optimizado"),
        ("E4_Periodico", "E3_Optimizado"),
        ("E1_Base", "E4_Periodico"),
        ("E1_Base", "E2_Conservador"),
    ]
    filas = []
    for a, b in pares:
        xa = replicas[a]["costo_total"].to_numpy(float)
        xb = replicas[b]["costo_total"].to_numpy(float)
        r_corr, pv_r = stats.pearsonr(xa, xb)
        sd_a = float(xa.std(ddof=1))
        sd_b = float(xb.std(ddof=1))
        var_ind = float(xa.var(ddof=1) + xb.var(ddof=1))
        var_crn = float((xa - xb).var(ddof=1))
        filas.append({
            "par": f"{a} vs {b}",
            "n": len(xa),
            "r_pearson": float(r_corr),
            "p_valor": float(pv_r),
            "sd_a": sd_a,
            "sd_b": sd_b,
            "sd_ind": float(np.sqrt(var_ind)),
            "sd_crn": float(np.sqrt(var_crn)),
            "reduc_var_pct": float(100.0 * (1.0 - var_crn / var_ind)),
            "significativo_05": bool(pv_r < 0.05),
        })
    return pd.DataFrame(filas)


def auditoria_estacionario(
    p: Parametros = PARAMETROS,
    s_opt: int = 53,
    q_opt: int = 220,
) -> pd.DataFrame:
    """Verifica estabilidad estacionaria ante warmup creciente y balance exacto de órdenes."""
    sem_main = semillas(N_REPLICAS, p.seed_principal)
    filas = []
    for w in WARMUP_VERIFICACION:
        pw = replace(p, warmup_dias=w)
        caminos = preparar_caminos(sem_main, p=pw)
        costos, invs, frs, ords_sim, ords_bal = [], [], [], [], []
        for dem, lts in caminos:
            traza = simular(dem, lts, politica="sQ", s=s_opt, Q=q_opt, inventario_inicial=q_opt, p=pw)
            m = calcular_metricas(traza, pw.warmup_dias, pw.dias_sim, fr_objetivo=pw.fill_rate_objetivo)
            costos.append(m["costo_total"])
            invs.append(m["inventario_promedio"])
            frs.append(m["fill_rate_pct"])
            ords_sim.append(m["num_ordenes"])
            ip_ini_w = int(traza["ip_cierre"][w - 1]) if w > 0 else int(q_opt)
            ip_fin_w = int(traza["ip_cierre"][pw.dias_sim - 1])
            ventas_w = int(traza["ventas"][w:pw.dias_sim].sum())
            ords_bal.append((ventas_w + ip_fin_w - ip_ini_w) / q_opt)
        ic_c = ic95(costos)
        ic_i = ic95(invs)
        filas.append({
            "warmup_dias": w,
            "costo_medio": ic_c["media"],
            "costo_ic95_half": ic_c["ic95_sup"] - ic_c["media"],
            "inv_promedio": ic_i["media"],
            "inv_ic95_half": ic_i["ic95_sup"] - ic_i["media"],
            "inv_aprox_analitico": float(q_opt / 2.0 + (s_opt - pw.lambda_demanda * pw.mu_lt)),
            "fill_rate_medio": float(np.mean(frs)),
            "num_ordenes_sim": float(np.mean(ords_sim)),
            "num_ordenes_balance": float(np.mean(ords_bal)),
            "discrepancia_balance": float(np.max(np.abs(np.asarray(ords_sim) - np.asarray(ords_bal)))),
        })
    return pd.DataFrame(filas)


def analisis_warmup(
    lista_semillas: list[np.random.SeedSequence] | list[tuple[np.ndarray, np.ndarray]],
    p: Parametros = PARAMETROS,
) -> pd.DataFrame:
    """Evalúa el transitorio de costo diario promedio bajo tres condiciones iniciales."""
    caminos = _asegurar_caminos(lista_semillas, p=p)
    s0 = p.s_sin_ss
    q0 = int(p.eoq)
    filas = []
    for inv0 in (0, s0, q0):
        acum = np.zeros(p.dias_sim)
        for demanda, lts in caminos:
            traza = simular(demanda, lts, politica="sQ", s=s0, Q=q0, inventario_inicial=inv0, p=p)
            acum += traza["c_mant"] + traza["c_ord"] + traza["c_falt"]
        prom = acum / len(caminos)
        for dia, val in enumerate(prom):
            filas.append({"inicial": f"I0={inv0}", "dia": dia, "costo_diario_prom": float(val)})
    return pd.DataFrame(filas)


def main() -> None:
    t0 = time.perf_counter()
    p = PARAMETROS
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

    caminos_main = preparar_caminos(semillas(N_REPLICAS, p.seed_principal), p=p)
    caminos_busq = preparar_caminos(semillas(N_REPLICAS_BUSQUEDA, p.seed_busqueda), p=p)
    caminos_ref = preparar_caminos(semillas(N_REPLICAS_REFINAMIENTO, p.seed_busqueda), p=p)

    print("=" * 76)
    print("CAMPAÑA EXPERIMENTAL — ETAPA 1 GRUESA + ETAPA 2 FINA + CONFIRMACIÓN (CRN)")
    print("=" * 76)

    df_sQ, mejor_sQ = buscar_sQ(caminos_busq, p=p)
    df_sQ.to_csv(DIR_RESULTADOS / "grid_sQ.csv", index=False)
    print(
        f"Etapa 1 (s, Q) [{len(df_sQ)} comb., {N_REPLICAS_BUSQUEDA} rep.]: "
        f"s={int(mejor_sQ['s'])}, Q={int(mejor_sQ['Q'])} -> {mejor_sQ['costo_medio']:.2f} USD"
    )

    df_sQ_fino, mejor_sQ_fino = buscar_sQ(
        caminos_ref, p=p, s_valores=GRIDS["s_fino"], Q_valores=GRIDS["Q_fino"],
    )
    s_opt, q_opt = int(mejor_sQ_fino["s"]), int(mejor_sQ_fino["Q"])
    df_sQ_fino.to_csv(DIR_RESULTADOS / "grid_sQ_fino.csv", index=False)
    print(
        f"Etapa 2 (s, Q) [{len(df_sQ_fino)} comb., {N_REPLICAS_REFINAMIENTO} rep.]: "
        f"s*={s_opt}, Q*={q_opt} -> {mejor_sQ_fino['costo_medio']:.2f} USD"
    )

    df_RS, mejor_RS = buscar_RS(caminos_busq, p=p)
    df_RS.to_csv(DIR_RESULTADOS / "grid_RS.csv", index=False)
    print(
        f"Etapa 1 (R, S) [{len(df_RS)} comb., {N_REPLICAS_BUSQUEDA} rep.]: "
        f"R={int(mejor_RS['R'])}, S={int(mejor_RS['S'])} -> {mejor_RS['costo_medio']:.2f} USD"
    )

    df_RS_fino, mejor_RS_fino = buscar_RS(
        caminos_ref, p=p, r_valores=GRIDS["R_fino"], s_margenes=GRIDS["S_margenes_fino"],
    )
    r_opt, s_rs_opt = int(mejor_RS_fino["R"]), int(mejor_RS_fino["S"])
    df_RS_fino.to_csv(DIR_RESULTADOS / "grid_RS_fino.csv", index=False)
    print(
        f"Etapa 2 (R, S) [{len(df_RS_fino)} comb., {N_REPLICAS_REFINAMIENTO} rep.]: "
        f"R*={r_opt}, S*={s_rs_opt} -> {mejor_RS_fino['costo_medio']:.2f} USD"
    )

    df_rs28 = df_RS_fino[(df_RS_fino["R"] == 28) & (df_RS_fino["fill_rate_medio"] >= p.fill_rate_objetivo)]
    mejor_rs28 = df_rs28.loc[df_rs28["costo_medio"].idxmin()]

    escenarios = copy.deepcopy(ESCENARIOS)
    escenarios["E3_Optimizado"].update(s=s_opt, Q=q_opt, inv_inicial=q_opt)
    escenarios["E4_Periodico"].update(R=r_opt, S=s_rs_opt, inv_inicial=s_rs_opt)

    replicas: dict[str, pd.DataFrame] = {}
    for nombre, cfg in escenarios.items():
        replicas[nombre] = correr_escenario(nombre, cfg, caminos_main, p=p)
        df = replicas[nombre]
        print(
            f"  {nombre:16s} costo={df['costo_total'].mean():8.2f} "
            f"FR={df['fill_rate_pct'].mean():6.2f}% CSL={df['csl_pct'].mean():6.2f}%"
        )

    pd.concat(replicas.values(), ignore_index=True).to_csv(
        DIR_RESULTADOS / "replicas_metricas.csv", index=False,
    )

    filas_res = []
    for nombre, cfg in escenarios.items():
        prm = _cfg_politica(cfg)
        etiqueta = ", ".join(f"{k}={v}" for k, v in prm.items() if k != "inv_inicial")
        for met in METRICAS:
            filas_res.append({
                "escenario": nombre,
                "politica": cfg["politica"],
                "parametros": etiqueta,
                "metrica": met,
                **ic95(replicas[nombre][met]),
            })
    pd.DataFrame(filas_res).to_csv(DIR_RESULTADOS / "resumen_escenarios.csv", index=False)

    df_tests = pruebas_hipotesis(replicas, p=p)
    df_tests.to_csv(DIR_RESULTADOS / "pruebas_hipotesis.csv", index=False)

    df_crn = correlaciones_crn(replicas)
    df_crn.to_csv(DIR_RESULTADOS / "correlaciones_crn.csv", index=False)

    df_warm = analisis_warmup(caminos_busq, p=p)
    df_warm.to_csv(DIR_RESULTADOS / "warmup_costos.csv", index=False)

    df_audit = auditoria_estacionario(p=p, s_opt=s_opt, q_opt=q_opt)
    df_audit.to_csv(DIR_RESULTADOS / "auditoria_estacionario.csv", index=False)

    df_sens = sensibilidad_penalizacion(caminos_ref, p=p, df_sQ_base=df_sQ_fino)
    df_sens.to_csv(DIR_RESULTADOS / "sensibilidad_penalizacion.csv", index=False)

    df_sens_k = sensibilidad_K(caminos_ref, p=p)
    df_sens_k.to_csv(DIR_RESULTADOS / "sensibilidad_K.csv", index=False)

    df_sens_lt = sensibilidad_lt(n_reps=N_REPLICAS_REFINAMIENTO, p=p)
    df_sens_lt.to_csv(DIR_RESULTADOS / "sensibilidad_LT.csv", index=False)

    demanda, lts = caminos_main[0]
    cfg2 = escenarios["E2_Conservador"]
    traza_ej = simular(
        demanda, lts, politica=cfg2["politica"], inventario_inicial=cfg2["inv_inicial"],
        **{k: v for k, v in _cfg_politica(cfg2).items() if k != "inv_inicial"},
    )
    traza_a_dataframe(traza_ej, p.dias_sim).to_csv(DIR_RESULTADOS / "traza_ejemplo.csv", index=False)

    fact_fino = df_sQ_fino[df_sQ_fino["fill_rate_medio"] >= p.fill_rate_objetivo]
    top5_sQ = [
        {
            "s": int(row["s"]),
            "Q": int(row["Q"]),
            "costo_medio": round(float(row["costo_medio"]), 2),
            "fill_rate_medio": round(float(row["fill_rate_medio"]), 2),
        }
        for _, row in fact_fino.nsmallest(5, "costo_medio").iterrows()
    ]

    optimos = {
        "s_opt": s_opt,
        "Q_opt": q_opt,
        "s_etapa1": int(mejor_sQ["s"]),
        "Q_etapa1": int(mejor_sQ["Q"]),
        "R_opt": r_opt,
        "S_opt": s_rs_opt,
        "margen_RS_opt": int(mejor_RS_fino["margen_sobre_analitico"]),
        "R_28_opt": {"R": 28, "S": int(mejor_rs28["S"]), "margen": int(mejor_rs28["margen_sobre_analitico"]), "costo_medio": round(float(mejor_rs28["costo_medio"]), 2)},
        "top5_sQ_fino": top5_sQ,
        "S_analitico_R": {int(r): int(p.s_analitico_rs(int(r))) for r in GRIDS["R_valores"]},
        "grid": dict(GRIDS),
        "n_replicas_busqueda": N_REPLICAS_BUSQUEDA,
        "n_replicas_refinamiento": N_REPLICAS_REFINAMIENTO,
        "n_replicas_final": N_REPLICAS,
        "costo_analitico_eoq": p.costo_total_eoq(),
        "eoq": p.eoq,
        "s_sin_ss": p.s_sin_ss,
        "s_con_ss": p.s_con_ss,
        "sigma_dl": p.sigma_dl,
        "ss_analitico": p.ss_analitico,
        "segundos_ejecucion": round(time.perf_counter() - t0, 2),
    }
    with open(DIR_RESULTADOS / "optimos.json", "w", encoding="utf-8") as f:
        json.dump(optimos, f, indent=2, ensure_ascii=False)

    print(f"Tiempo total: {optimos['segundos_ejecucion']} s")


if __name__ == "__main__":
    main()
