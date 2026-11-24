"""Verificación estructural, bondad de ajuste, trazas y validación contra EOQ."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import (
    CONFIG_VALIDACION,
    DIR_RESULTADOS,
    ESCENARIOS,
    HORIZONTES_VALIDACION,
    PARAMETROS,
    Parametros,
)
from .simulator import caminos_de_semilla, calcular_metricas, simular

N_LT_TRAZA = 20


def verificar_flujo(traza: dict, inventario_inicial: int) -> bool:
    """Comprueba la ecuación de balance diario de stock físico."""
    n = len(traza["inv_fin"])
    if traza["inv_ini"][0] != inventario_inicial + traza["arribos"][0]:
        return False
    if traza["inv_fin"][0] != traza["inv_ini"][0] - traza["ventas"][0]:
        return False
    for t in range(1, n):
        if traza["inv_ini"][t] != traza["inv_fin"][t - 1] + traza["arribos"][t]:
            return False
        if traza["inv_fin"][t] != traza["inv_ini"][t] - traza["ventas"][t]:
            return False
    return True


def verificar_no_negativo(traza: dict) -> bool:
    """Comprueba que el stock físico nunca tome valores negativos."""
    return bool((traza["inv_fin"] >= 0).all() and (traza["inv_ini"] >= 0).all())


def verificar_ip(traza: dict) -> bool:
    """Verifica la identidad IP(t) = I_fin(t) + en_camino(t)."""
    en_camino = np.cumsum(traza["pedidos"]) - np.cumsum(traza["arribos"])
    return bool((traza["ip_cierre"] == traza["inv_fin"] + en_camino).all())


def verificar_politica(traza: dict) -> bool:
    """Comprueba el cumplimiento estricto de las reglas (s, Q) y (R, S)."""
    prm = traza["parametros"]
    pedidos = traza["pedidos"]
    ip_pre = traza["ip_cierre"] - pedidos

    if traza["politica"] == "sQ":
        s, q = prm["s"], prm["Q"]
        if not np.isin(pedidos, [0, q]).all():
            return False
        ordeno = pedidos > 0
        return bool((ip_pre[ordeno] <= s).all() and (ip_pre[~ordeno] > s).all())

    r, s_obj = prm["R"], prm["S"]
    es_revision = (np.arange(len(pedidos)) + 1) % r == 0
    for t, ped in enumerate(pedidos):
        if ped > 0:
            if not es_revision[t] or ped != s_obj - ip_pre[t]:
                return False
        elif es_revision[t] and (s_obj - ip_pre[t] > 0):
            return False
    return True


def chi2_poisson(p: Parametros = PARAMETROS, semilla: int | None = None, n: int = 100_000) -> dict:
    """Contraste chi-cuadrado de bondad de ajuste para la demanda Poisson(lambda)."""
    seq = np.random.SeedSequence(p.seed_ejemplos if semilla is None else semilla).spawn(2)[0]
    datos = np.random.default_rng(seq).poisson(p.lambda_demanda, size=n)
    kmax = 40
    obs = np.bincount(datos, minlength=kmax + 2)[: kmax + 2].astype(float)
    esperada = np.array([stats.poisson.pmf(k, p.lambda_demanda) * n for k in range(kmax + 2)])
    while esperada[-1] < 5:
        esperada[-2] += esperada[-1]
        obs[-2] += obs[-1]
        esperada, obs = esperada[:-1], obs[:-1]
    stat, pval = stats.chisquare(obs, esperada)
    return {
        "stat": float(stat),
        "pvalor": float(pval),
        "gl": int(len(obs) - 1),
        "media_obs": float(datos.mean()),
        "var_obs": float(datos.var(ddof=1)),
    }


def chi2_lt(p: Parametros = PARAMETROS, semilla: int | None = None, n: int = 40_000) -> dict:
    """Contraste chi-cuadrado de bondad de ajuste para el lead time uniforme discreto."""
    seq = np.random.SeedSequence(p.seed_ejemplos if semilla is None else semilla).spawn(2)[1]
    datos = np.random.default_rng(seq).integers(p.lt_a, p.lt_b + 1, size=n)
    valores = np.arange(p.lt_a, p.lt_b + 1)
    obs = np.array([(datos == v).sum() for v in valores], dtype=float)
    esperada = np.full(len(valores), n / len(valores))
    stat, pval = stats.chisquare(obs, esperada)
    return {
        "stat": float(stat),
        "pvalor": float(pval),
        "gl": int(len(obs) - 1),
        "media_obs": float(datos.mean()),
    }


def traza_a_dataframe(traza: dict, n_dias: int) -> pd.DataFrame:
    """Convierte los primeros n_dias de una traza a DataFrame tabular."""
    df = pd.DataFrame({
        "dia": np.arange(n_dias),
        "demanda": traza["demanda"][:n_dias],
        "arribos": traza["arribos"][:n_dias],
        "inv_ini": traza["inv_ini"][:n_dias],
        "ventas": traza["ventas"][:n_dias],
        "perdidas": traza["perdidas"][:n_dias],
        "inv_fin": traza["inv_fin"][:n_dias],
        "ip_cierre": traza["ip_cierre"][:n_dias],
        "pedido": traza["pedidos"][:n_dias],
        "lt_pedido": traza["lt_pedidos"][:n_dias],
        "costo_mant": np.round(traza["c_mant"][:n_dias], 2),
        "costo_ord": np.round(traza["c_ord"][:n_dias], 2),
        "costo_falt": np.round(traza["c_falt"][:n_dias], 2),
    })
    df["costo_dia"] = (df["costo_mant"] + df["costo_ord"] + df["costo_falt"]).round(2)
    return df


def traza_determinista(n_dias: int = 15, inventario_inicial: int | None = None) -> pd.DataFrame:
    """Genera traza verificable manualmente con demanda y lead time constantes."""
    cfg = CONFIG_VALIDACION
    inv0 = cfg.inventario_inicial if inventario_inicial is None else inventario_inicial
    traza = simular(
        np.full(n_dias, cfg.demanda_constante, dtype=np.int64),
        np.full(N_LT_TRAZA, cfg.lt_fijo, dtype=np.int64),
        politica="sQ", s=cfg.s, Q=cfg.Q, inventario_inicial=inv0,
    )
    return traza_a_dataframe(traza, n_dias)


def traza_estocastica_15d(
    p: Parametros = PARAMETROS,
    semilla: np.random.SeedSequence | int | None = None,
    escenario: str = "E2_Conservador",
) -> pd.DataFrame:
    """Genera una traza estocástica de 15 días para inspección de eventos aleatorios."""
    sem = semilla if semilla is not None else np.random.SeedSequence(p.seed_ejemplos).spawn(4)[0]
    cfg = ESCENARIOS[escenario]
    demanda, lts = caminos_de_semilla(sem, p.dias_sim, p=p)
    traza = simular(
        demanda, lts, politica=cfg["politica"], s=cfg.get("s"), Q=cfg.get("Q"),
        R=cfg.get("R"), S=cfg.get("S"), inventario_inicial=cfg["inv_inicial"], p=p,
    )
    return traza_a_dataframe(traza, 15)


def validar_eoq(horizontes: list[int] | None = None, p: Parametros = PARAMETROS) -> pd.DataFrame:
    """Valida convergencia del simulador determinístico hacia el EOQ analítico de Wilson."""
    horizontes = HORIZONTES_VALIDACION if horizontes is None else horizontes
    cfg = CONFIG_VALIDACION
    filas = []
    for horizon in horizontes:
        demanda, lts = caminos_de_semilla(
            0, horizon, p=p, deterministico={"demanda": cfg.demanda_constante, "lt": cfg.lt_fijo},
        )
        traza = simular(
            demanda, lts, politica="sQ", s=cfg.s, Q=cfg.Q,
            inventario_inicial=cfg.inventario_inicial, p=p,
        )
        m = calcular_metricas(traza, warmup=0, n_dias=horizon)
        dias_ord = np.flatnonzero(traza["pedidos"] > 0)
        ciclo = float(np.diff(dias_ord).mean()) if len(dias_ord) > 1 else float("nan")
        anios = horizon / 365.0
        costo_anual = m["costo_total"] / anios
        tc_analitico = p.costo_total_eoq(cfg.Q)
        filas.append({
            "horizonte_dias": horizon,
            "anios": anios,
            "ordenes": m["num_ordenes"],
            "ordenes_anuales": m["num_ordenes"] / anios,
            "costo_total_simulado": round(m["costo_total"], 2),
            "costo_anualizado": round(costo_anual, 2),
            "costo_analitico_eoq": round(tc_analitico, 2),
            "error_pct": round(100.0 * (costo_anual - tc_analitico) / tc_analitico, 3),
            "ciclo_dias": round(ciclo, 3),
            "ciclo_analitico_dias": cfg.Q / cfg.demanda_constante,
            "fill_rate_pct": round(m["fill_rate_pct"], 3),
        })
    return pd.DataFrame(filas)


def ejecutar_validacion_completa(guardar: bool = True) -> bool:
    """Ejecuta toda la batería de verificación y validación y exporta las tablas CSV."""
    checks: list[tuple[str, bool, str]] = []

    def registrar(nombre: str, ok: bool, detalle: str = "") -> None:
        checks.append((nombre, ok, detalle))
        estado = "PASS" if ok else "FAIL"
        print(f"  [{estado}] {nombre}" + (f" — {detalle}" if detalle else ""))

    print("=" * 76)
    print("VERIFICACIÓN Y VALIDACIÓN DEL SIMULADOR DE INVENTARIOS")
    print("=" * 76)

    print("\n1. Verificación estructural (V1-V4)")
    semillas = np.random.SeedSequence(PARAMETROS.seed_ejemplos).spawn(4)
    configs = [
        ("(s,Q) E2_Conservador", "sQ", dict(s=45, Q=200, inv_inicial=200)),
        ("(s,Q) E1_Base", "sQ", dict(s=28, Q=200, inv_inicial=200)),
        ("(R,S) R=7, S=105", "RS", dict(R=7, S=105, inv_inicial=105)),
    ]
    for idx, (nombre, pol, cfgp) in enumerate(configs):
        demanda, lts = caminos_de_semilla(semillas[idx], PARAMETROS.dias_sim)
        args_pol = {k: v for k, v in cfgp.items() if k != "inv_inicial"}
        traza = simular(demanda, lts, politica=pol, inventario_inicial=cfgp["inv_inicial"], **args_pol)
        registrar(f"V1 conservación de flujo [{nombre}]", verificar_flujo(traza, cfgp["inv_inicial"]))
        registrar(f"V2 stock no negativo [{nombre}]", verificar_no_negativo(traza))
        registrar(f"V3 identidad IP [{nombre}]", verificar_ip(traza))
        registrar(f"V4 cumplimiento política {pol} [{nombre}]", verificar_politica(traza))

    print("\n2. Bondad de ajuste de generadores (V5)")
    r_p = chi2_poisson()
    registrar(
        "V5a demanda ~ Poisson(8)",
        r_p["pvalor"] > 0.05,
        f"chi2={r_p['stat']:.2f}, gl={r_p['gl']}, p={r_p['pvalor']:.4f}, "
        f"media={r_p['media_obs']:.3f}, var={r_p['var_obs']:.2f}",
    )
    r_l = chi2_lt()
    registrar(
        "V5b lead time ~ U{2..5}",
        r_l["pvalor"] > 0.05,
        f"chi2={r_l['stat']:.2f}, gl={r_l['gl']}, p={r_l['pvalor']:.4f}, media={r_l['media_obs']:.3f}",
    )

    print("\n3. Traza determinista 15 días (V6)")
    df_tr = traza_determinista(15, inventario_inicial=24)
    print(df_tr.to_string(index=False))
    ok_traza = (
        df_tr.loc[0, "pedido"] == 200
        and df_tr.loc[4, "arribos"] == 200
        and df_tr.loc[3, "perdidas"] == 8
        and df_tr.loc[5, "inv_fin"] == 184
    )
    registrar("V6 traza determinista coherente", ok_traza)

    print("\n4. Validación analítica contra modelo EOQ")
    df_eoq = validar_eoq()
    print(df_eoq.to_string(index=False))
    err_regimen = float(df_eoq.loc[df_eoq["horizonte_dias"] >= 1825, "error_pct"].abs().max())
    err_2anios = float(abs(df_eoq.loc[df_eoq["horizonte_dias"] == 730, "error_pct"].iloc[0]))
    err_ciclo = float((df_eoq["ciclo_dias"] - df_eoq["ciclo_analitico_dias"]).abs().max())
    registrar("EOQ error < 1% en régimen (T >= 5 años)", err_regimen < 1.0, f"máx |error| = {err_regimen:.3f}%")
    registrar("EOQ error < 1% en horizonte 2 años", err_2anios < 1.0, f"|error| = {err_2anios:.3f}%")
    registrar("EOQ sin faltantes en modo determinístico", df_eoq["fill_rate_pct"].min() == 100.0)
    registrar("EOQ ciclo exacto = 25 días", err_ciclo < 1e-6, f"máx desvío = {err_ciclo:.2e}")

    if guardar:
        DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
        df_tr.to_csv(DIR_RESULTADOS / "traza_determinista_15d.csv", index=False)
        traza_estocastica_15d().to_csv(DIR_RESULTADOS / "traza_estocastica_15d.csv", index=False)
        df_eoq.to_csv(DIR_RESULTADOS / "validacion_eoq.csv", index=False)

    fallos = sum(1 for _, ok, _ in checks if not ok)
    print(f"\nRESUMEN V&V: {len(checks) - fallos}/{len(checks)} chequeos PASS")
    return fallos == 0


if __name__ == "__main__":
    raise SystemExit(0 if ejecutar_validacion_completa() else 1)
