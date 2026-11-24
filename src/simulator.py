"""Motor de simulación discreta por paso diario para políticas (s, Q) y (R, S)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import PARAMETROS, Parametros

METRICAS = [
    "costo_total",
    "costo_mantenimiento",
    "costo_ordenamiento",
    "costo_faltante",
    "fill_rate_pct",
    "csl_pct",
    "cumple_fr95_pct",
    "sin_quiebre_pct",
    "dias_quiebre",
    "inventario_promedio",
    "num_ordenes",
    "undershoot_medio",
    "lt_promedio",
    "demanda_total",
    "unidades_vendidas",
    "unidades_perdidas",
]


def generar_caminos(
    rng_demanda: np.random.Generator,
    rng_lt: np.random.Generator,
    n_dias: int,
    n_ordenes_max: int | None = None,
    p: Parametros = PARAMETROS,
    demanda_const: int | None = None,
    lt_fijo: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-genera secuencias independientes de demanda diaria y lead times para CRN."""
    n_lt = (n_dias + 1) if n_ordenes_max is None else n_ordenes_max

    if demanda_const is not None:
        demanda = np.full(n_dias, demanda_const, dtype=np.int64)
    elif p.dist_demanda == "poisson":
        demanda = rng_demanda.poisson(p.lambda_demanda, size=n_dias)
    else:
        raise ValueError(f"Distribución de demanda no soportada: {p.dist_demanda}")

    if lt_fijo is not None:
        lts = np.full(n_lt, lt_fijo, dtype=np.int64)
    elif p.dist_lt == "uniforme_discreta":
        lts = rng_lt.integers(p.lt_a, p.lt_b + 1, size=n_lt)
    elif p.dist_lt == "gamma_discreta":
        muestras = np.ceil(rng_lt.gamma(p.gamma_shape, p.gamma_scale, size=n_lt)).astype(np.int64)
        lts = np.clip(muestras, 1, 15)
    else:
        raise ValueError(f"Distribución de lead time no soportada: {p.dist_lt}")

    return demanda, lts


def caminos_de_semilla(
    semilla: int | np.random.SeedSequence,
    n_dias: int,
    p: Parametros = PARAMETROS,
    deterministico: dict[str, int] | None = None,
    n_ordenes_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Deriva dos streams independientes (demanda y LT) desde una misma semilla."""
    if isinstance(semilla, np.random.SeedSequence):
        seq = np.random.SeedSequence(
            semilla.entropy, spawn_key=semilla.spawn_key, pool_size=semilla.pool_size
        )
    else:
        seq = np.random.SeedSequence(semilla)
    seq_d, seq_lt = seq.spawn(2)
    rng_d = np.random.default_rng(seq_d)
    rng_lt = np.random.default_rng(seq_lt)
    if deterministico is not None:
        return generar_caminos(
            rng_d, rng_lt, n_dias, n_ordenes_max=n_ordenes_max, p=p,
            demanda_const=deterministico["demanda"], lt_fijo=deterministico["lt"],
        )
    return generar_caminos(rng_d, rng_lt, n_dias, n_ordenes_max=n_ordenes_max, p=p)


def simular(
    demanda: np.ndarray,
    lts: np.ndarray,
    politica: str = "sQ",
    s: int | None = None,
    Q: int | None = None,
    R: int | None = None,
    S: int | None = None,
    inventario_inicial: int = 200,
    h: float | None = None,
    K: float | None = None,
    p_falta: float | None = None,
    p: Parametros = PARAMETROS,
) -> dict[str, Any]:
    """Ejecuta una corrida diaria con ventas perdidas y revisión al cierre.

    Una orden emitida al cierre del día t arriba al inicio del día t + LT + 1,
    dejando exactamente LT días completos de demanda entre emisión y recepción.
    El costo de mantenimiento se calcula sobre el inventario medio intradía
    (I_ini + I_fin)/2 para coincidir exactamente con el supuesto continuo del EOQ.
    """
    h = p.h_diario if h is None else h
    K = p.k_orden if K is None else K
    p_falta = p.p_falta if p_falta is None else p_falta

    if politica == "sQ" and (s is None or Q is None):
        raise ValueError("La política (s, Q) requiere definir s y Q.")
    if politica == "RS" and (R is None or S is None):
        raise ValueError("La política (R, S) requiere definir R y S.")
    if politica not in ("sQ", "RS"):
        raise ValueError(f"Política desconocida: {politica}")

    n_dias = len(demanda)
    max_lt = int(lts.max()) if lts.size else p.lt_b
    dem_l = demanda.tolist()
    lts_l = lts.tolist()
    n_lts = len(lts_l)
    arr = [0] * (n_dias + max_lt + 2)

    inv_ini = [0] * n_dias
    inv_fin = [0] * n_dias
    ventas = [0] * n_dias
    perdidas = [0] * n_dias
    arribos = [0] * n_dias
    pedidos = [0] * n_dias
    lt_pedidos = [0] * n_dias
    ip_cierre = [0] * n_dias
    c_mant = [0.0] * n_dias
    c_ord = [0.0] * n_dias
    c_falt = [0.0] * n_dias

    inv = int(inventario_inicial)
    en_camino = 0
    idx_lt = 0
    half_h = 0.5 * h
    es_sq = politica == "sQ"

    for t in range(n_dias):
        arribo = arr[t]
        if arribo:
            inv += arribo
            en_camino -= arribo
        arribos[t] = arribo
        ini_t = inv
        inv_ini[t] = ini_t

        dem = dem_l[t]
        if dem:
            vendido = dem if dem < inv else inv
            inv -= vendido
            perdido = dem - vendido
        else:
            vendido = 0
            perdido = 0
        ventas[t] = vendido
        perdidas[t] = perdido
        inv_fin[t] = inv

        c_mant[t] = half_h * (ini_t + inv)
        if perdido:
            c_falt[t] = p_falta * perdido

        pedido = 0
        if es_sq:
            if inv + en_camino <= s:
                pedido = Q
        elif (t + 1) % R == 0:
            faltante_nivel = S - (inv + en_camino)
            if faltante_nivel > 0:
                pedido = faltante_nivel

        if pedido > 0:
            if idx_lt >= n_lts:
                raise RuntimeError("Se agotó el arreglo pre-generado de lead times.")
            lt = lts_l[idx_lt]
            idx_lt += 1
            arr[t + lt + 1] += pedido
            en_camino += pedido
            c_ord[t] = K
            pedidos[t] = pedido
            lt_pedidos[t] = lt

        ip_cierre[t] = inv + en_camino

    return {
        "demanda": demanda,
        "arribos": np.asarray(arribos, dtype=np.int64),
        "inv_ini": np.asarray(inv_ini, dtype=np.int64),
        "inv_fin": np.asarray(inv_fin, dtype=np.int64),
        "ventas": np.asarray(ventas, dtype=np.int64),
        "perdidas": np.asarray(perdidas, dtype=np.int64),
        "pedidos": np.asarray(pedidos, dtype=np.int64),
        "lt_pedidos": np.asarray(lt_pedidos, dtype=np.int64),
        "ip_cierre": np.asarray(ip_cierre, dtype=np.int64),
        "c_mant": np.asarray(c_mant, dtype=np.float64),
        "c_ord": np.asarray(c_ord, dtype=np.float64),
        "c_falt": np.asarray(c_falt, dtype=np.float64),
        "politica": politica,
        "parametros": {"s": s, "Q": Q, "R": R, "S": S},
        "inventario_inicial": int(inventario_inicial),
    }


def _calcular_csl(traza: dict[str, Any], warmup: int, n_dias: int) -> float:
    """Nivel de servicio de ciclo (% de ciclos entre arribos sin ventas perdidas)."""
    arr_dias = np.flatnonzero(traza["arribos"][warmup:n_dias] > 0) + warmup
    if len(arr_dias) < 2:
        return 100.0 if traza["perdidas"][warmup:n_dias].sum() == 0 else 0.0
    sin_quiebre = 0
    total_ciclos = len(arr_dias) - 1
    for i in range(total_ciclos):
        if traza["perdidas"][arr_dias[i] : arr_dias[i + 1]].sum() == 0:
            sin_quiebre += 1
    return 100.0 * sin_quiebre / total_ciclos


def calcular_metricas(
    traza: dict[str, Any],
    warmup: int,
    n_dias: int,
    fr_objetivo: float = PARAMETROS.fill_rate_objetivo,
) -> dict[str, float]:
    """Resume indicadores económicos y operativos en la ventana [warmup, n_dias)."""
    w = slice(warmup, n_dias)
    dem = float(traza["demanda"][w].sum())
    ven = float(traza["ventas"][w].sum())
    per = float(traza["perdidas"][w].sum())
    mant = float(traza["c_mant"][w].sum())
    ordz = float(traza["c_ord"][w].sum())
    falt = float(traza["c_falt"][w].sum())
    inv_medio = (traza["inv_ini"][w] + traza["inv_fin"][w]) / 2.0
    lt_w = traza["lt_pedidos"][w]
    lts_usados = lt_w[lt_w > 0]
    fr_pct = (100.0 * ven / dem) if dem > 0 else 100.0

    undershoot_medio = float("nan")
    if traza["politica"] == "sQ":
        ped_w = traza["pedidos"][w]
        mask_ord = ped_w > 0
        if mask_ord.any():
            s_umbral = int(traza["parametros"]["s"])
            q_lote = int(traza["parametros"]["Q"])
            ip_antes = traza["ip_cierre"][w][mask_ord] - q_lote
            undershoot_medio = float((s_umbral - ip_antes).mean())

    return {
        "costo_total": mant + ordz + falt,
        "costo_mantenimiento": mant,
        "costo_ordenamiento": ordz,
        "costo_faltante": falt,
        "fill_rate_pct": fr_pct,
        "csl_pct": _calcular_csl(traza, warmup, n_dias),
        "cumple_fr95_pct": 100.0 if fr_pct >= fr_objetivo else 0.0,
        "sin_quiebre_pct": 100.0 if per == 0.0 else 0.0,
        "dias_quiebre": int((traza["perdidas"][w] > 0).sum()),
        "inventario_promedio": float(inv_medio.mean()),
        "num_ordenes": int((traza["pedidos"][w] > 0).sum()),
        "undershoot_medio": undershoot_medio,
        "lt_promedio": float(lts_usados.mean()) if lts_usados.size else 0.0,
        "demanda_total": dem,
        "unidades_vendidas": ven,
        "unidades_perdidas": per,
    }


def correr_replica(
    semilla: int | np.random.SeedSequence,
    politica: str,
    parametros_politica: dict[str, Any],
    p: Parametros = PARAMETROS,
    deterministico: dict[str, int] | None = None,
    n_dias: int | None = None,
    warmup: int | None = None,
    p_falta: float | None = None,
    devolver_traza: bool = False,
):
    """Ejecuta una réplica completa y retorna sus métricas (y opcionalmente la traza)."""
    n_dias = p.dias_sim if n_dias is None else n_dias
    warmup = p.warmup_dias if warmup is None else warmup
    demanda, lts = caminos_de_semilla(semilla, n_dias, p=p, deterministico=deterministico)
    inv_ini = parametros_politica.get("inv_inicial", parametros_politica.get("Q", 0))
    args_pol = {k: v for k, v in parametros_politica.items() if k != "inv_inicial"}
    traza = simular(
        demanda, lts, politica=politica, inventario_inicial=inv_ini,
        p_falta=p_falta, p=p, **args_pol,
    )
    metricas = calcular_metricas(traza, warmup, n_dias)
    return (metricas, traza) if devolver_traza else metricas
