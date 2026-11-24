"""Pruebas unitarias automatizadas para el motor de simulación y validación EOQ."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import CONFIG_VALIDACION, PARAMETROS
from src.simulator import calcular_metricas, caminos_de_semilla, correr_replica, simular
from src.validation import (
    validar_eoq,
    verificar_flujo,
    verificar_ip,
    verificar_no_negativo,
    verificar_politica,
)


def _replica(politica: str, cfg: dict, semilla: int = 11):
    demanda, lts = caminos_de_semilla(semilla, PARAMETROS.dias_sim)
    args_pol = {k: v for k, v in cfg.items() if k != "inv_inicial"}
    return simular(demanda, lts, politica=politica, inventario_inicial=cfg["inv_inicial"], **args_pol)


def test_flujo_conservacion():
    for pol, cfg in [("sQ", dict(s=45, Q=200, inv_inicial=200)), ("RS", dict(R=7, S=105, inv_inicial=105))]:
        assert verificar_flujo(_replica(pol, cfg, semilla=3), cfg["inv_inicial"])


def test_stock_no_negativo():
    for pol, cfg in [("sQ", dict(s=20, Q=120, inv_inicial=120)), ("RS", dict(R=5, S=90, inv_inicial=90))]:
        assert verificar_no_negativo(_replica(pol, cfg, semilla=7))


def test_identidad_ip():
    assert verificar_ip(_replica("sQ", dict(s=45, Q=200, inv_inicial=200), semilla=5))


def test_cumplimiento_politicas():
    assert verificar_politica(_replica("sQ", dict(s=45, Q=200, inv_inicial=200), semilla=2))
    assert verificar_politica(_replica("RS", dict(R=7, S=105, inv_inicial=105), semilla=2))


def test_deterministico_sin_faltantes():
    cfg = CONFIG_VALIDACION
    demanda, lts = caminos_de_semilla(0, 730, deterministico={"demanda": cfg.demanda_constante, "lt": cfg.lt_fijo})
    tr = simular(demanda, lts, politica="sQ", s=cfg.s, Q=cfg.Q, inventario_inicial=cfg.inventario_inicial)
    assert tr["perdidas"].sum() == 0
    dias_ord = np.flatnonzero(tr["pedidos"] > 0)
    assert np.all(np.diff(dias_ord) == cfg.Q / cfg.demanda_constante)


def test_validacion_eoq_error_menor_1pct():
    df = validar_eoq(horizontes=[1825, 3650])
    assert (df["error_pct"].abs() < 1.0).all()
    assert np.isclose(df["ciclo_dias"], df["ciclo_analitico_dias"]).all()


def test_ordenes_atributo_lt_valido():
    tr = _replica("sQ", dict(s=45, Q=200, inv_inicial=200), semilla=17)
    lts_usados = tr["lt_pedidos"][tr["lt_pedidos"] > 0]
    assert ((lts_usados >= PARAMETROS.lt_a) & (lts_usados <= PARAMETROS.lt_b)).all()


def test_borde_demanda_nula_y_reproducibilidad():
    d0, lt0 = caminos_de_semilla(99, 60, deterministico={"demanda": 0, "lt": 3})
    tr0 = simular(d0, lt0, politica="sQ", s=20, Q=100, inventario_inicial=50)
    assert tr0["pedidos"].sum() == 0 and tr0["perdidas"].sum() == 0

    d1, lt1 = caminos_de_semilla(1234, 425)
    d2, lt2 = caminos_de_semilla(1234, 425)
    assert np.array_equal(d1, d2) and np.array_equal(lt1, lt2)
    m1 = calcular_metricas(simular(d1, lt1, politica="sQ", s=45, Q=200, inventario_inicial=200), 60, 425)
    m2 = calcular_metricas(simular(d2, lt2, politica="sQ", s=45, Q=200, inventario_inicial=200), 60, 425)
    assert m1 == m2


def test_crn_seedsequence_sin_mutacion():
    ss = np.random.SeedSequence(2026)
    d1, lt1 = caminos_de_semilla(ss, 425)
    d2, lt2 = caminos_de_semilla(ss, 425)
    assert np.array_equal(d1, d2) and np.array_equal(lt1, lt2)

    m_e1 = correr_replica(ss, "sQ", dict(s=45, Q=200, inv_inicial=200))
    m_e3 = correr_replica(ss, "sQ", dict(s=58, Q=160, inv_inicial=160))
    tr_e1 = simular(d1, lt1, politica="sQ", s=45, Q=200, inventario_inicial=200)
    tr_e3 = simular(d1, lt1, politica="sQ", s=58, Q=160, inventario_inicial=160)
    assert np.array_equal(tr_e1["demanda"], tr_e3["demanda"])
    assert m_e1["demanda_total"] == m_e3["demanda_total"]


def test_metricas_undershoot_y_servicio_binario():
    m_sq = correr_replica(77, "sQ", dict(s=45, Q=200, inv_inicial=200))
    assert m_sq["undershoot_medio"] >= 0.0
    assert m_sq["cumple_fr95_pct"] in (0.0, 100.0)
    assert m_sq["sin_quiebre_pct"] in (0.0, 100.0)

    m_rs = correr_replica(77, "RS", dict(R=28, S=292, inv_inicial=292))
    assert np.isnan(m_rs["undershoot_medio"])
