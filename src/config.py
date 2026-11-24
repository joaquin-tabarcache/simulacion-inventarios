"""Configuración central, parámetros del sistema y derivaciones analíticas."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_RESULTADOS = RAIZ / "results"
DIR_IMG = DIR_RESULTADOS / "img"
DIR_NOTEBOOKS = RAIZ / "notebooks"

TITULO = (
    "Simulación y Optimización de Políticas de Inventarios con Demanda y "
    "Tiempos de Entrega Estocásticos"
)
INTEGRANTES = ["Leandro Joaquín Tabarcache"]
CATEDRA = "Modelos y Simulación — Cohorte 2026"
UNIVERSIDAD = "Universidad Católica de Salta (UCASAL)"
FACULTAD = "Facultad de Ingeniería e Informática"
CARRERA = "Ingeniería en Informática"
MODALIDAD = "Modalidad A — Aplicación Práctica"
FECHA = "Noviembre de 2026"

DIAS_ANIO = 365
N_REPLICAS = 100
N_REPLICAS_BUSQUEDA = 30
N_REPLICAS_REFINAMIENTO = 100
HORIZONTES_VALIDACION = [365, 730, 1825, 3650, 36500]
PENALIZACIONES_SENSIBILIDAD = [10.0, 15.0, 20.0, 30.0, 45.0, 60.0]
K_SENSIBILIDAD = [25.0, 50.0, 100.0, 150.0]
WARMUP_VERIFICACION = [60, 120, 200, 365]

GRIDS = {
    "s_valores": list(range(20, 84, 4)),
    "Q_valores": list(range(100, 380, 20)),
    "R_valores": [5, 7, 10, 14, 21, 28, 35, 42],
    "S_margenes": [-30, -20, -10, 0, 10, 20, 30],
    "s_fino": list(range(38, 63, 1)),
    "Q_fino": list(range(160, 250, 10)),
    "R_fino": list(range(20, 35, 1)),
    "S_margenes_fino": list(range(-6, 22, 2)),
}

ESCENARIOS = {
    "E1_Base": {
        "politica": "sQ",
        "s": 28,
        "Q": 200,
        "inv_inicial": 200,
        "descripcion": "(s, Q) con parámetros EOQ y sin stock de seguridad",
    },
    "E2_Conservador": {
        "politica": "sQ",
        "s": 45,
        "Q": 200,
        "inv_inicial": 200,
        "descripcion": "(s, Q) con stock de seguridad analítico (z = 1,645)",
    },
    "E3_Optimizado": {
        "politica": "sQ",
        "s": None,
        "Q": None,
        "inv_inicial": None,
        "descripcion": "(s*, Q*) optimizado por simulación sujeto a Fill Rate >= 95%",
    },
    "E4_Periodico": {
        "politica": "RS",
        "R": None,
        "S": None,
        "inv_inicial": None,
        "descripcion": "(R*, S*) de revisión periódica optimizada sujeto a Fill Rate >= 95%",
    },
}

NOMBRES_ESCENARIOS = {
    "E1_Base": "E1: Base (s, Q) sin SS",
    "E2_Conservador": "E2: Conservador (s, Q) con SS",
    "E3_Optimizado": "E3: Optimizado (s*, Q*)",
    "E4_Periodico": "E4: Periódico (R*, S*)",
}


@dataclass(frozen=True)
class Parametros:
    """Parámetros del sistema de inventario de un SKU bajo revisión diaria."""

    sku: str = "RT-W6-100"
    descripcion: str = (
        "Router Wi-Fi 6 dual-band — Centro de distribución regional (NOA)"
    )
    moneda: str = "USD"
    dias_registro: int = 365
    warmup_dias: int = 60

    dist_demanda: str = "poisson"
    lambda_demanda: float = 8.0

    dist_lt: str = "uniforme_discreta"
    lt_a: int = 2
    lt_b: int = 5
    gamma_shape: float = 4.0
    gamma_scale: float = 0.875

    k_orden: float = 50.0
    h_diario: float = 0.02
    p_falta: float = 30.0
    valor_unitario: float = 60.0

    z_servicio: float = 1.645
    fill_rate_objetivo: float = 95.0

    seed_principal: int = 2026
    seed_busqueda: int = 2027
    seed_ejemplos: int = 2028

    @property
    def dias_sim(self) -> int:
        return self.warmup_dias + self.dias_registro

    @property
    def mu_d(self) -> float:
        return self.lambda_demanda

    @property
    def var_d(self) -> float:
        return self.lambda_demanda

    @property
    def sigma_d(self) -> float:
        return math.sqrt(self.var_d)

    @property
    def mu_lt(self) -> float:
        if self.dist_lt == "uniforme_discreta":
            return (self.lt_a + self.lt_b) / 2.0
        return self.gamma_shape * self.gamma_scale

    @property
    def var_lt(self) -> float:
        if self.dist_lt == "uniforme_discreta":
            n = self.lt_b - self.lt_a + 1
            return (n * n - 1) / 12.0
        return self.gamma_shape * self.gamma_scale**2

    @property
    def demanda_anual(self) -> float:
        return DIAS_ANIO * self.mu_d

    @property
    def h_anual(self) -> float:
        return DIAS_ANIO * self.h_diario

    @property
    def eoq(self) -> float:
        """Lote económico de Wilson: sqrt(2 K D / h_anual)."""
        return math.sqrt(2.0 * self.k_orden * self.demanda_anual / self.h_anual)

    @property
    def sigma_dl(self) -> float:
        """Desvío estándar de la demanda durante el lead time bajo independencia."""
        return math.sqrt(self.mu_lt * self.var_d + self.mu_d**2 * self.var_lt)

    @property
    def ss_analitico(self) -> float:
        return self.z_servicio * self.sigma_dl

    @property
    def s_sin_ss(self) -> int:
        return round(self.mu_d * self.mu_lt)

    @property
    def s_con_ss(self) -> int:
        return round(self.mu_d * self.mu_lt + self.ss_analitico)

    def costo_total_eoq(self, Q: float | None = None) -> float:
        """Costo anual relevante del modelo EOQ (ordenamiento y mantenimiento)."""
        q_val = self.eoq if Q is None else Q
        return self.k_orden * self.demanda_anual / q_val + self.h_anual * q_val / 2.0

    def s_analitico_rs(self, R: int) -> int:
        """Nivel objetivo S analítico para cobertura de R + LT días con stock de seguridad."""
        sigma = math.sqrt((R + self.mu_lt) * self.var_d + self.mu_d**2 * self.var_lt)
        return round(self.mu_d * (R + self.mu_lt) + self.z_servicio * sigma)

    def resumen_dict(self) -> dict[str, float | int | str]:
        return {
            "sku": self.sku,
            "lambda_demanda": self.lambda_demanda,
            "demanda_anual": self.demanda_anual,
            "mu_LT": self.mu_lt,
            "sigma_LT": math.sqrt(self.var_lt),
            "mu_D": self.mu_d,
            "sigma_D": self.sigma_d,
            "sigma_DL": self.sigma_dl,
            "K": self.k_orden,
            "h_diario": self.h_diario,
            "h_anual": self.h_anual,
            "p": self.p_falta,
            "valor_unitario": self.valor_unitario,
            "EOQ": self.eoq,
            "s_sin_SS": self.s_sin_ss,
            "SS_analitico": self.ss_analitico,
            "s_con_SS": self.s_con_ss,
            "costo_total_anual_EOQ": self.costo_total_eoq(),
            "warmup_dias": self.warmup_dias,
            "dias_registro": self.dias_registro,
        }


PARAMETROS = Parametros()


@dataclass(frozen=True)
class ConfigValidacion:
    """Escenario determinístico calibrado para reproducir exactamente el EOQ de Wilson."""

    demanda_constante: int = 8
    lt_fijo: int = 3
    s: int = 24
    Q: int = 200
    inventario_inicial: int = 200


CONFIG_VALIDACION = ConfigValidacion()
