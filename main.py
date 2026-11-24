"""Orquestador principal de la campaña de simulación y generación de resultados."""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


def ejecutar_pipeline() -> None:
    from src import experiments, plotting, validation

    ok = validation.ejecutar_validacion_completa(guardar=True)
    if not ok:
        raise RuntimeError("Falló al menos un chequeo de verificación o validación.")

    experiments.main()
    plotting.main()


def main() -> None:
    ejecutar_pipeline()


if __name__ == "__main__":
    main()
