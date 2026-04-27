from __future__ import annotations

from mini_grin_rebuild.core.configs import SimulationConfig
from mini_grin_rebuild.simulation.engines.ideal_gradient import IdealGradientEngine
from mini_grin_rebuild.simulation.engines.instrument_lite import InstrumentLiteEngine
from mini_grin_rebuild.simulation.engines.base import SimulationEngine


def create_simulation_engine(cfg: SimulationConfig) -> SimulationEngine:
    name = str(getattr(cfg, "capture_engine", "ideal_gradient") or "ideal_gradient")
    if name in {"ideal_gradient", "legacy"}:
        return IdealGradientEngine(cfg)
    if name == "instrument_lite":
        return InstrumentLiteEngine(cfg)
    raise ValueError(f"Unknown capture_engine={name!r}")


__all__ = ["create_simulation_engine"]
