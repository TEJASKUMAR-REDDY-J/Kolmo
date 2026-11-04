"""System registry (docs/contracts.md): SYSTEMS name -> class, make(name or PDE spec) -> System, describe()."""
from kolmo.systems.boussinesq import Boussinesq
from kolmo.systems.diffusion import Diffusion
from kolmo.systems.fluid2d import Fluid2D
from kolmo.systems.pde import LIBRARY, PDE

SYSTEMS = {"fluid2d": Fluid2D, "diffusion": Diffusion, "boussinesq": Boussinesq, "pde": PDE}


def make(name_or_spec):
    """Registered name, PDE library name (e.g. "gray_scott") or a PDE spec dict -> System instance."""
    if isinstance(name_or_spec, dict):
        return PDE(name_or_spec)
    if name_or_spec in SYSTEMS:
        return SYSTEMS[name_or_spec]()
    if name_or_spec in LIBRARY:
        return PDE(LIBRARY[name_or_spec])
    raise ValueError(f"unknown system {name_or_spec!r}; known: {sorted(SYSTEMS)} and PDE library {sorted(LIBRARY)}")


def describe(name_or_spec):
    return make(name_or_spec).describe()
