"""Shared contracts of Kolmo. Every physical system, surrogate and execution mode speaks these types.

A state is a float32 array of shape (C, H, W): C named fields on a periodic H x W grid of side L (state.shape[0]
equals len(system.fields)). Parameters are a plain dict validated by the system; time advances in solver steps of
size params["dt"]. Nothing here imports a solver or a network, so a new domain only implements `System`."""
from dataclasses import dataclass, field
from typing import Callable, Protocol

import jax
import numpy as np

State = jax.Array  # (C, H, W) float32
Stepper = Callable[[State], State]  # one step of size dt with fixed parameters


class System(Protocol):
    name: str
    fields: tuple[str, ...]  # channel names, e.g. ("vorticity",)

    def validate(self, params: dict) -> dict:
        """Complete defaults, check ranges and stability limits; raise ValueError with a clear message."""

    def initial(self, params: dict, seed: int) -> State:
        """Initial condition drawn deterministically from `seed`."""

    def stepper(self, params: dict) -> Stepper:
        """Jitted reference step state(t) -> state(t + dt)."""

    def diagnostics(self, state: State, params: dict) -> dict[str, float]:
        """Physical quantities of one state (energy, enstrophy, max |field|, ...). Names are system-specific."""

    def residual(self, prev: State, nxt: State, params: dict) -> float:
        """Relative residual of the governing equation for the transition prev -> nxt over one dt (0 for the
        reference solver up to its own truncation error). Measures physical consistency, NOT prediction error."""


@dataclass
class Trajectory:
    """States at steps 0..T (inclusive) and how each step was produced."""
    states: np.ndarray  # (T + 1, C, H, W)
    source: list[str] = field(default_factory=list)  # per step 1..T: "reference" | "surrogate"
    events: list[dict] = field(default_factory=list)  # e.g. {"step": 40, "kind": "intervention", "reason": "..."}
    seconds: float = 0.0  # wall time of the steady-state loop (compilation excluded)
    compile_seconds: float = 0.0  # first-call compilation time
