"""Shared contact geometry for the MPM solver and the implicit fallback.

Single source of truth for the collider constants (sphere, box bounds, ground)
and the numpy contact projection. Both solvers consume this module so their
contact behavior can never drift apart:

  - `ImplicitClothSim` calls `project_contact(x, v)` on particle state after each
    step (pure numpy).
  - `MPMClothSim`'s Taichi `grid_update` kernel cannot call numpy, so it keeps
    its own kernel body but reads the *same constants* (`sphere_c`, `sphere_r`,
    `n_grid`, `box_margin_cells`) from a `ContactGeometry` instance. This is the
    guard against the constant-mismatch class of bug (e.g. a 3-cell box margin
    computed as `domain/64` on one side and `domain/128` on the other).

Coordinate frame: world coordinates in `[0, domain_size]^3`, matching
`mpm.yaml`. The sphere center is given in the same frame.

Physics: frictionless velocity projection. Inward/outward normal velocity
components are zeroed; tangential motion is untouched. Coulomb friction
(`contact.friction_mu` in the config) is intentionally not modeled here — the
MPM grid update is frictionless too, so this keeps the two solvers in parity.
Friction is deferred as future work.
"""

from typing import Any

import numpy as np

# Box/ground margin in grid cells. Matches the MPM `grid_update` kernel, which
# projects velocity on the outer 3 cells of every face.
BOX_MARGIN_CELLS = 3


class ContactGeometry:
    """Collider constants + numpy contact projection, built from a config dict.

    Read-only after construction. Holds everything both solvers need so the
    Taichi kernel and the numpy fallback stay in lockstep.
    """

    def __init__(self, config: dict[str, Any]):
        contact = config["contact"]
        sphere = contact["primitives"]["sphere"]
        self.sphere_enabled = bool(sphere.get("enabled", True))
        self.sphere_c = np.asarray(sphere["center_m"], dtype=np.float64)
        self.sphere_r = float(sphere["radius_m"])

        self.domain = float(config["mpm"]["domain_size_m"])
        self.n_grid = int(config["mpm"]["grid_resolution"])
        self.box_margin_cells = BOX_MARGIN_CELLS
        # World-space margin: BOX_MARGIN_CELLS cells in from each face. Uses the
        # real grid resolution, NOT a hardcoded 64.
        self.margin = self.box_margin_cells * (self.domain / self.n_grid)

        # Ground plane. The box floor already clamps y at `margin`; an explicit
        # ground above that margin raises the effective floor. Defaults below
        # the margin (0.0), so by default the box floor is the ground and this
        # is exact MPM parity.
        self.ground_y = float(contact.get("ground_y_m", 0.0))

    # -- projection -----------------------------------------------------------

    def project_contact(self, x: np.ndarray, v: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
        """Project particle positions/velocities out of all colliders.

        x, v: (N, 3) arrays. Returns new (x, v); inputs are not mutated.
        Order is sphere -> box -> ground; ground runs last so it wins on the
        bottom face when raised above the box margin.
        """
        x = np.array(x, dtype=x.dtype, copy=True)
        v = np.array(v, dtype=v.dtype, copy=True)
        self._project_sphere(x, v)
        self._project_box(x, v)
        self._project_ground(x, v)
        return x, v

    def _project_sphere(self, x: np.ndarray, v: np.ndarray) -> None:
        if not self.sphere_enabled:
            return
        rel = x - self.sphere_c
        dist = np.linalg.norm(rel, axis=-1, keepdims=True)
        inside = (dist < self.sphere_r).squeeze(-1)
        if not inside.any():
            return
        n = rel[inside] / np.maximum(dist[inside], 1e-12)
        # Push to the surface.
        x[inside] = self.sphere_c + n * self.sphere_r
        # Zero the inward (negative) normal velocity component only.
        vn = (v[inside] * n).sum(axis=-1, keepdims=True)
        v[inside] -= np.minimum(vn, 0.0) * n

    def _project_box(self, x: np.ndarray, v: np.ndarray) -> None:
        lo, hi = self.margin, self.domain - self.margin
        for dim in range(3):
            below = x[:, dim] < lo
            above = x[:, dim] > hi
            x[below, dim] = lo
            v[below, dim] = np.maximum(0.0, v[below, dim])  # no outward (down) vel
            x[above, dim] = hi
            v[above, dim] = np.minimum(0.0, v[above, dim])  # no outward (up) vel

    def _project_ground(self, x: np.ndarray, v: np.ndarray) -> None:
        floor = max(self.margin, self.ground_y)
        below = x[:, 1] < floor
        x[below, 1] = floor
        v[below, 1] = np.maximum(0.0, v[below, 1])
