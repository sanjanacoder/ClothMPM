"""Implicit mass-spring cloth solver (Baraff & Witkin 1998).

Mass-spring implicit Euler solver on the same 64x64 quad-grid particle layout
as the MPM simulator. Used as a stable fallback in high-complexity regions
where the MPM solver's explicit integration may lose accuracy.

Spring topology on the 64x64 quad grid:
  - structural: horizontal + vertical neighbors (each particle has up to 4)
  - shear     : both diagonals of each quad
  - bend      : skip-one neighbors along rows and columns

Implicit Euler step (Baraff-Witkin Eq. 6, symmetric form Eq. 15):
  (M - h df/dv - h^2 df/dx) Delta_v = h * (f_0 + h * df/dx * v_0)
followed by Delta_x = h * (v_0 + Delta_v).

The system is sparse SPD (after the symmetry trick); we solve via a modified
preconditioned conjugate gradient with a per-particle `filter` operator that
projects out constrained directions (Eq. 12-13). Pinned corners are exact
regardless of CG iteration count -- this is the load-bearing property of the
paper.

Implementation choices vs. the paper:
  - Regular 64x64 quad mesh (not arbitrary triangle continuum); spring forces
    rather than per-triangle stretch/shear conditions. This simplification is
    sufficient for a stable fallback; the MPM simulator is the high-fidelity solver.
  - Damping uses the same spring direction but the simpler form
        f_d = -k_d * <v_i - v_j, e_hat> * e_hat
    rather than reusing the condition C(x). df_d/dv is identical in structure
    to df/dx, so the linear system has the same block sparsity.
  - Spring stiffness k_s is Pa-equivalent: we map a target Young's modulus E
    [Pa] and rest length L_0 [m] to k_s = E * thickness * L_0 / L_0 = E * h
    (membrane analogy). This keeps a single material knob aligned with mpm.yaml.

Public API:
    class ImplicitClothSim:
        def __init__(self, config: dict | str | Path): ...
        def reset(self, x0: np.ndarray | None = None,
                  v0: np.ndarray | None = None,
                  pinned: list[int] | None = None) -> None: ...
        def step(self, f_ext: np.ndarray | None = None,
                 dt: float | None = None,
                 cg_max_iters: int = 50,
                 cg_tol: float = 1e-4) -> dict: ...
        def state(self) -> dict: ...
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import yaml


# -----------------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------------

def load_implicit_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read configs/mpm.yaml (the cloth/material section is shared) and add
    implicit-specific defaults."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "configs" / "mpm.yaml"
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.setdefault("implicit", {})
    cfg["implicit"].setdefault("damping_per_unit_stiffness", 0.05)
    cfg["implicit"].setdefault("bend_stiffness_ratio", 0.1)   # bend k = ratio * structural k
    cfg["implicit"].setdefault("shear_stiffness_ratio", 0.5)
    cfg["implicit"].setdefault("dt_s", 0.02)                  # Baraff-Witkin default 0.02 s
    return cfg


# -----------------------------------------------------------------------------
# Mesh / spring topology
# -----------------------------------------------------------------------------

@dataclass
class ClothTopology:
    grid_x: int
    grid_y: int
    structural_pairs: np.ndarray = field(repr=False)   # (S, 2) int
    shear_pairs: np.ndarray = field(repr=False)        # (S, 2) int
    bend_pairs: np.ndarray = field(repr=False)         # (S, 2) int
    rest_lengths: dict[str, np.ndarray] = field(repr=False)


def _idx(i: int, j: int, gy: int) -> int:
    return i * gy + j


def build_topology(grid_x: int, grid_y: int, size_x: float, size_y: float) -> ClothTopology:
    """Build spring index pairs and rest lengths for a regular grid of cloth particles.

    Particles are placed on the xz plane (y=0) with spacing (size_x/grid_x, size_y/grid_y),
    indexed by `i * grid_y + j`. Rest lengths are computed from these positions.
    """
    n = grid_x * grid_y
    dx = size_x / grid_x
    dy = size_y / grid_y

    structural, shear, bend = [], [], []
    for i in range(grid_x):
        for j in range(grid_y):
            p = _idx(i, j, grid_y)
            if i + 1 < grid_x:
                structural.append((p, _idx(i + 1, j, grid_y)))
            if j + 1 < grid_y:
                structural.append((p, _idx(i, j + 1, grid_y)))
            if i + 1 < grid_x and j + 1 < grid_y:
                shear.append((p, _idx(i + 1, j + 1, grid_y)))
                shear.append((_idx(i + 1, j, grid_y), _idx(i, j + 1, grid_y)))
            if i + 2 < grid_x:
                bend.append((p, _idx(i + 2, j, grid_y)))
            if j + 2 < grid_y:
                bend.append((p, _idx(i, j + 2, grid_y)))

    structural = np.asarray(structural, dtype=np.int32)
    shear = np.asarray(shear, dtype=np.int32)
    bend = np.asarray(bend, dtype=np.int32)

    # Rest-pose positions (x, 0, z)
    rest = np.zeros((n, 3), dtype=np.float64)
    for i in range(grid_x):
        for j in range(grid_y):
            rest[_idx(i, j, grid_y)] = [(i + 0.5) * dx, 0.0, (j + 0.5) * dy]

    def L0(pairs):
        if pairs.size == 0:
            return np.zeros((0,))
        return np.linalg.norm(rest[pairs[:, 1]] - rest[pairs[:, 0]], axis=-1)

    return ClothTopology(
        grid_x=grid_x, grid_y=grid_y,
        structural_pairs=structural, shear_pairs=shear, bend_pairs=bend,
        rest_lengths={"structural": L0(structural),
                      "shear":      L0(shear),
                      "bend":       L0(bend)},
    )


# -----------------------------------------------------------------------------
# Spring forces and Jacobians
# -----------------------------------------------------------------------------

def _spring_forces_and_blocks(
    pairs: np.ndarray,
    rest: np.ndarray,
    k: float,
    k_d: float,
    x: np.ndarray,
    v: np.ndarray,
):
    """For one spring family, compute per-particle force contributions and the
    sparse 3x3 blocks for df/dx and df/dv on each (i, j, j, i, i, j) endpoint.

    Returns:
      f       : (N, 3) accumulated force from this family
      rows, cols, blocks_xx, blocks_xv : sparse triplets for df/dx and df/dv

    The Jacobian formulation per spring (Baraff-Witkin Eq. 7-8 specialized to a
    one-condition spring):
      Let e = x_j - x_i, L = ||e||, L0 = rest length, e_hat = e / L.
      f_i = +k * (L - L0) * e_hat       (force on i pulling toward j when stretched)
      f_j = -f_i
      df_i/dx_j = +k * [(L - L0)/L * (I - e_hat e_hat^T) + e_hat e_hat^T]
      df_i/dx_i = -df_i/dx_j
      Damping (linear):
        f_d_i = +k_d * <v_j - v_i, e_hat> * e_hat
        df_d_i/dv_j = +k_d * (e_hat e_hat^T)
        df_d_i/dv_i = -df_d_i/dv_j
    """
    if pairs.size == 0:
        return (np.zeros((0, 3)),
                np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64),
                np.zeros((0, 3, 3)), np.zeros((0, 3, 3)))

    pi = pairs[:, 0]
    pj = pairs[:, 1]
    e = x[pj] - x[pi]                    # (S, 3)
    L = np.linalg.norm(e, axis=-1)
    L_safe = np.where(L > 1e-12, L, 1e-12)
    e_hat = e / L_safe[:, None]

    stretch = (L - rest)
    f_per = (k * stretch)[:, None] * e_hat   # force on i toward j (positive when stretched)

    # Damping
    rel_v = v[pj] - v[pi]
    rel_v_dot = (rel_v * e_hat).sum(axis=-1)
    f_d_per = (k_d * rel_v_dot)[:, None] * e_hat

    f_total = f_per + f_d_per

    # Per-particle accumulation: i gets +f_total, j gets -f_total
    n = x.shape[0]
    f = np.zeros((n, 3))
    np.add.at(f, pi, +f_total)
    np.add.at(f, pj, -f_total)

    # Stiffness blocks df/dx: per spring, the i->j cross block is K_ij_x
    # K_ij_x = +k * [(L-L0)/L * (I - e_hat eT) + e_hat eT]
    eye = np.eye(3)
    factor_proj = (k * stretch / L_safe)[:, None, None]            # (S, 1, 1)
    eet = e_hat[:, :, None] * e_hat[:, None, :]                    # (S, 3, 3)
    K_ij_x = factor_proj * (eye - eet) + k * eet                   # (S, 3, 3)
    # Damping blocks df/dv: K_ij_v = +k_d * eet
    K_ij_v = k_d * eet                                             # (S, 3, 3)
    return pi, pj, K_ij_x, K_ij_v, f


# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------

class ImplicitClothSim:
    """Implicit-Euler mass-spring cloth solver on a regular grid.

    Single-state object; not thread-safe (uses NumPy + SciPy sparse).
    """

    def __init__(self, config: dict[str, Any] | str | Path):
        if not isinstance(config, dict):
            config = load_implicit_config(config)
        self.cfg = config
        gx, gy = config["cloth"]["grid"]
        sx, sy = config["cloth"]["size_m"]
        self._gx, self._gy = gx, gy
        self._n = gx * gy
        self._size = (sx, sy)
        self.topology = build_topology(gx, gy, sx, sy)

        # Material -> spring stiffness
        E = float(config["material"]["young_modulus_warp_pa"])
        thickness = float(config["cloth"]["thickness_m"])
        # Membrane analogy: k = E * thickness, units N/m. This is the structural stiffness.
        self._k_structural = E * thickness
        self._k_shear = self._k_structural * float(config["implicit"]["shear_stiffness_ratio"])
        self._k_bend = self._k_structural * float(config["implicit"]["bend_stiffness_ratio"])
        self._k_d_factor = float(config["implicit"]["damping_per_unit_stiffness"])

        # Mass per particle
        self._mass_per = float(config["cloth"]["mass_kg"]) / self._n
        self._gravity = np.asarray(config["mpm"]["gravity_m_s2"], dtype=np.float64)
        self._dt = float(config["implicit"]["dt_s"])

        # State
        self.x = np.zeros((self._n, 3), dtype=np.float64)
        self.v = np.zeros((self._n, 3), dtype=np.float64)
        self.pinned = np.zeros(self._n, dtype=bool)

    # -- setup ----------------------------------------------------------------

    def reset(self, x0: np.ndarray | None = None,
              v0: np.ndarray | None = None,
              pinned: list[int] | None = None) -> None:
        if x0 is None:
            # Default: lay flat at the cloth's initial height in xy plane,
            # consistent with mpm_cloth's reset.
            h = float(self.cfg["cloth"]["initial_height_m"])
            sx, sy = self._size
            for i in range(self._gx):
                for j in range(self._gy):
                    self.x[_idx(i, j, self._gy)] = [
                        (i + 0.5) * (sx / self._gx),
                        h,
                        (j + 0.5) * (sy / self._gy),
                    ]
        else:
            assert x0.shape == (self._n, 3)
            self.x[:] = x0

        if v0 is None:
            self.v[:] = 0.0
        else:
            assert v0.shape == (self._n, 3)
            self.v[:] = v0

        self.pinned[:] = False
        if pinned:
            for p in pinned:
                self.pinned[p] = True

    # -- forces and Jacobians -------------------------------------------------

    def _accumulate_forces(self):
        """Compute total internal force `f` (N, 3) and assemble the sparse
        Jacobians df/dx and df/dv as scipy.sparse.csr_matrix of shape (3N, 3N).

        The block layout is per-particle (3 dofs each), interleaved.
        """
        n = self._n
        f_total = np.zeros((n, 3), dtype=np.float64)

        rows_x, cols_x, vals_x = [], [], []
        rows_v, cols_v, vals_v = [], [], []

        for family, k in [
            ("structural", self._k_structural),
            ("shear",      self._k_shear),
            ("bend",       self._k_bend),
        ]:
            pairs = getattr(self.topology, f"{family}_pairs")
            if pairs.size == 0:
                continue
            rest = self.topology.rest_lengths[family]
            k_d = self._k_d_factor * k
            pi, pj, K_ij_x, K_ij_v, f = _spring_forces_and_blocks(
                pairs, rest, k, k_d, self.x, self.v
            )
            f_total += f

            # Each spring contributes 4 3x3 blocks: (i,i)=-K, (i,j)=+K, (j,i)=+K, (j,j)=-K
            for ai, aj, K in [(pi, pj, +K_ij_x), (pj, pi, +K_ij_x),
                              (pi, pi, -K_ij_x), (pj, pj, -K_ij_x)]:
                # dense triplet expansion (3x3 blocks)
                base_r = (ai * 3)[:, None, None] + np.arange(3)[None, :, None]
                base_c = (aj * 3)[:, None, None] + np.arange(3)[None, None, :]
                rows_x.append(np.broadcast_to(base_r, K.shape).reshape(-1))
                cols_x.append(np.broadcast_to(base_c, K.shape).reshape(-1))
                vals_x.append(K.reshape(-1))
            for ai, aj, K in [(pi, pj, +K_ij_v), (pj, pi, +K_ij_v),
                              (pi, pi, -K_ij_v), (pj, pj, -K_ij_v)]:
                base_r = (ai * 3)[:, None, None] + np.arange(3)[None, :, None]
                base_c = (aj * 3)[:, None, None] + np.arange(3)[None, None, :]
                rows_v.append(np.broadcast_to(base_r, K.shape).reshape(-1))
                cols_v.append(np.broadcast_to(base_c, K.shape).reshape(-1))
                vals_v.append(K.reshape(-1))

        rows_x = np.concatenate(rows_x); cols_x = np.concatenate(cols_x); vals_x = np.concatenate(vals_x)
        rows_v = np.concatenate(rows_v); cols_v = np.concatenate(cols_v); vals_v = np.concatenate(vals_v)

        df_dx = sp.csr_matrix((vals_x, (rows_x, cols_x)), shape=(3 * n, 3 * n))
        df_dv = sp.csr_matrix((vals_v, (rows_v, cols_v)), shape=(3 * n, 3 * n))
        return f_total, df_dx, df_dv

    # -- modified PCG ---------------------------------------------------------

    def _filter(self, w: np.ndarray) -> np.ndarray:
        """Per-particle filter: zero out the velocity-change components on
        pinned particles. With S_i = 0 for pinned, S_i = I otherwise."""
        out = w.copy()
        if self.pinned.any():
            out[self.pinned] = 0.0
        return out

    def _modified_pcg(self, A: sp.csr_matrix, b: np.ndarray,
                      max_iters: int, tol: float) -> tuple[np.ndarray, int]:
        """Modified PCG (Baraff-Witkin Algorithm). Returns (delta_v, iterations).

        b shape: (N, 3)  (we operate on per-particle vectors)
        A is (3N, 3N) sparse but we apply it as A @ vec(w) where vec is
        row-major flattening; b and the iterates are kept (N, 3) for clarity.
        """
        n = self._n
        # Diagonal preconditioner from A
        diag = A.diagonal().reshape(n, 3)
        diag_safe = np.where(np.abs(diag) > 1e-12, diag, 1.0)
        Pinv = 1.0 / diag_safe

        # Initialize delta_v = z (zero, no prescribed velocity changes here)
        dv = np.zeros((n, 3), dtype=np.float64)
        # r = filter(b - A dv)
        r = self._filter(b - (A @ dv.reshape(-1)).reshape(n, 3))
        c = self._filter(Pinv * r)
        s_new = float((r * c).sum())
        s0 = max(s_new, 1e-30)
        for it in range(max_iters):
            if s_new / s0 < tol * tol:
                return dv, it
            q = self._filter((A @ c.reshape(-1)).reshape(n, 3))
            alpha = s_new / max(float((c * q).sum()), 1e-30)
            dv = dv + alpha * c
            r = r - alpha * q
            s = self._filter(Pinv * r)
            s_old = s_new
            s_new = float((r * s).sum())
            beta = s_new / max(s_old, 1e-30)
            c = self._filter(s + beta * c)
        return dv, max_iters

    # -- public step ----------------------------------------------------------

    def step(self,
             f_ext: np.ndarray | None = None,
             dt: float | None = None,
             cg_max_iters: int = 50,
             cg_tol: float = 1e-4) -> dict[str, Any]:
        """One implicit Euler step.

        f_ext: external per-particle force (N, 3). If None, gravity is applied.
               If provided, it should already include all external loading
               (gravity, wind, collision response) for this step.
        """
        if dt is None:
            dt = self._dt
        n = self._n
        h = dt
        m = self._mass_per

        f_int, df_dx, df_dv = self._accumulate_forces()
        if f_ext is None:
            f_ext = np.broadcast_to(m * self._gravity, (n, 3)).copy()
        f0 = f_int + f_ext

        # A = M - h df/dv - h^2 df/dx
        I = sp.eye(3 * n, format="csr")
        A = m * I - h * df_dv - (h * h) * df_dx
        # b = h * (f0 + h * df/dx * v)
        b = h * (f0 + h * (df_dx @ self.v.reshape(-1)).reshape(n, 3))
        b = self._filter(b)

        dv, iters = self._modified_pcg(A.tocsr(), b, cg_max_iters, cg_tol)
        # Apply: pinned particles stay put (dv already 0 for them)
        self.v = self.v + dv
        self.x = self.x + h * self.v
        return {"cg_iters": iters,
                "max_dv": float(np.linalg.norm(dv, axis=-1).max())}

    def state(self) -> dict[str, np.ndarray]:
        return {
            "x": self.x.copy(),
            "v": self.v.copy(),
            "pinned": self.pinned.copy(),
        }

    def kinetic_energy(self) -> float:
        return 0.5 * float(self._mass_per) * float((self.v * self.v).sum())
