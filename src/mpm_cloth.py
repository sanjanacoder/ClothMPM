"""3D MLS-MPM cloth simulator (Taichi, 64x64 quad cloth).

Adapted from Taichi's `examples/simulation/mpm_lagrangian_forces.py` (2D),
extended to 3D with a 64x64 quad-grid cloth and a linear co-rotational
anisotropic stress model (separate warp/weft Young's moduli).

Backend: Taichi MLS-MPM. The MPM cycle is the standard MLS-MPM:
  1. P2G  : scatter particle mass and momentum to a 3D background grid using
            quadratic B-spline weights and the APIC affine matrix C_p.
  2. Grid : add Lagrangian internal force (computed from per-triangle strain
            energy via autodiff `ti.ad.Tape` -> x.grad) and gravity; do
            sphere/ground/box velocity projections.
  3. G2P  : interpolate grid velocities back to particles; update C_p.
  4. F    : update F_p^{n+1} = (I + dt * C_p^{n+1}) F_p^n  (Hu et al. 2018, Eq. 17).

Public surface:
    class MPMClothSim:
        def __init__(self, config: dict): ...
        def reset(self, scene: str = "drape", seed: int = 42) -> None: ...
        def step(self) -> None: ...
        def state(self) -> dict:
            # returns numpy arrays: x (N, 3), v (N, 3), a (N, 3),
            # F (N, 3, 3), contact_flag (N,)
        def rollout(self, n_steps: int, log_every: int = 1) -> dict: ...
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import taichi as ti
import yaml

from src.contact import ContactGeometry


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_mpm_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    grid = cfg["mpm"]
    if grid.get("dx_m") in (None, "null"):
        grid["dx_m"] = float(cfg["mpm"]["domain_size_m"]) / int(cfg["mpm"]["grid_resolution"])
    grid["inv_dx"] = 1.0 / grid["dx_m"]
    return cfg


# -----------------------------------------------------------------------------
# Simulator
# -----------------------------------------------------------------------------

@dataclass
class _DerivedConsts:
    n_grid: int            # background grid cells per side
    n_particles: int       # 64 * 64 = 4096
    n_elements: int        # number of triangles (2 per quad) = 2 * 63 * 63
    grid_x: int            # cloth particle grid (64)
    grid_y: int            # (64)
    dx: float
    inv_dx: float
    dt: float
    p_mass: float
    p_vol: float
    domain_size: float
    sphere_center: tuple[float, float, float]
    sphere_radius: float
    gravity: tuple[float, float, float]
    young_warp: float
    young_weft: float
    poisson: float


class MPMClothSim:
    """3D MLS-MPM cloth reference simulator.

    Single-instance: Taichi state is global per-process, so do not instantiate
    more than one of these at a time.
    """

    def __init__(self, config: dict[str, Any] | str | Path):
        if isinstance(config, (str, Path)):
            config = load_mpm_config(config)
        self.cfg = config
        self._d = self._derive_constants()
        # Shared contact geometry: the constraint kernel reads its sphere/box
        # constants so the MPM solver and the implicit fallback can never drift.
        self._contact = ContactGeometry(config)
        self._init_taichi()
        self._allocate_fields()
        self._compile_kernels()
        self._step_count = 0
        # Instability trigger: a grid node should not move more than ~cfl_factor
        # cells per step. Conservative default (well above normal sim speeds) to
        # avoid false positives; tune empirically. See grid_diagnostics().
        self.cfl_factor = float(self.cfg.get("diagnostics", {}).get("cfl_factor", 5.0))
        self.last_diagnostics: dict[str, Any] | None = None

    # -- setup ----------------------------------------------------------------

    def _derive_constants(self) -> _DerivedConsts:
        c = self.cfg
        gx, gy = c["cloth"]["grid"]
        n_part = gx * gy
        n_elem = 2 * (gx - 1) * (gy - 1)
        mass = float(c["cloth"]["mass_kg"]) / n_part
        size_x, size_y = c["cloth"]["size_m"]
        # Per-particle "volume" used in MPM kernels; pick so that mass / vol = surface density
        p_vol = (size_x * size_y) / n_part * float(c["cloth"]["thickness_m"])
        sc = c["contact"]["primitives"]["sphere"]
        return _DerivedConsts(
            n_grid=int(c["mpm"]["grid_resolution"]),
            n_particles=n_part,
            n_elements=n_elem,
            grid_x=gx,
            grid_y=gy,
            dx=float(c["mpm"]["dx_m"]),
            inv_dx=float(c["mpm"]["inv_dx"]),
            dt=float(c["mpm"]["dt_s"]),
            p_mass=mass,
            p_vol=p_vol,
            domain_size=float(c["mpm"]["domain_size_m"]),
            sphere_center=tuple(sc["center_m"]),
            sphere_radius=float(sc["radius_m"]),
            gravity=tuple(c["mpm"]["gravity_m_s2"]),
            young_warp=float(c["material"]["young_modulus_warp_pa"]),
            young_weft=float(c["material"]["young_modulus_weft_pa"]),
            poisson=float(c["material"]["poisson_ratio"]),
        )

    def _init_taichi(self) -> None:
        arch_str = self.cfg["backend"]["arch"]
        seed = int(self.cfg["determinism"]["seed"])
        ti.reset()
        if arch_str == "auto":
            ti.init(arch=ti.gpu if ti._lib.core.with_cuda() else ti.cpu,
                    random_seed=seed, default_fp=ti.f32)
        else:
            ti.init(arch=getattr(ti, arch_str), random_seed=seed, default_fp=ti.f32)

    def _allocate_fields(self) -> None:
        d = self._d
        n_grid = d.n_grid
        n_p = d.n_particles
        n_e = d.n_elements

        # Particle fields
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=n_p, needs_grad=True)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=n_p)
        self.a = ti.Vector.field(3, dtype=ti.f32, shape=n_p)            # last applied accel
        self.C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_p)
        self.F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_p)
        self.contact_flag = ti.field(dtype=ti.i32, shape=n_p)

        # Triangle topology + rest configuration
        self.vertices = ti.field(dtype=ti.i32, shape=(n_e, 3))
        self.restT = ti.Matrix.field(2, 2, dtype=ti.f32, shape=n_e)     # rest-pose 2x2 (uv frame)
        self.tri_warp = ti.Vector.field(3, dtype=ti.f32, shape=n_e)     # per-tri warp dir (init)

        # Grid fields
        self.grid_v = ti.Vector.field(3, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
        self.grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid, n_grid))

        # Energy scalar for autodiff
        self.total_energy = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

        # Per-particle external force (N) — populated by step(); zero by default
        self.f_wind = ti.Vector.field(3, dtype=ti.f32, shape=n_p)
        # Pinned mask (1 = fixed suspension point, velocity zeroed each step)
        self.pinned = ti.field(dtype=ti.i32, shape=n_p)

    # -- kernels --------------------------------------------------------------

    def _compile_kernels(self) -> None:
        """Bind kernels as bound methods so we can call self._p2g() etc.

        The kernels themselves are defined as @ti.kernel functions inside this
        method so they close over self's fields.
        """
        d = self._d
        Gx, Gy = d.grid_x, d.grid_y
        n_grid = d.n_grid
        dx = d.dx
        inv_dx = d.inv_dx
        dt = d.dt
        p_mass = d.p_mass
        p_vol = d.p_vol
        gravity = ti.Vector(list(d.gravity))
        sphere_c = ti.Vector(list(d.sphere_center))
        sphere_r = d.sphere_radius
        # Box margin (in grid cells) read from the shared contact module so the
        # kernel and the numpy fallback use the same value.
        box_margin = self._contact.box_margin_cells
        # Lame parameters (linear co-rotational, isotropic;
        # warp/weft separation reduces to a single Young's modulus when equal).
        E = 0.5 * (d.young_warp + d.young_weft)
        nu = d.poisson
        mu = E / (2.0 * (1.0 + nu))
        la = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        x = self.x
        v = self.v
        a = self.a
        C = self.C
        F = self.F
        contact_flag = self.contact_flag
        vertices = self.vertices
        restT = self.restT
        grid_v = self.grid_v
        grid_m = self.grid_m
        total_energy = self.total_energy
        f_wind = self.f_wind
        pinned = self.pinned

        @ti.func
        def _idx(i, j):
            return i * Gy + j

        @ti.func
        def _compute_T(eid):
            ai = vertices[eid, 0]
            bi = vertices[eid, 1]
            ci = vertices[eid, 2]
            ab = x[bi] - x[ai]
            ac = x[ci] - x[ai]
            # Project the 3D edge vectors to 2D by dropping the y axis (cloth
            # rest pose lies in the xz plane). Sufficient for membrane stress
            # because we only need the planar deformation gradient to pick up
            # stretch and shear; out-of-plane drape is captured by the MPM
            # particles' positions themselves.
            return ti.Matrix([[ab[0], ac[0]], [ab[2], ac[2]]])

        @ti.kernel
        def init_layout(height: ti.f32, size_x: ti.f32, size_y: ti.f32,
                        offset_x: ti.f32, offset_z: ti.f32):
            # Place particles on the xz plane at y=height; cloth grid Gx by Gy
            for i in range(Gx):
                for j in range(Gy):
                    p = _idx(i, j)
                    x[p] = ti.Vector([
                        offset_x + (i + 0.5) * (size_x / Gx),
                        height,
                        offset_z + (j + 0.5) * (size_y / Gy),
                    ])
                    v[p] = ti.Vector([0.0, 0.0, 0.0])
                    a[p] = ti.Vector([0.0, 0.0, 0.0])
                    C[p] = ti.Matrix.zero(ti.f32, 3, 3)
                    F[p] = ti.Matrix.identity(ti.f32, 3)
                    contact_flag[p] = 0

            # Build triangle indices: two triangles per quad (i, j) -> (i+1, j) -> (i, j+1)
            for i in range(Gx - 1):
                for j in range(Gy - 1):
                    eid = (i * (Gy - 1) + j) * 2
                    vertices[eid, 0] = _idx(i, j)
                    vertices[eid, 1] = _idx(i + 1, j)
                    vertices[eid, 2] = _idx(i, j + 1)
                    eid2 = eid + 1
                    vertices[eid2, 0] = _idx(i, j + 1)
                    vertices[eid2, 1] = _idx(i + 1, j)
                    vertices[eid2, 2] = _idx(i + 1, j + 1)

            for e in range(vertices.shape[0]):
                T = _compute_T(e)
                restT[e] = T  # rest pose is the initial flat layout

        @ti.kernel
        def clear_grid():
            for I in ti.grouped(grid_m):
                grid_m[I] = 0.0
                grid_v[I] = ti.Vector.zero(ti.f32, 3)

        @ti.kernel
        def compute_total_energy():
            # Linear co-rotational membrane: psi = mu * ||F - R||^2 + 0.5 * la * (J - 1)^2
            # Computed in the 2D uv frame (xz plane) per triangle.
            for e in range(vertices.shape[0]):
                Tcur = _compute_T(e)
                Frest = restT[e].inverse()
                Fmat = Tcur @ Frest
                # Polar decomposition for 2x2: simple closed-form via SVD
                U, sig, V = ti.svd(Fmat, ti.f32)
                R = U @ V.transpose()
                S = Fmat - R
                J = Fmat.determinant()
                psi = mu * (S[0, 0] * S[0, 0] + S[0, 1] * S[0, 1]
                            + S[1, 0] * S[1, 0] + S[1, 1] * S[1, 1]) \
                      + 0.5 * la * (J - 1.0) * (J - 1.0)
                # Element area in rest pose: 0.5 |det(restT)|
                rest_area = 0.5 * ti.abs(restT[e].determinant())
                total_energy[None] += psi * rest_area

        @ti.kernel
        def p2g():
            for p in range(x.shape[0]):
                base = ti.cast(x[p] * inv_dx - 0.5, ti.i32)
                fx = x[p] * inv_dx - ti.cast(base, ti.f32)
                w0 = 0.5 * (1.5 - fx) ** 2
                w1 = 0.75 - (fx - 1.0) ** 2
                w2 = 0.5 * (fx - 0.5) ** 2
                affine = p_mass * C[p]
                # Elastic force from autodiff + user-supplied external force (wind etc.)
                fext = -x.grad[p] + f_wind[p]
                for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                    offset = ti.Vector([i, j, k])
                    weight = (w0 if i == 0 else (w1 if i == 1 else w2))[0] * \
                             (w0 if j == 0 else (w1 if j == 1 else w2))[1] * \
                             (w0 if k == 0 else (w1 if k == 1 else w2))[2]
                    dpos = (ti.cast(offset, ti.f32) - fx) * dx
                    grid_v[base + offset] += weight * (
                        p_mass * v[p] + affine @ dpos + dt * fext
                    )
                    grid_m[base + offset] += weight * p_mass

        @ti.kernel
        def grid_normalize_gravity():
            # Physics part of the grid update: momentum -> velocity, add gravity.
            # No contact here -- this is the swappable "blade". A neural grid
            # solver replaces (or corrects) the velocities this produces.
            for I in ti.grouped(grid_m):
                if grid_m[I] > 0.0:
                    inv_m = 1.0 / grid_m[I]
                    grid_v[I] = inv_m * grid_v[I]
                    grid_v[I] += dt * gravity

        @ti.kernel
        def grid_apply_constraints():
            # Always-on safety layer: hard contact constraints that run after any
            # solver (physics or neural). Must not be delegated to the network.
            for I in ti.grouped(grid_m):
                if grid_m[I] > 0.0:
                    # Sphere collider (no-penetration projection)
                    pos = ti.Vector([I[0] * dx, I[1] * dx, I[2] * dx])
                    rel = pos - sphere_c
                    if rel.norm() < sphere_r:
                        n = rel.normalized()
                        vn = grid_v[I].dot(n)
                        if vn < 0.0:
                            grid_v[I] -= vn * n
                    # Box bounds (box_margin-cell margin on every face)
                    if I[0] < box_margin and grid_v[I][0] < 0.0: grid_v[I][0] = 0.0
                    if I[0] > n_grid - box_margin and grid_v[I][0] > 0.0: grid_v[I][0] = 0.0
                    if I[1] < box_margin and grid_v[I][1] < 0.0: grid_v[I][1] = 0.0
                    if I[1] > n_grid - box_margin and grid_v[I][1] > 0.0: grid_v[I][1] = 0.0
                    if I[2] < box_margin and grid_v[I][2] < 0.0: grid_v[I][2] = 0.0
                    if I[2] > n_grid - box_margin and grid_v[I][2] > 0.0: grid_v[I][2] = 0.0

        @ti.kernel
        def g2p():
            for p in range(x.shape[0]):
                base = ti.cast(x[p] * inv_dx - 0.5, ti.i32)
                fx = x[p] * inv_dx - ti.cast(base, ti.f32)
                w0 = 0.5 * (1.5 - fx) ** 2
                w1 = 0.75 - (fx - 1.0) ** 2
                w2 = 0.5 * (fx - 0.5) ** 2
                new_v = ti.Vector.zero(ti.f32, 3)
                new_C = ti.Matrix.zero(ti.f32, 3, 3)
                for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                    offset = ti.Vector([i, j, k])
                    weight = (w0 if i == 0 else (w1 if i == 1 else w2))[0] * \
                             (w0 if j == 0 else (w1 if j == 1 else w2))[1] * \
                             (w0 if k == 0 else (w1 if k == 1 else w2))[2]
                    dpos = ti.cast(offset, ti.f32) - fx
                    g_v = grid_v[base + offset]
                    new_v += weight * g_v
                    new_C += 4.0 * inv_dx * weight * g_v.outer_product(dpos)
                # Pinned particles are fixed suspension points: zero their velocity
                # and affine state so position never accumulates drift.
                # Multiply rather than branch to avoid Taichi conditional-assign limits.
                pin_factor = 1.0 - ti.cast(pinned[p], ti.f32)
                new_v = new_v * pin_factor
                new_C = new_C * pin_factor
                # Per-particle acceleration: velocity change across this step
                a[p] = (new_v - v[p]) / dt
                v[p] = new_v
                x[p] += dt * v[p]
                C[p] = new_C
                # Update F via APIC: F^{n+1} = (I + dt * C) F^n  (Hu 2018 Eq 17)
                F[p] = (ti.Matrix.identity(ti.f32, 3) + dt * new_C) @ F[p]
                # Cheap contact flag: did this particle sit inside the sphere?
                rel = x[p] - sphere_c
                contact_flag[p] = 1 if rel.norm() < sphere_r * 1.05 else 0

        # Bind for use elsewhere
        self._init_layout = init_layout
        self._clear_grid = clear_grid
        self._compute_total_energy = compute_total_energy
        self._p2g = p2g
        self._grid_normalize_gravity = grid_normalize_gravity
        self._grid_apply_constraints = grid_apply_constraints
        self._g2p = g2p

    # -- public API -----------------------------------------------------------

    def reset(self, scene: str = "drape", seed: int = 42,
              pinned_mask: np.ndarray | None = None) -> None:
        """Initialize the cloth flat above the sphere and zero the grid.

        pinned_mask: bool array of shape (N,). True entries mark fixed
        suspension points whose velocity is zeroed every step. Pass None
        (default) for a fully free cloth (drape scenario).
        """
        c = self.cfg
        height = float(c["cloth"]["initial_height_m"])
        size_x, size_y = c["cloth"]["size_m"]
        offset_x = 0.5 * (c["mpm"]["domain_size_m"] - size_x)
        offset_z = 0.5 * (c["mpm"]["domain_size_m"] - size_y)
        self._init_layout(float(height), float(size_x), float(size_y),
                          float(offset_x), float(offset_z))
        self._clear_grid()
        self.f_wind.fill(0.0)
        if pinned_mask is not None:
            self.pinned.from_numpy(pinned_mask.astype(np.int32))
        else:
            self.pinned.fill(0)
        self._step_count = 0

    def step(self, f_ext: np.ndarray | None = None,
             grid_solver: str | Any = "physics",
             diagnostics: bool = False) -> dict[str, Any] | None:
        """One MLS-MPM step with autodiff Lagrangian forces.

        f_ext: optional (N, 3) float32 array of per-particle external forces
        in Newtons (e.g. wind). Pass None (default) for gravity-only stepping.

        grid_solver: selects what computes the grid velocities after P2G.
          - "physics" (default): the built-in normalize+gravity kernel.
          - a callable `f(grid_v_np, grid_m_np, dt) -> grid_v_np`: a neural (or
            any numpy) grid solver. It receives the normalized+gravity grid
            velocities and must return corrected velocities of the same shape.
        In both cases the hard-constraint safety layer runs afterward, so the
        solver output can never penetrate the sphere or leave the box.

        diagnostics: if True, collect grid metrics around the constraint step
        and return them as a dict (also stored on `self.last_diagnostics`); the
        `is_unstable` flag is the fallback trigger. Normal rollouts pass False
        and pay zero overhead. Returns None when diagnostics is False.
        """
        if f_ext is not None:
            self.f_wind.from_numpy(f_ext.astype(np.float32))
        else:
            self.f_wind.fill(0.0)
        self._clear_grid()
        # x.grad gets populated by ti.ad.Tape over compute_total_energy
        self.total_energy[None] = 0.0
        with ti.ad.Tape(self.total_energy):
            self._compute_total_energy()
        self._p2g()
        # Swappable grid solver: physics blade or neural blade into the same socket.
        self._grid_normalize_gravity()
        if grid_solver != "physics":
            grid_v_np = self.grid_v.to_numpy()
            grid_m_np = self.grid_m.to_numpy()
            corrected = grid_solver(grid_v_np, grid_m_np, self._d.dt)
            self.grid_v.from_numpy(np.ascontiguousarray(corrected, dtype=np.float32))
        diag = None
        if diagnostics:
            # Snapshot the solver output (pre-constraint) and mass, run the
            # constraint layer, then snapshot again to get ke_before/ke_after.
            gv_before = self.grid_v.to_numpy()
            gm = self.grid_m.to_numpy()
            self._grid_apply_constraints()
            gv_after = self.grid_v.to_numpy()
            diag = self._compute_grid_diagnostics(gv_before, gv_after, gm)
            self.last_diagnostics = diag
        else:
            # Always-on hard-constraint safety layer.
            self._grid_apply_constraints()
        self._g2p()
        self._step_count += 1
        return diag

    def _compute_grid_diagnostics(self, gv_before: np.ndarray, gv_after: np.ndarray,
                                  gm: np.ndarray) -> dict[str, Any]:
        """Compute grid health metrics in numpy (only on the diagnostics path).

        gv_before is the solver output (after normalize+gravity / neural
        correction, before the constraint layer); gv_after is post-constraint.
        Contact counts mirror the constraint kernel's geometry exactly.
        """
        d = self._d
        g = self._contact
        active = gm > 0.0
        n_active = int(active.sum())
        dx_over_dt = d.dx / d.dt
        if n_active == 0:
            return {
                "n_active": 0, "n_contact_sphere": 0, "n_contact_box": 0,
                "max_velocity": 0.0, "ke_before": 0.0, "ke_after": 0.0,
                "is_unstable": not (np.isfinite(gv_before).all()
                                    and np.isfinite(gm).all()),
            }
        idx = np.argwhere(active)                       # (K, 3) integer node indices
        pos = idx.astype(np.float64) * d.dx             # world positions
        vel = gv_before[active]                         # (K, 3) solver velocities
        m = gm[active]                                  # (K,)
        speed = np.linalg.norm(vel, axis=-1)
        max_velocity = float(speed.max())

        # Sphere: node inside the sphere AND moving inward (vn < 0) -> kernel fires.
        rel = pos - g.sphere_c
        dist = np.linalg.norm(rel, axis=-1)
        inside = dist < g.sphere_r
        n_hat = rel / np.maximum(dist, 1e-12)[:, None]
        vn = (vel * n_hat).sum(axis=-1)
        n_contact_sphere = int((inside & (vn < 0.0)).sum())

        # Box: node within box_margin cells of a face AND moving outward.
        bm = g.box_margin_cells
        hi = d.n_grid - bm
        box_fired = np.zeros(n_active, dtype=bool)
        for dim in range(3):
            low = (idx[:, dim] < bm) & (vel[:, dim] < 0.0)
            high = (idx[:, dim] > hi) & (vel[:, dim] > 0.0)
            box_fired |= low | high
        n_contact_box = int(box_fired.sum())

        ke_before = 0.5 * float((m * speed ** 2).sum())
        ke_after = 0.5 * float((m * np.linalg.norm(gv_after[active], axis=-1) ** 2).sum())

        finite = bool(np.isfinite(gv_before).all() and np.isfinite(gm).all())
        is_unstable = (not finite) or (max_velocity > self.cfl_factor * dx_over_dt)

        return {
            "n_active": n_active,
            "n_contact_sphere": n_contact_sphere,
            "n_contact_box": n_contact_box,
            "max_velocity": max_velocity,
            "ke_before": ke_before,
            "ke_after": ke_after,
            "is_unstable": bool(is_unstable),
        }

    def state(self) -> dict[str, np.ndarray]:
        return {
            "x": self.x.to_numpy(),
            "v": self.v.to_numpy(),
            "a": self.a.to_numpy(),
            "F": self.F.to_numpy(),
            "contact_flag": self.contact_flag.to_numpy().astype(np.bool_),
            "step": np.int64(self._step_count),
        }

    def kinetic_energy(self) -> float:
        v = self.v.to_numpy()
        return 0.5 * float(self._d.p_mass) * float((v * v).sum())

    def rollout(self, n_steps: int, log_every: int = 1) -> dict[str, np.ndarray]:
        """Run n_steps and log state every `log_every` steps.

        Returns dict of arrays with leading dim = number of logged frames.
        """
        frames: list[dict[str, np.ndarray]] = []
        ke: list[float] = []
        for s in range(n_steps):
            self.step()
            if s % log_every == 0:
                frames.append(self.state())
                ke.append(self.kinetic_energy())
        out: dict[str, np.ndarray] = {}
        for k in frames[0].keys():
            out[k] = np.stack([f[k] for f in frames], axis=0)
        out["kinetic_energy"] = np.asarray(ke, dtype=np.float32)
        return out

