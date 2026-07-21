"""GPU (torch) implicit mass-spring cloth solver -- a matrix-free port of
`src.cloth_implicit.ImplicitClothSim`.

The CPU version assembles the (3N x 3N) sparse implicit-Euler system and runs a
modified PCG. The dominant cost there is scipy's single-threaded CG. Here we keep
the *identical* math (same spring forces + Baraff-Witkin Jacobians with the
definiteness clamp, same modified PCG, same contact) but never assemble the
matrix: the CG only needs A @ w, which we evaluate matrix-free per spring as
gather(w) -> 3x3 einsum -> scatter-add. That maps to a few parallel torch ops per
CG iteration and runs entirely on the GPU.

Interface mirrors ImplicitClothSim (`reset`, `step(f_ext, dt, cg_max_iters,
cg_tol)`, `.x`, `.v`) so it can drop into HybridRollout as `fallback_sim`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.cloth_implicit import build_topology, load_implicit_config
from src.contact import ContactGeometry


class GPUImplicitClothSim:
    def __init__(self, config: dict[str, Any] | str | Path, device: str = "cuda"):
        if not isinstance(config, dict):
            config = load_implicit_config(config)
        self.cfg = config
        self.device = torch.device(device)
        d = self.device
        gx, gy = config["cloth"]["grid"]
        sx, sy = config["cloth"]["size_m"]
        self._n = int(gx) * int(gy)
        topo = build_topology(int(gx), int(gy), float(sx), float(sy))

        E = float(config["material"]["young_modulus_warp_pa"])
        th = float(config["cloth"]["thickness_m"])
        k_struct = E * th
        k_by_family = {
            "structural": k_struct,
            "shear": k_struct * float(config["implicit"]["shear_stiffness_ratio"]),
            "bend": k_struct * float(config["implicit"]["bend_stiffness_ratio"]),
        }
        kd_factor = float(config["implicit"]["damping_per_unit_stiffness"])
        pairs, rest, kk = [], [], []
        for fam, k in k_by_family.items():
            p = getattr(topo, f"{fam}_pairs")
            r = topo.rest_lengths[fam]
            if p.size:
                pairs.append(p.astype(np.int64))
                rest.append(r)
                kk.append(np.full(len(r), k))
        pairs = np.concatenate(pairs)
        self.pi = torch.as_tensor(pairs[:, 0], device=d)
        self.pj = torch.as_tensor(pairs[:, 1], device=d)
        self.L0 = torch.as_tensor(np.concatenate(rest), dtype=torch.float32, device=d)
        self.k = torch.as_tensor(np.concatenate(kk), dtype=torch.float32, device=d)
        self.kd = self.k * kd_factor

        self.m = float(config["cloth"]["mass_kg"]) / self._n
        self.gravity = torch.as_tensor(config["mpm"]["gravity_m_s2"], dtype=torch.float32, device=d)
        self._dt = float(config["implicit"]["dt_s"])

        cg = ContactGeometry(config)
        self.sphere_enabled = bool(cg.sphere_enabled)
        self.sphere_c = torch.as_tensor(cg.sphere_c, dtype=torch.float32, device=d)
        self.sphere_r = float(cg.sphere_r)
        self.ground_y = float(cg.ground_y)
        self.margin = float(cg.margin)
        self.domain = float(cg.domain)

        self.eye = torch.eye(3, device=d)
        self.x = torch.zeros(self._n, 3, device=d)
        self.v = torch.zeros(self._n, 3, device=d)
        self.pinned = torch.zeros(self._n, dtype=torch.bool, device=d)
        self.keep = torch.ones(self._n, 1, device=d)   # 0 on pinned (mask filter)

    # -- setup ---------------------------------------------------------------
    def reset(self, x0=None, v0=None, pinned=None) -> None:
        if x0 is not None:
            self.x = torch.as_tensor(x0, dtype=torch.float32, device=self.device).clone()
        self.v = (torch.zeros_like(self.x) if v0 is None else
                  torch.as_tensor(v0, dtype=torch.float32, device=self.device).clone())
        self.pinned = torch.zeros(self._n, dtype=torch.bool, device=self.device)
        if pinned is not None and len(pinned):
            self.pinned[torch.as_tensor(list(pinned), dtype=torch.long, device=self.device)] = True
        self.keep = (~self.pinned).float()[:, None]

    # -- matrix-free pieces --------------------------------------------------
    def _scatter(self, g: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(self._n, 3, device=self.device, dtype=g.dtype)
        out.index_add_(0, self.pi, g)
        out.index_add_(0, self.pj, -g)
        return out

    def _filter(self, w: torch.Tensor) -> torch.Tensor:
        # Mask-multiply (no clone / no boolean scatter) so the CG loop is a static
        # graph of elementwise+gather/scatter ops -> torch.compile / CUDA-graph safe.
        return w * self.keep

    def _springs(self):
        e = self.x[self.pj] - self.x[self.pi]
        L = e.norm(dim=-1)
        Ls = L.clamp_min(1e-12)
        eh = e / Ls[:, None]
        stretch = L - self.L0
        f_elastic = (self.k * stretch)[:, None] * eh
        rel_v = self.v[self.pj] - self.v[self.pi]
        f_damp = (self.kd * (rel_v * eh).sum(-1))[:, None] * eh
        f_spring = f_elastic + f_damp
        eet = eh[:, :, None] * eh[:, None, :]
        # definiteness clamp on the (L-L0)/L projection term (Baraff-Witkin 5.3)
        fac = (self.k * stretch.clamp_min(0.0) / Ls)[:, None, None]
        Kx = fac * (self.eye - eet) + self.k[:, None, None] * eet
        Kv = self.kd[:, None, None] * eet
        return f_spring, Kx, Kv

    @torch.no_grad()
    def solve_dv(self, x: torch.Tensor, v: torch.Tensor, h: float,
                 n_iters: int) -> torch.Tensor:
        """Fixed-iteration implicit-Euler PCG solve for the velocity change dv,
        as a pure function of (x, v). No early-out and no contact projection, so
        the whole thing is a static graph -> torch.compile / CUDA-graph safe.
        step() uses the eager (early-out + contact) path; this is for the fast
        compiled fallback."""
        m = self.m
        e = x[self.pj] - x[self.pi]
        L = e.norm(dim=-1)
        Ls = L.clamp_min(1e-12)
        eh = e / Ls[:, None]
        stretch = L - self.L0
        f_spring = ((self.k * stretch)[:, None] * eh
                    + (self.kd * ((v[self.pj] - v[self.pi]) * eh).sum(-1))[:, None] * eh)
        eet = eh[:, :, None] * eh[:, None, :]
        Kx = (self.k * stretch.clamp_min(0.0) / Ls)[:, None, None] * (self.eye - eet) \
            + self.k[:, None, None] * eet
        Kv = self.kd[:, None, None] * eet

        def apply_K(K, w):
            return self._scatter(torch.einsum("sij,sj->si", K, w[self.pj] - w[self.pi]))

        def matvec(w):
            return m * w - h * apply_K(Kv, w) - (h * h) * apply_K(Kx, w)

        f0 = self._scatter(f_spring) + m * self.gravity.expand(self._n, 3)
        b = self._filter(h * (f0 + h * apply_K(Kx, v)))
        diagA = torch.full((self._n, 3), m, device=self.device)
        contrib = h * h * torch.diagonal(Kx, dim1=-2, dim2=-1) + h * torch.diagonal(Kv, dim1=-2, dim2=-1)
        diagA.index_add_(0, self.pi, contrib)
        diagA.index_add_(0, self.pj, contrib)
        Pinv = 1.0 / torch.where(diagA.abs() > 1e-12, diagA, torch.ones_like(diagA))

        dv = torch.zeros_like(x)
        r = self._filter(b)
        c = self._filter(Pinv * r)
        s_new = (r * c).sum()
        for _ in range(n_iters):
            q = self._filter(matvec(c))
            alpha = s_new / (c * q).sum().clamp_min(1e-30)
            dv = dv + alpha * c
            r = r - alpha * q
            s = self._filter(Pinv * r)
            s_old = s_new
            s_new = (r * s).sum()
            c = self._filter(s + (s_new / s_old.clamp_min(1e-30)) * c)
        return dv

    def step_fixed(self, n_iters: int = 30, dt: float | None = None,
                   solver=None) -> None:
        """One step using a fixed-iteration solve (optionally a torch.compiled
        `solver(x, v, h, n_iters)`), then eager contact projection."""
        h = self._dt if dt is None else float(dt)
        fn = solver if solver is not None else self.solve_dv
        dv = fn(self.x, self.v, h, n_iters)
        self.v = self.v + dv
        self.x = self.x + h * self.v
        self._contact()

    # -- public step ---------------------------------------------------------
    @torch.no_grad()
    def step(self, f_ext=None, dt: float | None = None,
             cg_max_iters: int = 50, cg_tol: float = 1e-4) -> dict[str, Any]:
        h = self._dt if dt is None else float(dt)
        m = self.m
        f_spring, Kx, Kv = self._springs()

        def apply_K(K, w):
            g = torch.einsum("sij,sj->si", K, w[self.pj] - w[self.pi])
            return self._scatter(g)

        def matvec(w):
            return m * w - h * apply_K(Kv, w) - (h * h) * apply_K(Kx, w)

        f_int = self._scatter(f_spring)
        if f_ext is None:
            f_ext_t = m * self.gravity.expand(self._n, 3)
        else:
            f_ext_t = torch.as_tensor(f_ext, dtype=torch.float32, device=self.device)
        f0 = f_int + f_ext_t
        b = self._filter(h * (f0 + h * apply_K(Kx, self.v)))

        # Jacobi preconditioner: diagonal of A (per-particle 3-vector).
        diagKx = torch.diagonal(Kx, dim1=-2, dim2=-1)
        diagKv = torch.diagonal(Kv, dim1=-2, dim2=-1)
        diagA = torch.full((self._n, 3), m, device=self.device)
        contrib = h * h * diagKx + h * diagKv
        diagA.index_add_(0, self.pi, contrib)
        diagA.index_add_(0, self.pj, contrib)
        Pinv = 1.0 / torch.where(diagA.abs() > 1e-12, diagA, torch.ones_like(diagA))

        dv = torch.zeros_like(self.x)
        r = self._filter(b)                      # b - A@0
        c = self._filter(Pinv * r)
        s_new = (r * c).sum()
        s0 = s_new.clamp_min(1e-30)
        it = 0
        for it in range(cg_max_iters):
            if (s_new / s0) < cg_tol * cg_tol:
                break
            q = self._filter(matvec(c))
            alpha = s_new / (c * q).sum().clamp_min(1e-30)
            dv = dv + alpha * c
            r = r - alpha * q
            s = self._filter(Pinv * r)
            s_old = s_new
            s_new = (r * s).sum()
            c = self._filter(s + (s_new / s_old.clamp_min(1e-30)) * c)

        self.v = self.v + dv
        self.x = self.x + h * self.v
        self._contact()
        return {"cg_iters": it}

    def _contact(self) -> None:
        x, v = self.x, self.v
        if self.sphere_enabled:
            rel = x - self.sphere_c
            dist = rel.norm(dim=-1, keepdim=True)
            inside = (dist < self.sphere_r).squeeze(-1)
            if inside.any():
                n = rel[inside] / dist[inside].clamp_min(1e-12)
                x[inside] = self.sphere_c + n * self.sphere_r
                vn = (v[inside] * n).sum(-1, keepdim=True)
                v[inside] = v[inside] - vn.clamp_max(0.0) * n
        lo, hi = self.margin, self.domain - self.margin
        for dm in range(3):
            below = x[:, dm] < lo
            above = x[:, dm] > hi
            x[below, dm] = lo
            v[below, dm] = v[below, dm].clamp_min(0.0)
            x[above, dm] = hi
            v[above, dm] = v[above, dm].clamp_max(0.0)
        floor = max(self.margin, self.ground_y)
        bl = x[:, 1] < floor
        x[bl, 1] = floor
        v[bl, 1] = v[bl, 1].clamp_min(0.0)
        self.x, self.v = x, v
