# Why the wind scenario is deferred (formal note)

**Decision:** the released dataset covers **drape** and **collision** only. The
wind scenario is deferred to a stretch goal. This note explains why, precisely
enough to cite, and what a real fix would require.

Short version: at production resolution the wind scenario has **no bounded-strain
state the solver can settle into**, so it deforms until a cloth element turns
inside-out, at which point the explicit co-rotational MPM step cannot integrate
and diverges. This is a solver limitation, not a bug in the dataset pipeline, and
it is independent of the machine-level crash noted separately in
`datagen-validation-findings.md`.

---

## 1. The model and the scenario

Each cloth triangle `e` carries an in-plane (2×2) deformation gradient

    F = T_cur · T_rest⁻¹        (maps the rest triangle to its current shape)

with elastic energy density (linear co-rotational membrane)

    ψ(F) = μ ‖F − R‖²_F + (λ/2)(J − 1)²,     J = det F,

where `R` is the rotation from the polar decomposition `F = R·S`. Nodal forces are
`f = −∂E/∂x`, `E = Σ_e ψ(F_e)·A_e`, and the state is advanced with **explicit**
symplectic Euler at `Δt = 1e-4`.

**Wind scenario:** two top corners pinned (`v = 0`), a **constant** body force
`f_wind` on every particle, plus gravity `g`.

## 2. There is no reachable static equilibrium → the sheet keeps deforming

A static rest state needs total force zero everywhere with `v = 0`. Holding a sheet
at only two points under a *constant transverse* load cannot balance that load with
the pin reactions alone — the unconstrained bulk must rotate/translate about the
pin line. We can see this cleanly by adding a linear drag `−k·v` and solving the
per-particle force balance at steady state:

    f_wind − k·m·v_term = 0   ⇒   v_term = f_wind / (k·m)  ≠ 0.

Terminal velocity is **nonzero for any finite drag** `k`. A nonzero steady velocity
means the material never stops moving, so the accumulated strain grows without
bound. (Empirically, with `k = 50` the max speed reached a *perfect plateau*
`v_max ≈ 0.218 m/s` and the cloth still failed — steady speed, not steady shape.)

## 3. Unbounded drift forces element inversion

As the sheet keeps deforming, some triangle's in-plane map degenerates:

    J = det F  →  0⁺   (triangle flattens to a line)   →   J < 0 (folds through itself).

`J < 0` is **inversion**: the element is turned inside-out. Two things then break at
once in an explicit co-rotational solve:

- **The rotation `R` is ill-conditioned.** As `F` approaches singular, the polar/SVD
  factor's sensitivity scales like `1/σ_min → ∞`, so `∂ψ/∂F` — and hence the nodal
  force — becomes huge and erratic. (Replacing the SVD with a regularized closed-form
  polar did **not** help: the issue is the force magnitude near degeneracy, not the
  particular rotation formula.)
- **The model has no inversion recovery.** The linear co-rotational energy is not
  inversion-aware: for `J < 0` its gradient does not correctly push the element back
  out, so the force can drive the fold deeper.

- **Explicit stability collapses.** Explicit elastic MPM is only conditionally stable,
  roughly `Δt ≲ C·Δx·√(ρ/E_eff)`. Near inversion the *effective* stiffness `E_eff → ∞`,
  so the stable timestep bound `→ 0`. Any fixed `Δt` is then too large: the velocity
  update overshoots and amplifies every step, i.e. divergence. A particle leaves the
  grid within a few steps and the run crashes.

## 4. Damping delays but cannot cure it

Because failure is driven by *accumulated strain*, not by speed, reducing the drift
rate only postpones the inevitable. Time-to-inversion `t*` is set by
`∫₀^{t*}(strain rate) dt = ε_crit`; drag lowers the strain rate (`∝ v_term ∝ 1/k`) but
leaves it positive, so `t*` grows with `k` and never becomes infinite:

| drag `k` | `v_term` (4 N wind) | inversion at step |
|---|---|---|
| 0 | ∞ | 860 |
| 8 | ~2.5 | 957 |
| 20 | ~1.0 | 1145 |
| 50 | ~0.2 | 1796 |

Monotone delay, no cure — the signature of "slow the drift" rather than "remove the
cause." Reducing `Δt` (5e-5) and lowering wind magnitude behave the same way: they
delay, they do not prevent.

## 5. A separate, label-level problem: codimensional `F` drift

Even setting the crash aside, the wind clips would ship a **corrupt label**. The
per-particle APIC deformation gradient updates multiplicatively,

    F_p^{n+1} = (I + Δt·C_p)·F_p^n,   ⇒   J_p^N = J_p^0 · Π_n det(I + Δt·C_p^n).

For a **thin sheet embedded in 3D**, the out-of-plane (normal) direction carries no
material stiffness, so `C_p`'s normal component is unconstrained. If the per-step
factors have geometric mean ≠ 1, `J_p` drifts exponentially. We observed
`det F ≈ ±10²⁹` roughly 40 steps *before* the velocity blew up — while positions and
in-plane forces still looked normal. So `F`, one of the training targets, is
ill-posed for this scenario regardless of the inversion crash.

## 6. Why drape and collision are fine

Both reach **bounded-strain configurations** and stay there:

- **Drape:** the sheet falls freely, then rests on the sphere/ground. Strain settles;
  there is no sustained load pumping deformation.
- **Collision:** pinned, but the sphere *supports and spreads* the contact; the sheet
  does not rotate freely about the pins.

The distinguishing feature of wind is **sustained forcing with too few constraints to
reach any bounded-strain equilibrium**, which is exactly what makes it fail.

## 7. What a non-deferred fix would require

Not a coefficient or a one-line constitutive tweak — a solver effort:

1. **Inversion-aware elasticity** (invertible FEM à la Irving/Stomakhin: detect `J < 0`
   and flip the sign of the smallest singular value so the force resists inversion),
   most robustly with **analytical** forces rather than autodiff.
2. **A codimensional / membrane MPM** formulation so the normal-direction `F` is
   well-defined for thin sheets.
3. Likely **implicit integration** for the forced regime (unconditionally stable),
   e.g. routing wind entirely to the existing `ImplicitClothSim`.

Given the 12-week scope and that wind is naturally a *fallback regime* for the hybrid
(a case you route to the safety net, not one the MPM-surrogate must learn), it is
deferred to a stretch goal. The wind sampler is kept in `generate_dataset.py`
(`--n-wind N` to opt in) so the work can resume without re-deriving the setup.
