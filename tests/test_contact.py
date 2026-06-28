"""Unit tests for the shared contact module (Phase 1)."""

import numpy as np

from src.contact import ContactGeometry


def test_constants_from_config(base_cfg):
    g = ContactGeometry(base_cfg)
    assert g.n_grid == 128
    assert g.domain == 2.0
    # 3-cell margin at the real resolution, NOT domain/64.
    assert np.isclose(g.margin, 3 * (2.0 / 128))
    assert np.allclose(g.sphere_c, [1.0, 0.4, 1.0])
    assert np.isclose(g.sphere_r, 0.2)


def test_sphere_pushes_particle_to_surface(base_cfg):
    g = ContactGeometry(base_cfg)
    # A particle just inside the sphere, moving inward (downward toward center).
    x = (g.sphere_c + np.array([0.0, 0.1, 0.0]))[None, :]  # 0.1 < r=0.2 -> inside
    v = np.array([[0.0, -1.0, 0.0]])
    x2, v2 = g.project_contact(x, v)
    dist = np.linalg.norm(x2[0] - g.sphere_c)
    assert np.isclose(dist, g.sphere_r, atol=1e-9)   # on the surface
    # Inward normal velocity removed; here normal is +y, inward vel was -y.
    assert v2[0, 1] >= -1e-9


def test_sphere_leaves_outside_particles_untouched(base_cfg):
    g = ContactGeometry(base_cfg)
    x = (g.sphere_c + np.array([0.0, 0.5, 0.0]))[None, :]  # outside
    v = np.array([[0.3, -0.4, 0.1]])
    x2, v2 = g.project_contact(x, v)
    assert np.allclose(x2, x)
    assert np.allclose(v2, v)


def test_box_clamps_position_and_velocity(base_cfg):
    g = ContactGeometry(base_cfg)
    lo, hi = g.margin, g.domain - g.margin
    # One particle below low x bound moving further out, one above high z bound.
    x = np.array([[lo - 0.1, 1.0, 1.0],
                  [1.0, 1.0, hi + 0.1]])
    v = np.array([[-2.0, 0.0, 0.0],
                  [0.0, 0.0, 3.0]])
    x2, v2 = g.project_contact(x, v)
    assert np.isclose(x2[0, 0], lo)
    assert v2[0, 0] >= -1e-9          # outward (negative) velocity zeroed
    assert np.isclose(x2[1, 2], hi)
    assert v2[1, 2] <= 1e-9           # outward (positive) velocity zeroed


def test_inputs_not_mutated(base_cfg):
    g = ContactGeometry(base_cfg)
    x = (g.sphere_c + np.array([0.0, 0.05, 0.0]))[None, :].copy()
    v = np.array([[0.0, -1.0, 0.0]])
    x_orig, v_orig = x.copy(), v.copy()
    g.project_contact(x, v)
    assert np.allclose(x, x_orig)
    assert np.allclose(v, v_orig)


def test_disabled_sphere_skips_projection(base_cfg):
    import copy
    cfg = copy.deepcopy(base_cfg)
    cfg["contact"]["primitives"]["sphere"]["enabled"] = False
    g = ContactGeometry(cfg)
    x = (g.sphere_c + np.array([0.0, 0.05, 0.0]))[None, :]  # would be inside
    v = np.array([[0.0, -1.0, 0.0]])
    x2, v2 = g.project_contact(x, v)
    assert np.allclose(x2, x)
    assert np.allclose(v2, v)
