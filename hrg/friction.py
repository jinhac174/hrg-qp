"""Joint friction: the feed-forward the controller adds, and its identification.

Feed-forward (``hrg.controller``, ``friction:`` in the configs)::

    tau_f = scale (coulomb tanh(qd_cmd / v_smooth) + viscous qd_cmd),   |tau_f| <= max

on the *commanded* joint velocity, not the measured one: it is zero whenever
the plan says stop, so it cannot push a holding arm, and an over-estimate is
not the negative damping it would be on the measured velocity. ``scale`` < 1
keeps it below the identified friction (start at 0.5).

Identification (``scripts/robot/identify_friction.py``): one joint at a time
moves back and forth over the same range at a few constant speeds while a PD
tracks it with the gravity model. On a constant-speed pass the PD supplies
what the model lacks: friction (odd in the velocity) plus the model's gravity
error over that range (the same in both directions). So, per speed,
``(tau_+ - tau_-) / 2`` is the friction and ``(tau_+ + tau_-) / 2`` the model
error, and a line through the friction over the speeds gives Coulomb
(intercept) and viscous (slope) friction.
"""

from __future__ import annotations

import numpy as np


def friction_torque(qd_cmd, scale, coulomb, viscous, v_smooth: float, cap, v_fade: float = 0.0) -> np.ndarray:
    """The feed-forward (see the module docstring); ``scale`` per joint or one
    for all. ``v_fade`` > 0 (hand-guiding): the compensation fades at speed,
    ``x 1 / (1 + (qd / v_fade)^2)``: full when starting and moving slowly, little
    once the arm coasts (a thrown arm stops sooner)."""
    qd_cmd = np.asarray(qd_cmd, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if np.all(scale <= 0.0):
        return np.zeros_like(qd_cmd)
    tau = scale * (np.asarray(coulomb) * np.tanh(qd_cmd / v_smooth) + np.asarray(viscous) * qd_cmd)
    if v_fade > 0.0:
        tau = tau / (1.0 + (qd_cmd / v_fade) ** 2)
    return np.clip(tau, -np.asarray(cap), np.asarray(cap))


def fit_friction(speeds, tau_pos, tau_neg):
    """One joint: ``speeds`` (S,) > 0, ``tau_pos`` / ``tau_neg`` (S,) the mean
    PD torque on the passes at ``+speed`` / ``-speed``. Returns ``(coulomb,
    viscous, friction (S,), model_error (S,))``; both coefficients >= 0."""
    v = np.asarray(speeds, dtype=float)
    f = 0.5 * (np.asarray(tau_pos, dtype=float) - np.asarray(tau_neg, dtype=float))
    off = 0.5 * (np.asarray(tau_pos, dtype=float) + np.asarray(tau_neg, dtype=float))
    if len(v) >= 2:
        A = np.column_stack([np.ones_like(v), v])
        (fc, fv), *_ = np.linalg.lstsq(A, f, rcond=None)
        if fv < 0.0:  # no viscous part visible: Coulomb only
            fc, fv = float(np.mean(f)), 0.0
        if fc < 0.0:
            fc, fv = 0.0, float(max(np.sum(f * v) / np.sum(v * v), 0.0))
    else:
        fc, fv = float(max(f[0], 0.0)), 0.0
    return float(fc), float(fv), f, off


def sweep_profile(c: float, amp: float, speeds, t_ramp: float = 0.3, dt: float = 0.001):
    """Reference for one joint: from ``c`` to ``c - amp``, then for every speed
    a pass up to ``c + amp`` and back down, finally back to ``c``. Trapezoidal
    velocity (ramps of ``t_ramp``). Returns ``(q (T,), qd (T,), label (T,))``:
    label ``+-(k + 1)`` on the constant-speed part of a pass at ``speeds[k]``
    up / down, 0 elsewhere."""
    qs, qds, labels = [], [], []

    def move(q0, q1, v, lab):
        d = q1 - q0
        L = abs(d)
        if L < 1e-9:
            return
        s = np.sign(d)
        tr = min(t_ramp, L / max(v, 1e-9))  # short moves: triangular
        vmax = min(v, L / tr)
        t_total = L / vmax + tr
        t = np.arange(0.0, t_total + dt, dt)
        # velocity: ramp up over tr, constant, ramp down over tr (raised-cosine ramps)
        up = np.clip(t / tr, 0.0, 1.0)
        down = np.clip((t_total - t) / tr, 0.0, 1.0)
        prof = vmax * np.minimum(0.5 - 0.5 * np.cos(np.pi * up), 0.5 - 0.5 * np.cos(np.pi * down))
        pos = np.concatenate([[0.0], np.cumsum(0.5 * (prof[1:] + prof[:-1]) * dt)])
        k = L / max(pos[-1], 1e-12)  # the sampled integral lands exactly on q1
        qs.append(q0 + s * k * pos)
        qds.append(s * k * prof)
        const = (t > tr + 0.2) & (t < t_total - tr) if lab else np.zeros(len(t), dtype=bool)
        labels.append(np.where(const, lab, 0))

    v0 = float(min(speeds))
    move(c, c - amp, v0, 0)
    for k, v in enumerate(speeds):
        move(c - amp, c + amp, float(v), k + 1)
        move(c + amp, c - amp, float(v), -(k + 1))
    move(c - amp, c, v0, 0)
    return np.concatenate(qs), np.concatenate(qds), np.concatenate(labels)
