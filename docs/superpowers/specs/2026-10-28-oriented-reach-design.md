# Oriented reach, upright lift, and the paper's low-level law

Status: agreed with the user in chat on 2026-10-28 ("that is perfectly what I wanted ... get all done").
Built the same day, then cleaned up the same day at the user's request ("no band-aid patches"): a
first version had collected threshold rules, and one joint-space grasp choice replaced them (see
"History"). This document describes the cleaned-up version.

## Why

The paper's reach (eqs. 3–12) only steers the two fingertip *points*. On the 2F-85, reaching both
points fixes the closing axis at the end, but not the roll about it. That roll is the approach
direction. So a rotated object is not approached in its own frame, and the wrist ends up wherever
the QP drifted. After the lift the object stays tilted.

The low level departed from the paper too. The paper integrates `qdot_des` over one control
interval from the measured `q`, which makes the law a damping (velocity) controller: Kp adds about
1% on top of Kd. Our code kept a running integral instead. The user's professor wants the paper's
law. That law exposes friction and gravity-model errors, so friction compensation, the wrist
camera's mass and the held object's mass come with it.

## Goals

1. **Approach in the object frame.**
   - Every target pair has hand orientations in the object frame, one per admissible approach
     direction.
   - Online `R* = R_obj · R_grasp`, so a rotated object is approached in its rotated frame.
   - The object config's `approach_axis` (default: from above) is the preferred approach.
2. **Sensible wrist motion.**
   - The two finger orders differ by 180° about the approach axis.
   - Of all grasps, the one whose arm configuration is nearest the current joints wins. That picks
     the finger order (and everything else) with the least motion.
3. **Last stretch straight in.** The fingertips travel the last `approach_dist` (10 cm) along the
   approach axis.
4. **Upright after the lift.**
   - The object ends in its known upright pose (object `+z` up) by the smallest tilt.
   - The gripper ends wherever that puts it.
   - The held object is kept clear of the table, and its weight is in `g(q)`.
5. **The paper's low level.**
   - `q_des = q + qdot_des dt`, Kp = 0: `τ = Kd (qdot_des − qdot) + g(q) + C qdot + τ_f`.
   - Position feedback comes from the 100 Hz planner. The states that hold do so with a joint field.
6. **Friction feed-forward** on the commanded velocity, `τ_f = s (f_c tanh(qdot_des / v_s) + f_v
   qdot_des)`, capped. It is measured per robot by `scripts/robot/identify_friction.py`, and off on
   the robot until then.
7. **Wrist camera in the model:** a `wrist_camera` body at its calibrated pose. Its 45 g mount mass
   is an estimate.

## Non-goals / anti-goals

- No new perception: normals and the object frame come from the mesh and FoundationPose's pose.
- The paper's behaviour stays one flag away (`hrg.orient: false`; `ref_mode: integral` + gains).
- **Anti-goal: threshold rules added to make test cases pass** (gates, stall counters, hysteresis,
  fallbacks, caps). A failure gets a root cause and a principled fix, or it is reported as a
  limitation.

## The objective

Decision variable `z = (qdot (7), wdot)`. Constraints are unchanged: Γ, hand spheres vs object /
table / obstacles, joint and width limits over `H`, velocity limits.

**Reach** (`ReachPlanner.step`):

```
min_z  Σ_i |xdot*_i − J_i z|² + α Σ_i |ω_i,des − J_ω z|² + Σ_n α(θ_n) |J_r,b z − ω_des,b,n|²
       + κ |qdot − k_q (q_grasp − q)|² + β(h_pre) |wdot* − S z|² + reg |z|²

ω_des,b,n = R_sbᵀ sat( k (R_sb n̂_b) × n̂_t, ω_max ),   α(θ_n) = γ min(1, θ_n / θ_0)
```

- **Alignment** (the user's formulation, 2026-10-28), in the body frame `J_r,b = R_sbᵀ J_r`:
  - For each gripper vector `n̂_b` and target `n̂_t`, the desired angular velocity is the cross
    product `(R_sb n̂_b) × n̂_t` taken to the body frame, capped at `ω_max`.
  - The weight `α(θ)` is full above `θ_0` and decays linearly to 0 as the vectors align.
  - Two pairs:
    - the pads' normal (TCP `y`) → the pair axis (for an antipodal pair, the wall's normal);
    - the approach (TCP `z`) → the grasp's approach `z*`. That fixes the roll about the wall's
      normal, i.e. the approach in the object's frame.
  - `γ = 0.1`, `k = 2 /s`, `ω_max = 1 rad/s`, `θ_0 = 30°`, where `k sin θ_0` is the cap.

- **The grasp `(j, k, s)`, R* and q_grasp** come from `_select_oriented`, a choice in joint space,
  every plan:
  - Candidates: every usable (pair `j`, direction `k`) in both finger orders `s`.
  - Cost: `|q_grasp − q| + w_pref ∠(z*, R_obj a_pref)`. `q_grasp` is the arm configuration at the
    grasp pose (IK; reachable = in the joint range and Γ clear).
  - Ranking: by the first-order estimate `|J⁺ e|`. The cheapest few are solved by IK, warm-started
    from their cached solutions, and the cheapest reachable one is taken.
  - Stability: the current choice keeps winning as the arm closes in on its configuration, with no
    hysteresis rule.
  - No reachable grasp: the plan is `None`, and the controller holds and says so.
- **The posture task `κ`** makes the arm end in the configuration the choice was made for, which is
  what makes the joint-space choice meaningful. Weight 0.05, gain 1 /s.
- **Fingertip fields `xdot*_i` (eqs. 3–5) with an approach corridor.**
  - Outside the corridor, eq. (3) plans to pre-grasp points: the targets' midpoint backed off along
    `z*`, the fingertips spread along the target closing axis at the current opening.
  - Inside the corridor (the midpoint within a cone around the targets', with hysteresis), the
    paths are straight lines to the targets.
  - The eq. (5) speed uses the distance to the targets throughout.
- **β(h_pre):** the gripper posture fades with the midpoint's distance to the pre-grasp (×√2, the
  paper's scale; 0 inside the corridor). From the pre-grasp on, the targets set the opening.
- **Grasp trigger:** both fingertips within `grasp_trigger_dist` and the hand within
  `grasp_trigger_rot` (10°) of R*. This is the orientation counterpart of the paper's position
  trigger: the definition of being at the grasp.

**Lift** (`lift_step`, the 6-D pose field, plus rows keeping 64 points of the held object above the
table):

- Rise: 15 cm up.
- Upright: a feedback loop on the object's tilt in the hand.
  - Every plan the goal is recomputed: the smallest rotation putting the object's `+z` up, about
    the TCP, at the fixed lifted position, raised so the object clears the table by 3 cm.
  - It ends when the tilt is within 3°.
  - The object's pose in the hand is taken when the gripper reports the grasp. It is refreshed from
    tracking whenever that is plausible: the grasped point within 2 cm of where it was between the
    pads, the object turned at most 45°. That check rejects outliers, not a patch.
- While held, the object's config mass at its surface-mean centre is added to `g(q)`.

## Target data (`hrg/targets.py`)

- **Rim pinches:** both kinds, the horizontal ray's and the one along the wall normal. Of pairs at
  the same place the more antipodal one is kept (`misalign`).
- **Approach directions:** 24 per pair, about the pair axis. Admissible: the open hand is clear at
  the pre-grasp and swept back `approach_dist` along the approach.
- **Online:** usable = admissible and clear of the table at both ends of the sweep.

## Parameters

| key | value | meaning |
|---|---|---|
| `hrg.orient` | true | false: the paper's reach |
| `hrg.rot_weight` γ, `rot_gain` k, `rot_max`, `rot_threshold` θ_0 | 0.1, 2 /s, 1 rad/s, 30° | alignment terms |
| `hrg.posture_weight` κ, `posture_gain` | 0.05, 1 /s | the chosen configuration |
| `hrg.select_w_pref` | 2.0 rad/rad | joint motion per rad off the preferred approach |
| `hrg.select_shortlist` | 6 | new candidates solved by IK per plan |
| `hrg.approach_dist` | 0.10 m | straight final approach |
| `grasp.grasp_trigger_rot` | 10° | grasp criterion |
| `grasp.upright*`, `held_*`, `payload_comp` | | upright lift, held object |
| `ref_mode`, `kp`, `kd`, `friction.*`, `stale_hold_*`, `hold_*` | | low level |

## Known limitations (measured, not patched)

- **No reachable oriented grasp.** Objects lying on their side close to the base can leave no
  reachable approach: joint 6 at its limit, or the swept approach blocked. The planner then reports
  that and holds. The paper's reach would grasp with whatever roll it ends at.
- **Local minima of the task-space fields against the arm's own geometry.** A mug lying near the
  base stalls 5 cm short with link 6 at its self-collision margin. The grasp configuration itself is
  clear, but the fingertip fields drive along a Cartesian path the arm cannot follow there. This is
  the paper's own limitation: fingertip targets planned independently of the joint configuration.
  A stronger posture weight fixes these cases and breaks others, so there is no weight that is
  right for both.

## History: what the clean-up removed

The first build added rules after test failures:
- a wrist-roll heuristic with a joint-7 tie rule;
- an IK gate with blocking of whole (pair, direction) combinations;
- a 120° joint-7 limit;
- selection hysteresis;
- stall detection with neighbour blocking;
- a fallback to the paper's reach;
- a wait-at-the-pre-grasp entry gate;
- two correction passes for the upright turn.

Each stood in for one missing piece: the arm's configuration in the grasp choice. The joint-space
choice above replaced all of them. Two real causes found on the way were fixed at their root, not
gated:
- The posture field kept the gripper open while the pre-grasp points assumed the target opening,
  so the QP stood still. The fix was the pre-grasp at the current opening, and β on the midpoint.
- Approach directions at 30° steps missed the narrow band of approaches that are both
  collision-free and reachable. The fix was 24 steps.
