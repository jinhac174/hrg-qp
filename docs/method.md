# The method, equation by equation, and where it lives in the code

Paper: Lee, Lin, Alexiev, Kim, *Hierarchical Reactive Grasping via Task-Space
Velocity Fields and Joint-Space Quadratic Programming*, arXiv 2509.01044
(<https://reactivegrasp.github.io>). Their system: a 7-DoF arm with two 4-DoF
fingers. Here: a Franka FR3 with a Robotiq 2F-85 parallel gripper.

Every adaptation the hardware forces is marked **[FR3 + 2F-85]** with its
reason. Everything else follows the paper as written.

## The loop

Every plan (100 Hz, `hrg/controller.py` plan thread; the paper: "maximum
planning frequency 100 Hz, in a separate thread from the main control loop"),
at the **measured** joints `q` and gripper width `w`, `ReachPlanner.step`
(`hrg/planner.py`) runs eqs. (2)-(12) in this order:

| Step | Paper | Code |
|---|---|---|
| 1 | fingertips `x(q)` and their Jacobians | `hand.fingertip_state` |
| 2 | target selection from the candidate set, eq. (12) | `targets.FingertipTargets.valid`, `targets.select_target` |
| 3 | fingertip velocity fields, eqs. (3)-(5) (or the eq. (2) baseline) | `fields.FingertipPathField`, `path_opt`, `fields.linear_fingertip_field` |
| 4 | auxiliary fields, eqs. (6)-(10) | `fields.grasp_stability_field`, `fields.gripper_width_field`, `fields.tanh_weight` |
| 5 | the QP, eq. (11) | `planner.ReachPlanner.step`, `qp.solve_tracking_qp`, `robot.RobotModel.gamma_rows` |
| 6 | close at 1 cm (sim) / 1.5 cm (robot), then lift | `controller.HRGController._reach`, `_plan` |

The control thread (1 kHz) turns `qdot_des` into torques (Sec. II-D, below).

## Equation by equation

### Fingertips `x_i(q)` — `hrg/hand.py`

**[FR3 + 2F-85]** The two fingertips are the pad-face centres. They depend on
the 7 arm joints *and* on the opening width `w`, the gripper's one joint, so
`w` is the QP's 8th variable, `z = (qdot, wdot)`; this is the paper's
"gripper joints" `q_gr` with `n_gr = 1`. In the TCP frame:
`x_{1,2} = (0, ∓w/2, pad_z(w) − TCP_Z)`, where `pad_z` is the 2F-85's linkage
arc (checked against the MuJoCo pad sites to 0.5 mm,
`test_fingertips_match_mujoco_pad_sites`). Jacobians: the rigid-body part plus
`R_tcp ∂x/∂w` in the width column (`point_jacobians`; checked by finite
differences).

### Candidate targets `{x*_j}` and eq. (12) — `hrg/targets.py`

Paper: candidate fingertip pairs "predefined in the object frame" (from
T2SQNet), the selection metric
`|x1 − x1*| + |x2 − x2*| + w ∠(x2 − x1, x2* − x1*)`.

* Generation (offline, `scripts/prepare_objects.py`): rim pinches as T2SQNet
  places them on tableware (Fig. 4), plus antipodal pairs elsewhere. Each pad
  backs off 1.5 cm along the pair axis (the pre-grasp target). A target must
  satisfy eq. (3)'s end-point condition `d(x*, O) >= ε_O`. **[FR3 + 2F-85]**
  The rigid hand must also fit there: for some rotation of the hand about the
  pair axis, every hand sphere is `qp_eps_obj` clear of the object and the
  gripper boxes clear of the mesh. Those approach directions are stored.
* Online: a pair is valid if one of its stored approach directions keeps the
  hand above the table at the object's current pose (`valid`); eq. (12)
  selects among the valid ones, in both finger orders (`selection_costs`).
  `select_w_angle = 0.05` m/rad.

### Eq. (2), the baseline — `fields.linear_fingertip_field`

`xdot_des = K (x* − x)`, `K = linear_gain · I` (1 /s). Same QP, same
constraints. `--hrg.field linear`.

### Eq. (3), fingertip paths — `hrg/path_opt.py`, `fields.FingertipPathField`

`min ∫ |c'(s)|² ds  s.t.  d(c(s), O) >= ε_O,  c(0) = x, c(1) = x*`, with the
non-target obstacles and the table as further constraints (paper: 5 cm for
obstacles, 1 cm for the target object).

* 100 knots, end points fixed; the discrete energy `Σ |c[k+1] − c[k]|²`
  (Hessian: the second-difference matrix) — `path_opt.sqp`.
* Initial guess as in the paper: the straight line if it is free; otherwise
  two segments through the via point 10 cm from the target with the fewest
  violating knots (ties: shortest), from 100 directions on the sphere —
  `path_opt.initial_paths`.
* 3 SQP iterations per plan (paper: "typically ≈ 3"), OSQP. Each linearises
  every distance constraint at the current knots.
* **[FR3 + 2F-85]** The two fingertips of a parallel jaw are a rigid pair:
  `|c1(s) − c2(s)| <= 85 mm`, the gripper's opening, linearised in the SQP
  like the distances. The via search uses one via direction for both
  fingertips, which satisfies it by construction. Without it the two paths
  can pass on opposite sides of an object: fields no motion of the gripper
  can follow. (The paper's two fingers move independently.)
* Two numerical safeguards, both standard for SQP:
  * A trust region: each knot moves at most 2 cm per iteration, so the
    linearised constraints stay valid.
  * Receding horizon: the previous plan's paths, re-anchored at the current
    fingertips, compete with the via-point guess as the initial guess
    (fewer violations, then shorter). Over consecutive plans the iterations
    then accumulate toward the optimum of eq. (3), instead of restarting
    from the via path every 10 ms. A solution on its active constraints
    counts as feasible within the QP's tolerance (1 mm).

### Eqs. (4)-(5), the fingertip velocities — `FingertipPathField.__call__`

`xdot_i,des = v(x_i, x_i*) ċ0/|ċ0|` with `ċ0` the first segment of the path,
`v = v_const` beyond `δ_x`, `v_const (1 − (1 − e/δ_x)²)` within
(`quad_decay_speed`). `v_const = 0.15` m/s, `δ_x = 5` cm.

### Eqs. (6)-(8), grasp stability — `fields.grasp_stability_field`

Paper: `g` = the mean of the angles between `x2 − x1` (and `x1 − x2`) and the
central axis of each fingertip's contact region; `ω_i = −unskew(∂g/∂R_i R_iᵀ)`;
weight `α(g) = ½ + ½ tanh((g − η_g)/T_g)`.

Implemented literally (the gradient in closed form:
`ω_i = (a_i × u_i) / (2 sin θ_i)`; a test checks it is the steepest descent).
**[FR3 + 2F-85]** The 2F-85's pad normals are `±y_tcp` and `x2 − x1 = w y_tcp`,
so `g ≡ 0` in every configuration: `α = 0` and the term is inactive. The
unstable contact of the paper's Fig. 3 (a finger touching with its side)
cannot occur with parallel pads (`test_grasp_stability_is_zero_on_the_parallel_jaw`).

### Eqs. (9)-(10), gripper posture — `fields.gripper_width_field`

`wdot_des = v(w, w*) sign(w* − w)` toward the nominal open posture
`w* = 85 mm` (`v_gr = 0.1` m/s, `δ_gr = 2` cm), weighted by
`β(h) = ½ + ½ tanh((h − η_h)/T_h)` with `h = |x − x*|`, `η_h = 6` cm,
`T_h = 2` cm: open far away, free to close near the target.

### Eq. (11), the QP — `planner.ReachPlanner.step`, `hrg/qp.py`

```
min_z  Σ_i |xdot_i,des − J_i z|² + α Σ_i |ω_i,des − J_ω z|² + β |wdot_des − S z|² + 1e-3 |z|²
s.t.   Γ(q) + dΓ/dz z H >= ε_Γ                     robot vs walls / floor, robot vs itself
       d(X(q,w), O) + dd/dz z H >= ε_O + r         hand spheres vs object, table, obstacles
       q_min <= q + qdot H <= q_max,  0 <= w + wdot H <= 85 mm
       |qdot| <= qdot_max,  |wdot| <= 150 mm/s
```

* `H = 0.10` s. Margins: object 1 cm, obstacles 2 cm, table 5 mm, Γ 2 cm.
  When a pair is already inside its margin, the demanded separation rate
  `(ε − d)/H` is capped at 0.25 m/s so the QP stays feasible while it
  recovers.
  **[FR3 + 2F-85]** The paper leaves `H` to the user ("`H` and the margins
  control the conservativeness"). Linearised over `H`, a distance row lets
  the hand approach its margin at up to `(d − ε)/H` and pushes it out at
  `(ε − d)/H`: a first-order margin servo with gain `1/H`. That gain must
  be slow next to the closed loop below it. Measured on the simulated FR3,
  a commanded joint velocity is followed about 40 ms later (plan period and
  plan time, the command filter, the PD on the integrated reference). With
  `H = 0.05` s a hand resting on its object margin overshot it by 5 mm and
  bounced at 3–4 Hz (wrist velocity ±0.2 rad/s), visible as a shaking hand
  whenever a reach stalls or hovers next to the object. With `H = 0.10` s
  the same state settles within 1 mm of the margin with no oscillation, and
  reaches away from any margin are unchanged (same lift times to 0.1 s).
* `1e-3 |z|²`: with `α = β = 0` near the target the cost constrains six of
  the eight velocities (two fingertips × 3); the regulariser picks the
  smallest motion for the rest.
* `Γ(q)` — `robot.RobotModel.gamma_rows`, FCL as in the paper: the robot's
  capsules / boxes against the scene's floor and walls, and the self-collision
  pairs Franka's SRDF leaves enabled. A capsule is measured as its core
  segment minus its radius; on the segments and boxes FCL's GJK is exact, so
  each row `n · J(p)` is the exact gradient (checked by finite differences to
  1e-4). FCL's GJK on the capsule's curved ends returns nearest points that
  are millimetres off, which tilts the rows by about 1 %.
* `X(q, w)` — `hand.hand_spheres`: 22 spheres for the open 2F-85 (16 on the
  fingers, 2 on the linkages, 4 on the palm). The finger and linkage spheres
  move with `w`, so their Jacobians have a width column too.
* `d(X, O)` — `hrg/sdf.py`. As in the paper: thin concave objects (bowl,
  mug, wine glass, plate) are surface point clouds, nearest-point distance.
  The paper uses superquadrics for convex objects and obstacles, and
  requires `d` to be the signed *Euclidean* distance. Our convex object (the
  box) and the obstacles are boxes, so their exact signed distance is used —
  the e → 0 superquadric. A superquadric's implicit function is not
  Euclidean: with box-like exponents it overestimated the clearance near
  edges by up to 17 mm, and a box was knocked over in an earlier version.
* OSQP, as in the paper.

### Sec. II-D, the low level — `controller.HRGController._update`

Paper: "Given `qdot_des` from (11), we integrate over one control interval to
obtain the desired joint position and then apply PD control with gravity
compensation." Every 1 ms (`ref_mode: step`):

```
q_des = q + qdot_des dt
τ     = Kp (q_des − q) + Kd (qdot_des − qdot) + g(q) + C(q, qdot) qdot + τ_f(qdot_des)
```

* `Kp (q_des − q) = Kp dt qdot_des` is about 1 % of the Kd term (600 × 1 ms
  next to 50), the same direction: `Kp = 0`. The law is a damping (velocity)
  controller, `Kd = [50 ×3, 20 ×3, 10]`; position is fed back by the planner,
  which solves the QP at the measured state every 10 ms. States that command
  no motion (idle, close, reopen, hold) hold the joints through the planner
  (`hold_gain (q_hold − q)`), not through Kp.
* **[FR3]** That leaves joint friction and gravity-model errors unhidden: an
  unmodelled torque Δτ becomes a velocity error Δτ / Kd. Hence the friction
  feed-forward `τ_f` (`hrg/friction.py`): `scale (f_c tanh(qdot_des / v_s) +
  f_v qdot_des)`, capped, on the *commanded* velocity (zero when the plan
  stops; an over-estimate is not negative damping), `scale = 0.5`. Sim: the
  MuJoCo model's `frictionloss` / `damping`; robot: measured by
  `scripts/robot/identify_friction.py`. And the payload: the wrist camera is
  in the URDF, a held object's weight is added while it is held.
* `ref_mode: integral` is the earlier variant (`q_des ← q_des + qdot_des dt`,
  within 0.05 rad of `q`, `Kp = [600 ×4, 250, 150, 50]`): Kp is then the
  integral of the velocity error, which hides friction and model errors
  behind a stiff spring (the eq. (2) baseline restarted from `q` without
  friction compensation stalled 2–3 cm from its targets).
* `g(q) + C qdot` from the URDF. **[FR3]** The hardware process cancels the
  FR3's own gravity compensation (`gravity_model: controller`), so sim and
  robot see the same law.
* **[FR3]** `qdot_des` passes a first-order filter (0.25 per tick, ~4 ms):
  the plan changes in steps at 100 Hz and libfranka rejects torque
  discontinuities. If no plan arrives for 0.2 s the joints are held where
  they were (`stale_hold_kp` / `kd`).
* **[2F-85]** The Robotiq is position-servoed (and so is its sim). `wdot_des`
  is realised the same way, as a set-point `w_ref += wdot_des dt` kept within
  1 cm of the measured width and streamed as `move` commands at 50 Hz.

### Grasp and lift — `controller.HRGController._plan`

Paper: "once the fingertips come within a specified threshold (1 cm) of the
target positions, the gripper closes, the normal force is gradually
increased until the object is firmly secured without slipping, and the
object is lifted."

* Trigger: both fingertips within 1 cm (sim) / 1.5 cm (robot,
  `configs/robot/controller.yaml`).
* **[2F-85]** The Robotiq takes its grip force before it closes: a `grasp`
  command at 30 N. It succeeds if the fingers stop on something, between half
  the contact width and just inside the pre-grasp opening; otherwise the
  gripper opens and the reach starts again. If the object is lost during the
  lift or hold, the same happens.
* Lift: the TCP 15 cm straight up through the same QP (a pose field as the
  task; `Γ` and the hand vs table / obstacles as constraints, plus 64 points
  of the held object vs the table), then (not in the paper) **upright**: the
  hand pose that puts the object back in its upright pose (object `+z` up,
  the smallest tilt about the TCP, raised until it clears the table by
  3 cm), from the object's pose in the hand recorded when the gripper
  reported the grasp. Tracking refreshes it after the rise and after the
  turn when plausible (the grasped point still within 2 cm between the pads,
  the object turned at most 45°): a bowl or glass that pivoted in the
  fingers is corrected, twice at most. Then hold.

## Beyond the paper: the oriented reach — `planner._select_oriented`, `targets`, `fields`

The paper steers fingertip *points*. On the parallel jaw, reaching both points fixes the closing
axis at the end, but not the roll about it, i.e. the approach direction, so a rotated object is not
approached in its own frame. Design and history:
`docs/superpowers/specs/2026-10-28-oriented-reach-design.md`.

```
min_z  Σ_i |xdot*_i − J_i z|² + α Σ_i |ω_i,des − J_ω z|² + Σ_n α(θ_n) |J_r,b z − ω_des,b,n|²
       + κ |qdot − k_q (q_grasp − q)|² + β(h_pre) |wdot* − S z|² + reg |z|²
ω_des,b,n = R_sbᵀ sat( k (R_sb n̂_b) × n̂_t, ω_max ),   α(θ_n) = γ min(1, θ_n / θ_0)
n: the pads' normal (TCP y) → the pair axis (the wall's normal), the approach (TCP z) → z*
R* = R_obj R_grasp,j,k (· Rz(π) for the other finger order)
```

* **Target data.**
  - Every pair has 24 approach directions about its axis. One is admissible if the open hand is
    clear at the pre-grasp and swept back 10 cm along it.
  - `R_grasp = [y × z, y, z]`, with `y` the pair axis and `z` the approach, in the object frame.
  - Rim pinches come in two kinds per rim hit: horizontal (T2SQNet's) and along the wall's normal.
    Of two at the same place the more antipodal one is kept (`misalign`). On the gold bowl's
    22°-slanted wall a horizontal pinch squeezed the bowl 22° out of upright.
* **The grasp is chosen in joint space**, every plan, replacing eq. (12):
  - Candidates: every usable (pair, direction, finger order).
  - Cost: the joint motion to its configuration `|q_grasp − q|` (IK; reachable = joint range and Γ
    clear), plus 2 rad per rad between its approach and the object's preferred one
    (`approach_axis`, from above by default).
  - The first-order estimate `|J⁺ e|` ranks them; the cheapest few are solved by IK, warm-started
    from cached solutions.
  - The current choice keeps winning as the arm closes in on its configuration. With no reachable
    grasp the plan is `None` and the controller holds.
* **The posture task** `κ = 0.05` toward `q_grasp` makes the arm end in the configuration the choice
  was made for. Without it the QP reached the same hand orientation on the other wrist branch,
  joint 7 half a turn further.
* **Final approach.**
  - The fingertip paths (eq. 3) end at pre-grasp points: the targets' midpoint 10 cm back along
    `z*`, the fingertips spread at the current opening. Then comes the straight segment in.
  - Inside the corridor (the midpoint within a cone around the targets') the paths are straight
    lines.
  - The gripper posture weight `β` fades with the midpoint's distance to the pre-grasp. Pre-grasp
    points at the target opening, against a posture field holding the gripper open, deadlocked the
    QP.
* **Alignment.** For each vector pair, the gripper's `n̂_b` turns toward the target's
  `n̂_t` at `k |n̂ × n̂_t|`, capped at `ω_max` (`k = 2 /s`, `ω_max = 1 rad/s`: capped above
  30°). The weight `α(θ)` is `γ = 0.1` while the vectors are more than `θ_0 = 30°` apart and
  falls linearly to 0 as they align. The pads' normal onto the wall's normal is what makes the
  grasp antipodal. The approach onto `z*` fixes the remaining roll about the wall's normal,
  i.e. from where the hand comes, in the object's frame. The gripper closes only within 10°
  of `R*`.
* **Limits** (kinematic rollouts, `docs/results.md`):
  - Objects lying on their side near the base can have no reachable oriented grasp; the planner
    says so.
  - A few stall at the arm's self-collision margin while the grasp configuration itself is clear:
    the paper's own limitation (fingertip targets planned independently of the joints).
  - A wine glass on its side is not approached through its opening at table height: link 6 would
    meet the table and joint 2 its limit. Its stem or foot is taken.

## What the paper leaves open, and the choices made

| Item | Choice |
|---|---|
| Selection weight `w` (eq. 12) | 0.05 m/rad |
| `K` (eq. 2) | 1 /s |
| `η_g, T_g, η_h, T_h, δ_gr, v_gr` | 0.15 rad, 0.05, 6 cm, 2 cm, 2 cm, 0.1 m/s |
| `H`, margins | 0.10 s (above the ~40 ms closed-loop latency, see eq. 11); object 1 cm (paper), obstacles 5 cm in eq. (3) (paper), 2 cm in eq. (11) |
| Start poses "left-bottom, center-up, right-bottom" | `hrg/study.py`: the ready pose, and the gripper pointing down 27 cm to either side with the fingertips at rim height |
| GPU nearest-neighbour / via search | CPU (KD-tree, NumPy); plans take 7–15 ms |

## Known limitations

* **Object pose in the hand.** The upright turn trusts the object's pose
  recorded at the grasp, refreshed from tracking when plausible; without
  tracking, an object that pivots in the fingers afterwards (a heavy bowl
  pinched at its rim) ends tilted by that much.
* **Fingertip-only targets.** Paper Fig. 12 and Sec. IV: "fingertip velocity
  targets are planned independently of the full joint configuration." On the
  FR3 this shows when an object lies on its side close to the base: the
  elbow (joint 4) reaches its limit and link 6 its self-collision margin to
  link 1 while the hand spheres rest on their 1 cm object margin, and the
  reach stops 1–2 cm short (5 of 150 rotated trials, one of them with the
  finger bodies rather than a joint limit at their margin;
  `docs/results.md`). Every plan
  reports what binds (`ReachStep.info['binding']`, in the controller's log);
  `scripts/dissect.py` replays a recorded trial to the same answer.
* **Thin rims.** A 1 mm wine-glass rim pinched with one pad near the 1 cm
  trigger distance can slip out while lifting. The Robotiq's grasp
  detection (fingers stopped within a width window) cannot tell a 1 mm wall
  from empty fingers.
* **Dish.** No 2F-85 target on the scanned plate (or a deeper test dish)
  satisfies the paper's margins: the fingers reach about 2 cm below the pad
  centre and meet the table or the plate.
