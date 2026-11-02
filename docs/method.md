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
(`quad_decay_speed`). **[FR3]** `v_const = 0.5` m/s, `δ_x = 10` cm (the
paper: 0.15 m/s, 5 cm); why these, and what bounds them, is in "Speed"
below. Near the target the decay is linear with gain `2 v_const / δ_x`
(10 /s), so a faster `v_const` also ends faster. The direction is read
over the first 2 cm of the path (`fields.path_direction`), not its first
1.5 mm segment: the SQP moves the interior knots by fractions of a
millimetre from plan to plan, and on a moving target that turned the first
segment by tens of degrees between plans. On a moving object the targets'
own velocity is added (feed-forward: the field alone trails a moving target
by `v_obj / K`, 3.5 cm at 0.35 m/s), the fingertip speed still at most
`v_const` (one factor for both fingertips): an object moved faster than that
is chased at `v_const` and caught as it slows. Followed at its own speed, an
object moved by hand took the hand to 1.1 m/s and the arm past the FR3's
velocity limits (the robot stopped, 2026-11-02).

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
`w* = 85 mm` (`v_gr = 0.15` m/s, the 2F-85's fastest, `δ_gr = 2` cm),
weighted by `β(h) = ½ + ½ tanh((h − η_h)/T_h)` with `h = |x − x*|`,
`η_h = 6` cm, `T_h = 2` cm: open far away, free to close near the target.
**[2F-85]** With the oriented reach (below) the grasp is known from the
first plan, so the posture **pre-shapes**: `w*` is the chosen targets'
opening plus `preshape_margin` = 1 cm on each side (85 mm for the paper's
reach). The fingers then straddle the object as the hand comes in along the
approach and the width never limits the straight approach: closing from
fully open to a bowl's rim takes 0.3–0.4 s at the gripper's top speed, the
last 10 cm of approach 0.2 s. The collision rows on the finger spheres still
decide the width where an obstacle or the object is in the way.

### Eq. (11), the QP — `planner.ReachPlanner.step`, `hrg/qp.py`

```
min_z  Σ_i |xdot_i,des − J_i z|² + α Σ_i |ω_i,des − J_ω z|² + β |wdot_des − S z|² + |N (qdot − qdot_post)|² + 1e-3 |z|²
s.t.   Γ(q) + dΓ/dz z H >= ε_Γ                     robot vs walls / floor, robot vs itself
       d(X(q,w), O) + dd/dz z H >= ε_O + r         hand spheres vs object, table, obstacles
       q_min <= q + qdot H <= q_max,  0 <= w + wdot H <= 85 mm
       s qdot_lo(q) <= qdot <= s qdot_hi(q),  |qdot| <= qdot_max,  |wdot| <= 150 mm/s
       |z − z_prev| <= a Δt_plan                     acceleration limits (joints, width)
       |M(q) â + g(q) + C(q, v) v| <= s_τ τ_max,  |M(q) (â − â_prev)| <= s_τ τ̇_max Δt_plan
       |â − â_prev| <= j Δt_plan     â = (qdot − v) / Δt_plan, v = the previous plan
                                                      torque, torque rate, jerk
(infeasible only: the distance rows get slacks s >= 0, + ρ Σ s + |s|² in the cost; the torque
 rows too only if they conflict with the box)
```

* `H = 0.10` s. Margins: object 1 cm, obstacles 2 cm, table 5 mm, Γ 2 cm.
  When a pair is already inside its margin, the demanded separation rate
  `(ε − d)/H` is capped at 0.25 m/s so the QP stays feasible while it
  recovers. **[moving object]** The object's rows carry its own motion:
  `d + (n·J z − n·v_obj) H >= ε`, with `v_obj` the object's velocity at the
  sphere (from the pose predictor below). Without that term a carried
  bowl ran into a hand the rows believed was standing clear.
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
* **[FR3]** `qdot_lo(q)`, `qdot_hi(q)`: the FR3's own joint velocity limits
  (`fr3/robot_platform/fr3_limits.py`, libfranka 0.17.0). They depend on
  the joint's position and direction (the speed from which it still stops
  before its end) and fall far below the nominal 2.62 rad/s toward the ends
  of the range: joint 4 at −2.68 rad may move toward its end at 1.42 rad/s.
  The robot stops itself above them (`joint_velocity_violation`), so the QP
  plans within `s = 0.8` of them (`joint_velocity_limit_scale`: the arm
  follows a command with ~0.1 rad/s of error at speed); `qdot_max` (2 rad/s
  on joints 1–4, 2.5 on the wrist) caps them. With the fixed 2 rad/s alone,
  a chased object took joint 4 to 135 % of its limit there.
* **[FR3]** `|z − z_prev| <= a Δt_plan`: the velocity may change by at most
  the joint acceleration limits `a` = 10 rad/s² on every joint (the FR3's:
  FCI documentation, libfranka) and 2 m/s² for the width over one plan
  period (`z_prev` the previous plan). The paper's QP is first order: its velocity may jump,
  and a torque-controlled arm realises the jump with a lag the rows know
  nothing about. Bounded, the plan is a velocity profile the arm can
  follow, and the low level (Sec. II-D) realises exactly it. See "Speed".
* **[FR3]** The torque rows: the torque the profile asks of the arm, the low
  level's feed-forward `M â + g + C v`, within `s_τ = 0.7` of the FR3's
  torque (87 Nm on joints 1–4, 12 on 5–7) and its change from the previous
  plan within `s_τ` of the FR3's torque rate (1000 Nm/s, libfranka) over the
  period: 7 Nm per plan (`joint_torque_limit_scale`; the rest is the Kd
  feedback's and a held object's). The per-joint acceleration box alone does
  not bound either: `M(q)` couples the joints (joint 1 carries
  `M_13 ≈ 2 kg m²` of joint 3's acceleration), so 10 rad/s² on every joint at
  once asked for torque steps of 20–49 Nm within one millisecond and, on a
  lift far out, 139 Nm of joint 2. On the robot (2026-11-02, recorded with
  `lcm-logger`) libfranka rate-limited the steps and joint 2 saturated: the
  torque arrived 20–60 ms late, the arm overshot the lift by 2.2 cm and came
  back, and on a fast reach the late Kd term drove joints 1 and 3 against
  each other at 8 Hz (their antiphase mode weighs only 0.27 kg m²) until a
  `power_limit_violation` stopped the robot. From rest the acceleration now
  builds over a few plans (the first plan moves the hand a few cm/s); the
  reaches and lifts take the same time.
* **[FR3]** The jerk rows: each joint's acceleration changes from the previous
  plan's by at most `j Δt_plan`, `j = 200 rad/s³` (`joint_jerk_limit`). The
  FR3's link-side torque follows its command 1–7 ms late and with a 12–38 ms
  first-order lag (joint 1 the slowest), fitted to the flight recorder of a
  stop (`outputs/logs/reflex_20261102_174526.npz`, the torque sent against
  `tau_J`), so an acceleration reaches a joint over ~30 ms: `10 rad/s² / 30 ms`.
  Plans that switched joints between ±10 rad/s² from one period to the next
  (the QP riding its box toward a field's demand) asked for more, and on
  light directions, which the torque rows let change fast (joints 1 and 3
  turning against each other weigh 0.17–0.28 kg m² at the lift), the arm
  fell behind and the Kd term drove an 8 Hz oscillation. The simulation
  with that torque response (`configs/sim/env.yaml`) reproduces it: without
  the jerk rows it shook from the low start poses, with them the joint 1 / 3
  counter-motion stays below 0.9 rad/s.
* **[FR3]** The rows are solved hard first (the paper's QP). Only when that
  is infeasible — inside a margin, the arm at rest and the acceleration
  and torque limits between it and the demanded separation rate, e.g. the
  rows of an obstacle in the way — the distance rows are
  relaxed through slacks
  `s >= 0` with the linear penalty `ρ Σ s`, `ρ = 100` (plus a small `|s|²`
  that keeps OSQP converging). A linear penalty is exact: above the
  tracking cost's gradient (at most `2 |v| ≈ 2` for a 1 m/s field) every
  row the box allows holds with `s = 0`, so the relaxed problem has the
  hard one's optimum whenever that exists, and otherwise returns the
  fastest recovery the box allows instead of nothing. (Solving the slack
  form always was slower and, with the box's 95 rows, OSQP sometimes ran
  out of iterations.) The torque rows are relaxed the same way only if they
  conflict with the box itself (a velocity limit approached faster than the
  torque rate can turn): what the arm can do comes first, then the margins.
  A failed solve brakes toward rest at the acceleration limit.
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
compensation." Every 1 ms (`ref_mode: step`; `hrg/lowlevel.py`):

```
qd_cmd = the plan's qdot_des, ramped from the previous plan's over the plan period
q_des  = q + qd_cmd dt
τ      = Kp (q_des − q) + Kd (qd_cmd − qdot) + M(q) qdd_cmd + g(q) + C(q, qdot) qdot + τ_f(qd_cmd)
```

* `Kp (q_des − q) = Kp dt qdot_des` is about 1 % of the Kd term (600 × 1 ms
  next to 50), the same direction: `Kp = 0`. The law is a damping (velocity)
  controller, `Kd = [50 ×3, 20 ×3, 10]`; position is fed back by the planner,
  which solves the QP at the measured state every 10 ms. States that command
  no motion (idle, reopen) hold the joints through the planner
  (`hold_gain (q_hold − q)`), not through Kp; holding a lifted object keeps
  the lift's goal pose (below).
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
* **[FR3]** `M(q) qdd_cmd` (`inertia_ff`): the command's acceleration fed
  forward through the mass matrix. The Kd loop alone follows a velocity
  command with the time constant `M_ii / Kd_i`: 30–60 ms on joints 1–4 at
  the gains that are quiet on the robot (measured 25–31 ms lag on joints 2
  and 4, `scripts/diagnose_tracking.py`). That lag is what the horizon had
  to cover, and at speed it is the distance `v × lag` by which a margin is
  overshot. With the feed-forward the arm follows the planned velocity
  profile by construction and `Kd` only corrects errors: the lag falls to
  2–9 ms on every joint (the plan period). The plans keep `M â` within the
  FR3's torque and its change within the torque rate (eq. 11), and the low
  level moves `M(q) qdd_cmd` to each plan's value at that rate (0.8 Nm per
  ms): a ramp within the period, not a step that libfranka rate-limits and
  that holds back the Kd term with it. The earlier first-order filter on `qdot_des` is gone: the ramp
  between plans is the filter, and it is exact. If no plan arrives for
  0.2 s the joints are held where they were (`stale_hold_kp` / `kd`).
* **[2F-85]** The Robotiq is position-servoed (and so is its sim), with a
  speed setting per command. `wdot_des` is realised as a stream of `move`
  set-points at 50 Hz, each a little ahead (`|wdot| × 2 periods`, at least
  1 mm) of the integrated width `w_ref` (kept within 1 cm of the measured
  width: no wind-up against a blocked finger) **at the speed `|wdot_des|`**
  (`lowlevel.WidthStream`). The fingers then move continuously at the
  commanded rate. The earlier stream sent each set-point at the gripper's
  full speed: the fingers raced to it and waited for the next one (a third
  of the stream stopped, measured), the twitching seen on the robot. The
  simulated gripper, for its part, restarted its set-point ramp at the
  (lagging) measured width on every streamed command, which left the servo
  3 mm ahead of the fingers per period and the fingers creeping at 17 mm/s
  whatever speed was asked for: that, not the controller, made every
  grasp in sim wait a second with the hand at the targets. It now
  re-targets a running motion like the real one.

### Grasp and lift — `controller.HRGController._plan`

Paper: "once the fingertips come within a specified threshold (1 cm) of the
target positions, the gripper closes, the normal force is gradually
increased until the object is firmly secured without slipping, and the
object is lifted."

* Trigger: both fingertips within 1 cm (sim) / 1.5 cm (robot,
  `configs/robot/controller.yaml`).
* **[2F-85]** The Robotiq takes its grip force before it closes: a `grasp`
  command at 30 N and the gripper's top speed (0.15 m/s; from the
  pre-shaped opening the pads meet the object after 2 cm, 0.15 s). While it
  closes the hand holds its pose relative to the object (a moving object is
  followed; see "Speed"). It succeeds if the fingers stop on something, between half
  the contact width and just inside the pre-grasp opening; otherwise the
  gripper opens and the reach starts again. If the object is lost during the
  lift or hold, the same happens.
* Lift: the TCP 15 cm straight up through the same QP (a pose field as the
  task, saturated at 0.5 m/s with gain 8 /s; `Γ` and the hand vs table /
  obstacles as constraints, plus 64 points of the held object vs the
  table, 2 cm: a point closer than that, as at the grasp, may only not come
  closer; demanding the margin back from rest was infeasible and the
  fallback raised the bowl the fastest way the box allowed, a kick through
  every joint at its acceleration limit that set off the oscillation on the
  robot; 0.4 s), then (not in the paper) **upright**: the
  hand pose that puts the object back in its upright pose (object `+z` up,
  the smallest tilt about the TCP, raised until it clears the table by
  3 cm), from the object's pose in the hand recorded when the gripper
  reported the grasp. Tracking refreshes it after the rise and after the
  turn when plausible (the grasped point still within 2 cm between the pads,
  the object turned at most 45°): a bowl or glass that pivoted in the
  fingers is corrected, twice at most. Then hold.
* **One goal from the grasp on.** The upright turn keeps the lift's goal
  position, and the hold keeps the final goal pose (the same pose field):
  the hand never gets a new goal where it happens to be when a state ends.
  The lift ends within 1 cm of its goal, still rising; re-anchored there
  (the upright goal at the current position, the hold at the joints at
  entry) it overshot that point and came back down, visibly on the robot
  (2026-11-02; 2 mm in the simulation, where the arm follows with less
  lag). Now the height rises monotonically to the goal and stays.

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
| `η_g, T_g, η_h, T_h, δ_gr, v_gr` | 0.15 rad, 0.05, 6 cm, 2 cm, 2 cm, 0.15 m/s |
| `v_const`, `δ_x` (eq. 5) | 0.5 m/s, 10 cm (the paper: 0.15 m/s, 5 cm); see "Speed" |
| `H`, margins | 0.10 s (the stopping time: `v <= a H`, see "Speed"); object 1 cm (paper), obstacles 5 cm in eq. (3) (paper), 2 cm in eq. (11) |
| Velocity, acceleration, torque limits | 0.8 × the FR3's position-dependent velocity limits within 2 / 2.5 rad/s; the FR3's 10 rad/s² (not in the paper), the width 2 m/s²; the plan's torque within 0.8 × the FR3's torque and torque rate (87 / 12 Nm, 1000 Nm/s); jerk 300 rad/s³ (the robot's torque lag) |
| Start poses "left-bottom, center-up, right-bottom" | `hrg/study.py`: the ready pose, and the gripper pointing down 27 cm to either side with the fingertips at rim height |
| GPU nearest-neighbour / via search | CPU (KD-tree, NumPy); plans take 3–15 ms |

## Smoothness: why the arm shook, and what shapes the motion — `hrg/profile.py`, `planner._null_task`

Speed (below) is about what bounds the motion. This is about its shape. Measured with
`scripts/motion_quality.py`, which records through the settling and, unlike
`diagnose_tracking.py`, keeps going after `hold` begins.

* **The arm's self-motion.** The hand's six tasks leave one joint direction free (two at the home
  pose, where joints 1 / 3 and 5 / 7 line up): joints 1 and 3 turning against each other, mostly.
  Only `1e-3 |z|²` priced it, so whenever a torque or jerk row bound, the QP poured motion into it:
  during a straight lift the plan itself swung joints 1 and 3 at ±0.5 rad/s, with the hand going
  nowhere. That direction is also the arm's lightest (0.064 kg m² at the lift, where joint 1 alone
  is 2.2). This is the "8 Hz joint 1 / 3 oscillation" of 2026-11-02. It is now a task,
  `|N (qdot − qdot_post)|²` with `N = V diag(λ² / (s² + λ²)) Vᵀ` from the TCP Jacobian's SVD,
  `λ = 0.03`: soft, so it stays continuous at a singularity, and narrow, because the weakest real
  direction has `s ≈ 0.18` (at `λ = 0.15` that was priced at 40 % and the lift stalled 1 cm short).
  After the grasp, `qdot_post` pulls toward the configuration at the grasp, along the self-motion.
* **The lift is a trajectory** (`PoseProfile`). A saturated field steps to top speed and brakes on
  a corner: 0.6 g, and the torque at 95 % of joint 2's limit. Now a minimum-jerk motion (the
  quintic re-planned every step from the reference's own position, velocity and acceleration, so a
  goal that moves stays continuous) with its error fed back through the pose field. The time left
  grows only when the *goal* moves; from the distance left it would stretch every motion. The
  error is read from where the reference was `lift_lag` ago: the arm trails a moving reference by
  its lag, and read as a fault that lag adds speed and overshoots the goal (4 mm).
* **The references of the reach are bounded** (`TaskAccelerationLimit`): the fingertip velocity
  moves toward the field's within `task_accel` / `task_jerk`, measured from what the last plan
  asked, so nothing winds up while the QP holds the hand back. The hand's turn and the posture
  field too; the posture field stepped to `3 (q_grasp − q)`, 3 rad/s per joint, at the first plan:
  the hand went to 0.82 m/s in a 0.5 m/s field. Below 6 m/s² the hand cuts the corners of the path
  around the object and the reach takes twice as long.
* **The Kd loop was nearly unstable.** With the FR3's torque lag, per-joint gains of 50 on joints 1
  and 3 leave a damping ratio of 0.02-0.09 near 15 Hz at the poses used, and go unstable at 1 % of
  random poses (poles of `M` + rotor inertia, the fitted delay and lag). `kd = [30, 50, 35, ...]`:
  0.12-0.18, none unstable. It can be that low because the feed-forward carries the motion.
* **Joint limits with room.** 8 rad/s², 200 rad/s³, 0.7 of the torque: limits the plans rarely
  reach, not a profile they ride. The rest of the torque is the feedback's.
* **The low level.** The velocity ramp lasts the plans' measured period (over a fixed 10 ms a late
  plan dropped `M a` to zero until it came); the held object's weight fades in over 0.3 s (it is
  still on the table at the grasp); the friction feed-forward's zero crossing is softer
  (`v_smooth` 0.05 rad/s).

## Speed: how the velocity is made, what sets it, what bounds it

The method is first order: every plan **creates** a velocity from fields
and the QP picks the joint velocity that tracks them under the constraints.
Nothing else decides how fast the arm goes.

**Created by the fields.** The fingertip field (eq. 5): `v_const` far from
the target, decaying over `δ_x`, linear near the target with gain
`K = 2 v_const / δ_x`. The alignment (oriented reach): `k θ` capped at
`ω_max` (3 /s, 1.5 rad/s). The posture task toward `q_grasp` (3 /s). The
width field (eq. 9, 0.15 m/s). After the grasp: the lift's pose field
(0.5 m/s, 8 /s; 2 rad/s, 5 /s), the joint hold (10 /s, 1 rad/s), home (2 /s).
The robot brakes about one torque lag (~25 ms) after its plan, so a hand
stopping from `v` overshoots by about `v × 25 ms`: 1.6 cm from 0.8 m/s in the
simulation with the robot's torque response, none from 0.5 m/s.
Raising these is what makes the robot move faster; everything below only
says how far they can be raised.

**Bounded by the QP (eq. 11).** The joint velocity limits (0.8 of the
FR3's own at the current joints, within 2 rad/s on joints 1–4 and 2.5 on
the wrist; the hardware brakes above 0.9 of the FR3's, and above 2 / 3) bind
only near the ends of the joints' ranges at the speeds here. The fingertip
speed is at most `v_const` also on a moving object. The distance rows bind: linearised
over `H`, a row lets the hand approach a margin at `(d − ε)/H`, so the hand
starts braking `v H` before any margin (5 cm at 0.5 m/s) and then closes in
with the time constant `H`. The required deceleration is `v/H`, so the
rows are only satisfiable while **`v <= a H`**, with `a` the hand's
acceleration (5–8 m/s² at the FR3's joint acceleration limits over the
arm's lever; less where the torque rows bind, and it builds up over a few
plans at the torque rate). And the eq. (5) decay itself must be feasible: its peak
deceleration is about `1.5 v_const² / δ_x`, 3.8 m/s² at 0.5 m/s and 10 cm.
(0.8 m/s with 8 cm asks for 12 m/s²: the plans then ride the acceleration
bounds and the arm visibly shakes; `v_const` was lowered from 0.8 to 0.5
for that reason, at a cost of 0.25 s on a reach.) The acceleration bounds
in the QP are what makes this consistent: without them the QP's velocity
could jump and the rows could demand more braking than the arm has; with
them, and the slack fallback, the QP always returns the fastest motion the
arm can make. The torque rows add what the per-joint box cannot see: the
joints' accelerations together, through `M(q)`, within the torque the FR3
has and the rate at which it changes it.

**Realised by the low level (Sec. II-D).** The velocity ramp between plans
and the inertia feed-forward make the arm follow the planned velocity
profile with 2–9 ms of lag (measured; 25–31 ms on joints 2 and 4 before,
the `M / Kd` time constant). The lag no longer sets `H`: `H` is the
stopping time above. A faster robot would come from a larger `a H`, i.e.
from more acceleration (the FR3's limits) or a longer horizon (braking
earlier).

**A moving object.** Five things had to hold for a carried object to be
followed without the arm shaking, each found by replaying a recorded carry
through the planner and counting the plans at the acceleration bounds:

1. The final-approach goal is continuous (`fields.corridor`): the path's
   end slides along the approach segment with the hand's alignment instead
   of switching between "to the pre-grasp" and "to the target" (the switch,
   with hysteresis, flipped several times a second on a moving object).
2. That goal is never behind the hand along the approach: a hand lagging a
   moving object had overshot the pre-grasp and was sent back to it every
   plan.
3. The path direction is read over 2 cm, not the first 1.5 mm segment
   (eqs. 4–5 above).
4. The object pose is extrapolated at its estimated velocity between
   perception frames (`lowlevel.PosePredictor`; 30 Hz in sim, ~24 Hz on
   the robot): held, the targets stepped every 3–4 plans and each step was
   answered at the acceleration bounds (plans at the bounds: 0.15–0.19 of
   them held, 0.07–0.12 extrapolated, the same as with the true pose).
5. The object's velocity enters the fields (target feed-forward) and its
   collision rows (drift), and while the fingers close the hand keeps its
   pose **relative to the object** (the lift's pose field toward
   `T_obj · T_obj⁻¹T_tcp|trigger`, without the object's rows): with the arm
   holding still for the 0.3 s close a moving object ran into the palm, and
   with the reach's fingertip tasks and object rows kept on, the closing
   pads dragged the hand sideways and tipped boxes over.

Beyond these, a chase at 0.35 m/s keeps the plans at the bounds 7–12 % of
the time (the bounds are doing their job), the arm follows with 0.05–0.1
rad/s of velocity error, and the carrier's release (the simulated hand-over
letting go when both pads hold the object) is the one remaining impulse.

**What is not in the chain.** The planner at 100 Hz (3–8 ms per plan,
8–13 ms with obstacles). On the robot FoundationPose at 24 Hz, with its
~60 ms filter lag, is the slowest link for a moving object, not the arm;
the predictor covers the frame interval, not the filter lag.

Measured in the real-time simulation (`scripts/diagnose_tracking.py`,
`docs/results.md`): the gold bowl from the left-bottom start is grasped
1.0 s after `v` (gripper close included) and lifted 0.5 s later; before
this work, 4.0 s and 1.3 s.

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
