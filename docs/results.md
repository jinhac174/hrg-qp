# Simulation results (FR3 + 2F-85, MuJoCo)

The paper's simulation studies (Sec. III-A: Table I, Figs. 6-8) repeated with
this implementation on the Franka FR3 + Robotiq 2F-85. The paper's robot is a
15-DoF arm-hand; the numbers below are ours on the FR3, not the paper's, and
the paper's are quoted for comparison only.

Everything here comes from `scripts/eval_sim.py` on a MacBook (Apple
Silicon), the real-time simulation running alone on the machine (one trial at
a time: `run_env.py` + `run_controller.py` over LCM, the planner at 100 Hz,
the control loop at 1 kHz, 20 s per trial as in the paper, 30 s for the
disturbance and obstacle studies), with the parameters of
`configs/controller.yaml` (planning horizon `H = 0.10` s, see
docs/method.md eq. 11). The commands:

```bash
python scripts/eval_sim.py --objects box blue_bowl blue_mug wine_glass --orientations all \
    --seeds 5 --fields path linear --out outputs/eval/table1                         # Table I, 360 trials
python scripts/eval_sim.py --objects blue_bowl blue_mug --poses center_up --seeds 3 \
    --push_at 1.5 4.0 --jump_at 3.0 --timeout 30 --out outputs/eval/fig7              # Fig. 7
python scripts/eval_sim.py --env configs/sim/env_obstacles.yaml --objects wine_glass blue_bowl \
    --seeds 3 --timeout 30 --out outputs/eval/fig8                                     # Fig. 8
```

Each trial directory holds the env / controller logs and `trace.npz`
(fingertip and target centres over time for `scripts/plot_trace.py`; the
measured state for `scripts/dissect.py`). A trial counts as a success when
the object is held (`is_grasped`) 5 cm above its rest height within the time
limit; trials where the bus or a process died are marked `infra` and
repeated, never counted.

## The oriented reach and the paper's low level (2026-10-28)

The sections after this one were measured before 2026-10-28, with what were
then the defaults: the paper's reach (fingertip points only) and the stiff
low level (a running-integral reference with Kp). To reproduce them, add
`--ctrl hrg.orient=false ref_mode=integral kp=[600,600,600,600,250,150,50]`.
The defaults are now the oriented reach, the paper's low level (Kp = 0) and
friction feed-forward (docs/method.md). Everything below was run on the lab
workstation (rllab2).

**Kinematic rollouts** (`scripts/rollout.py --orientations all --placements
5`): perfect tracking, the Table I objects × orientations × start poses.
Converged = both fingertips within 1 cm, and with the oriented reach also
the hand within 10° of its target orientation (the grasp trigger).

| | converged | hand vs R* at the end | joint 7 turn |
|---|---|---|---|
| paper's reach (`hrg.orient=false`) | 172 / 178 | (not steered) | — |
| oriented reach | 169 / 178 | ≤ 10° (the trigger) | ≤ 110° |

The 9 failures are all a mug lying on its side:
- **6 placements, no reachable oriented grasp:** the planner reports it and holds. Every
  approach that is collision-free and swept clear needs joint 6 past its limit.
- **3 placements, stalled 5 cm short:** link 6 sits at its self-collision margin against
  link 1. The chosen grasp configuration itself is clear, but the fingertip fields drive
  along a Cartesian path the arm cannot follow there. That is the paper's own limitation:
  fingertip targets planned independently of the joints.

Upright objects are approached from above in their own frame, and objects on their side
from their own "above" where reachable. A wine glass on its side is never approached
through its opening at table height: link 6 would meet the table and joint 2 its limit,
so its stem or foot is taken.

**Real-time simulation** (`scripts/eval_sim.py --objects box blue_bowl gold_bowl blue_mug
wine_glass --orientations all --seeds 2 --timeout 30`): grasp + lift within 30 s, then the
object's tilt once held, after the upright turn.
- **78 / 78** grasped, lifted and still held at the end, with no failed closes.
- Lift time: median 5.4 s, at most 8.5 s after `v`.
- The object's tilt at the end: at most 5.4° (median 2.8°), including the 48 trials that
  started lying on their side (83–89°).
- Gold bowl: 6 / 6, ending within 2° of upright.

**Friction identification in the simulation**
(`scripts/robot/identify_friction.py`, against MuJoCo's own friction):
joint 2 Coulomb 1.152 Nm (model 1.137), joint 5 0.763 (0.763), joint 7
0.248 (0.248); viscous 0.14–0.28 Nm s/rad (model 0.21).

## Table I: success rates

Paper's protocol: three start poses (left-bottom, center-up, right-bottom),
five placements each; the mug and the wine glass in five orientations
(upright, ±90° about x, ±90° about y), the others upright. `path` is the
method (eqs. 3-5 fingertip fields), `linear` the eq. (2) baseline, both
through the same eq. (11) QP.

| Object | Trials | **path** (this work, FR3) | **linear** (this work, FR3) | paper "Ours" | paper "Linear" |
|---|---|---|---|---|---|
| Box | 15 | **100.0 %** (15/15) | 100.0 % (15/15) | 100.0 % | 100.0 % |
| Bowl | 15 | **100.0 %** (15/15) | 100.0 % (15/15) | 100.0 % | 33.3 % |
| Mug | 75 | **96.0 %** (72/75) | 70.7 % (53/75) | 100.0 % | 30.7 % |
| Wine glass | 75 | **97.3 %** (73/75) | 82.7 % (62/75) | 100.0 % | 32.0 % |
| Dish | - | not run (no 2F-85 grasp satisfies the paper's margins, docs/method.md) | | 100.0 % | 93.3 % |
| **All** | 180 | **97.2 %** (175/180) | 80.6 % (145/180) | | |

By orientation (successes / 15, the three start poses × five placements):

| | upright | rx+90 | rx−90 | ry+90 | ry−90 |
|---|---|---|---|---|---|
| Mug, path | 15 | 15 | **12** | 15 | 15 |
| Mug, linear | 9 | 12 | 7 | 13 | 12 |
| Wine glass, path | 15 | 15 | 15 | 15 | **13** |
| Wine glass, linear | 15 | 13 | 11 | 13 | 10 |

Time from `v` to the object lifted: median 4.7 s (path), 7.5 s (linear).
Planning time per cycle: median 8.0 ms (path: eq. 3 via search + 3 SQP
iterations + eq. 11), 2.3 ms (linear), 12.8 ms with the four moving
obstacles, inside the 10 ms planning period except with obstacles.
Control-loop rates logged per trial: joint measurements 950-1000 Hz,
commands 700-800 Hz (the Python control thread on the laptop; on the robot
libfranka's 1 kHz applies).

**Why our linear baseline is stronger than the paper's.** The 2F-85's two
pads are a rigid pair 85 mm apart at most, and its rim and handle targets
are 1.5 cm off the surface; sliding a parallel jaw down over a rim reaches
most of them without any detour, so the eq. (2) attractor succeeds on the
bowl always and on the upright wine glass always. Where it fails it fails
the way the paper's Fig. 6 shows: trapped against the object. Every one of
its 35 failures ends with hand spheres at their 1 cm object margin
(`scripts/rollout.py --fields linear` prints what binds): the finger pushed
into the mug's body or handle, or the glass's bowl, with the fingertips
0.3-6 cm from their targets. Since the paper's own fingertips are
independent, its linear baseline has more ways to get stuck.

## Fig. 6: fingertip error at convergence

Centre error of the two fingertips to the selected pair when the gripper
closed (path field, 175 successes): median 0.37 cm, 90th percentile 0.54 cm,
maximum 0.91 cm; the closing trigger is 1 cm per fingertip (paper: 1 cm).
Linear, 145 successes: median 0.39 cm, maximum 0.96 cm. Linear failures:
median 0.63 cm, up to 6.1 cm (above). The paper's "mostly below 1 cm" for its
method, and its linear failures spread above the threshold, are what we see.

## Fig. 7: disturbances

Six trials (bowl and mug, center-up start, three placements): a push on the
hand at 1.5 s (a random horizontal 80 N force for 0.3 s), the object moved
to a new random pose at 3 s, a second push at 4 s. **6 of 6 lifted** the
object, in 6.3-7.9 s except one at 13.0 s. The traces
(`outputs/eval/fig7/*/trace.png`, like the paper's Fig. 7) show the
fingertip centre pushed away and returning, the target curves jumping when
the object moves, and the re-convergence; the target also changes slightly
during the pushes (eq. 12 re-selects as the hand moves), as the paper notes.

The 13.0 s trial (mug, seed 2) is the one where the second push drives the
hand into the mug and knocks it over (the recorded object height goes from 0
to 3.4 cm, its axis to 45°): the controller re-targets the lying mug at once
(`trace.png`: the target z drops at 4.3 s) and gets there with joint 5 near
its limit. This trial is marginal: with `H = 0.10` s it lifted in two of
three runs (13.0 s and 13.2 s) and stalled 1 cm short in the third (video
19), with the earlier `H = 0.05` s it was likewise a coin flip. A push that
knocks the object over is outside the paper's protocol; the reach itself
recovers every time.

## Fig. 8: moving obstacles

Four boxes oscillating around the object (`configs/sim/env_obstacles.yaml`,
exact box distance functions, 5 cm margin in the paths, 2 cm in the QP),
wine glass and bowl, three start poses × three placements: **18 of 18
lifted, no robot-obstacle contact in any trial** (the env counts contacts
between robot and obstacle geometry every step).

## The failures, dissected

All five failures of the method reproduce in a kinematic replay with
perfect tracking from the recorded state (`scripts/dissect.py`), so they
are properties of the method on this robot, not of the simulation's
dynamics or timing:

| Trial | Object pose | Short by | What binds at the end |
|---|---|---|---|
| mug rx−90, left-bottom, seed 2 | lying, 48 cm ahead, 4 cm right of the base | 0.5 / 1.1 cm | joint 4 at −3.04 rad (its limit); link6-link1 at the 2 cm self-collision margin; finger spheres at the 1 cm object margin |
| mug rx−90, left-bottom, seed 3 | lying, 46 cm / 5 cm right | 0.8 / 1.4 cm | the same set, plus linkage and palm spheres |
| mug rx−90, right-bottom, seed 3 | lying, 46 cm / 5 cm right | 1.0 / 0.2 cm | joint 4 at its limit; link6-link1 and link5-link2 at their margins; finger spheres at the object margin |
| wine glass ry−90, right-bottom, seed 3 | lying, 46 cm / 2 cm right | 2.1 / 0.7 cm | joint 4 at its limit; link6-link1 and link5-link2 at their margins; finger spheres at the object margin |
| wine glass ry−90, center-up, seed 4 | lying, 54 cm ahead, yaw 171° | 1.0 / 0.2 cm | no joint limit: finger spheres of both fingers at the object margin (the pads cannot close on the pair from this hand orientation) |

Four of the five are the pattern the paper describes in Fig. 12 and
Sec. IV: the fingertip targets are planned without the joint configuration,
and when the object lies on its side close to the base the FR3 has to fold
its elbow fully to bring the pads in horizontally; the joint limit, the
self-collision margin and the hand's object margin become active together
and the QP's best feasible motion is zero, 1-2 cm from the trigger. The
fifth is the same class with the hand's own geometry: the finger bodies
reach the 1 cm margin before the pad centres reach a pair 1.5 cm off the
surface, and nothing in the fingertip-level fields asks for the roll about
the finger axis that would free them. The paper's remedy (velocity targets
for the elbow, Sec. IV) is future work there as here; no such term has been
added. Every plan reports this set in `ReachStep.info['binding']` and the
controller prints it with its periodic plan-time line, so the same question
can be answered on the robot.

These stalls are marginal: 1-3 mm beyond the 1 cm trigger for one
fingertip. With the earlier horizon `H = 0.05` s the hand overshot its
margin by several millimetres and bounced at 3-4 Hz, and three of these
trials then "succeeded" in some runs by that overshoot (177/180 in the
first validation run). With `H = 0.10` s the hand holds its margin and the
outcome is repeatable; the numbers above are the ones to quote.

## Videos

`outputs/videos/` (`scripts/record_video.py`, offscreen MuJoCo with the
fingertips, their targets and the planned paths drawn in, close-up inset).
The recorder samples the bus at 30 Hz during the trial and renders the
frames afterwards, so the simulation runs undisturbed and the video is real
time with exact frame timing. Side-by-side videos run the path field (left)
and the eq. (2) linear field (right) from the same start and placement,
chosen among the Table I placements where the baseline failed.

1. `1_box.mp4`: the box from the center-up start.
2. `2_bowl_from_above.mp4`, 3. `3_bowl_from_the_side.mp4`: the bowl, rim pinch, from the center-up and the left-bottom start.
4. `4_mug_upright.mp4`: the upright mug from the center-up start.
5. `5_mug_upright_from_the_side_path_vs_linear.mp4`: upright mug from the right-bottom start; the linear field stalls 3.8 cm out.
6. `6_mug_lying_rx+90_path_vs_linear.mp4`: mug on its side (+90° about x); linear stalls 5.8 cm out against the body.
7. `7_mug_lying_rx-90_path_vs_linear.mp4`: mug on its side (−90° about x) from the left-bottom start; linear stalls 6.1 cm out.
8. `8_mug_lying_ry+90.mp4`: mug on its side (+90° about y).
9. `9_mug_lying_ry-90_path_vs_linear.mp4`: mug on its side (−90° about y); linear stalls 2.5 cm out.
10. `10_wine_glass_upright_rim_pinch.mp4`: the upright wine glass, rim pinch.
11. `11_wine_glass_lying_rx+90_path_vs_linear.mp4`: lying glass; the linear field only creeps in after 20 s.
12. `12_wine_glass_lying_rx-90_path_vs_linear.mp4`, 13. `13_wine_glass_lying_ry+90_path_vs_linear.mp4`: lying glass; linear stuck.
14. `14_wine_glass_lying_ry-90.mp4`: lying glass (−90° about y).
15. `15_red_cup.mp4`, 16. `16_tape.mp4`: the real-experiment objects (paper Sec. III-B).
17. `17_disturbances_bowl.mp4`, 18. `18_disturbances_mug.mp4`: Fig. 7 on both objects (a push at 1.5 s, the object moved at 3 s, a push at 4 s).
19. `19_disturbance_knocks_the_mug_over.mp4`: the Fig. 7 trial whose second push knocks the mug over; the controller re-targets the lying mug and, in this recording, stalls 1 cm short at joint 5's limit (it lifted in two of three validation runs).
20. `20_moving_obstacles_wine_glass.mp4`, 21. `21_moving_obstacles_bowl.mp4`, 22. `22_moving_obstacles_wine_glass_from_the_side.mp4`: Fig. 8 from two starts, no contact with the boxes.
23. `23_marginal_lying_mug_near_the_base.mp4`: the Table I failure case (lying mug, left-bottom, seed 2): the elbow at its limit, the fingertips about 1 cm from the trigger, the hand resting still on its margin.
