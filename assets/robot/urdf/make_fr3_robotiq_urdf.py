"""Write fr3_robotiq.urdf and fr3_robotiq_rigid.urdf from the MuJoCo model.

    python assets/scene/franka_fr3/urdf/make_fr3_robotiq_urdf.py

The arm part (franka_description, everything up to the flange fr3_link8) is
kept from the current fr3_robotiq.urdf; the gripper part (Robotiq 2F-85 on
the flange) is generated from fr3_robotiq_robot.xml so MuJoCo, Pinocchio and
Viser share one geometry. Rerun after changing the gripper in the XML (e.g.
its mounting angle).

* fr3_robotiq.urdf: one gripper joint, ``robotiq_driver_joint`` (driver
  angle, 0 = open .. 0.8 rad = closed); the rest of the linkage follows via
  <mimic> (multipliers fitted from the MuJoCo linkage). For Viser.
* fr3_robotiq_rigid.urdf: the gripper as a rigid payload (open). For the
  controllers' Pinocchio model: 7 joints, the arm only.
"""

import os

import mujoco as mj
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "..", "fr3_robotiq_robot.xml")

# Linkage joints follow the driver angle: angle = multiplier * driver.
# Fitted from a MuJoCo sweep (residual < 6 mrad); the couplers barely move.
MIMIC = {
    "left_driver": 1.0,
    "right_spring_link": 0.9996,
    "left_spring_link": 0.9997,
    "right_follower": -0.9628,
    "left_follower": -0.9626,
}


def fmt(v):
    return " ".join(f"{x:.9g}" for x in v)


def rpy(R):
    # URDF rpy: R = Rz(y) Ry(p) Rx(r)
    p = np.arcsin(-np.clip(R[2, 0], -1, 1))
    if abs(np.cos(p)) > 1e-9:
        r, y = np.arctan2(R[2, 1], R[2, 2]), np.arctan2(R[1, 0], R[0, 0])
    else:
        r, y = np.arctan2(-R[1, 2], R[1, 1]), 0.0
    return np.array([r, p, y])


def tf(pos, quat):
    T = np.eye(4)
    m = np.zeros(9)
    mj.mju_quat2Mat(m, np.asarray(quat, dtype=float))
    T[:3, :3] = m.reshape(3, 3)
    T[:3, 3] = pos
    return T


def origin(T):
    return f'<origin xyz="{fmt(T[:3, 3])}" rpy="{fmt(rpy(T[:3, :3]))}"/>'


def main():
    m = mj.MjModel.from_xml_path(XML)
    root = m.body("fr3_gripper").id
    bodies = [b for b in range(root, m.nbody)]
    name = lambda b: m.body(b).name

    def joint_of(b):
        return m.body_jntadr[b] if m.body_jntnum[b] else -1

    # Link frame = MuJoCo body frame shifted to its joint position.
    shift = {}
    for b in bodies:
        j = joint_of(b)
        shift[b] = tf(m.jnt_pos[j], [1, 0, 0, 0]) if j >= 0 else np.eye(4)

    def link_xml(b):
        S_inv = np.linalg.inv(shift[b])
        out = [f'  <link name="{name(b)}">']
        if m.body_mass[b] > 0:
            Ti = S_inv @ tf(m.body_ipos[b], m.body_iquat[b])
            I = np.diag(m.body_inertia[b])
            R = Ti[:3, :3]
            I = R @ I @ R.T
            out += [
                "    <inertial>",
                f'      <origin xyz="{fmt(Ti[:3, 3])}" rpy="0 0 0"/>',
                f'      <mass value="{m.body_mass[b]:.9g}"/>',
                f'      <inertia ixx="{I[0,0]:.6g}" ixy="{I[0,1]:.6g}" ixz="{I[0,2]:.6g}" '
                f'iyy="{I[1,1]:.6g}" iyz="{I[1,2]:.6g}" izz="{I[2,2]:.6g}"/>',
                "    </inertial>",
            ]
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b:
                continue
            Tg = S_inv @ tf(m.geom_pos[g], m.geom_quat[g])
            if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
                mesh = m.mesh(m.geom_dataid[g]).name.replace("robotiq_", "")
                geo = f'<mesh filename="meshes/robotiq/{mesh}.stl" scale="0.001 0.001 0.001"/>'
                # compiled mesh geoms are re-centred; undo with the mesh offset
                Tm = tf(m.mesh_pos[m.geom_dataid[g]], m.mesh_quat[m.geom_dataid[g]])
                Tg = Tg @ np.linalg.inv(Tm)
            elif m.geom_type[g] == mj.mjtGeom.mjGEOM_BOX:
                geo = f'<box size="{fmt(2 * m.geom_size[g])}"/>'
            else:
                continue
            tag = "visual" if m.geom_group[g] == 2 else "collision"
            rgba = m.geom_rgba[g] if m.geom_matid[g] < 0 else m.mat_rgba[m.geom_matid[g]]
            out.append(f"    <{tag}>")
            out.append(f"      {origin(Tg)}")
            out.append(f"      <geometry>{geo}</geometry>")
            if tag == "visual":
                out.append(f'      <material name=""><color rgba="{fmt(rgba)}"/></material>')
            out.append(f"    </{tag}>")
        out.append("  </link>")
        return "\n".join(out)

    def joint_xml(b, rigid):
        p = m.body_parentid[b]
        T = np.linalg.inv(shift[p]) @ tf(m.body_pos[b], m.body_quat[b]) @ shift[b]
        j = joint_of(b)
        short = name(b).replace("robotiq_", "")
        if j < 0 or rigid or "coupler" in short:
            jname = f"{name(b)}_joint" if j < 0 else m.joint(j).name
            return (f'  <joint name="{jname}" type="fixed">\n    {origin(T)}\n'
                    f'    <parent link="{name(p)}"/>\n    <child link="{name(b)}"/>\n  </joint>')
        lo, hi = m.jnt_range[j]
        jname = "robotiq_driver_joint" if short == "right_driver" else m.joint(j).name
        s = (f'  <joint name="{jname}" type="revolute">\n    {origin(T)}\n'
             f'    <parent link="{name(p)}"/>\n    <child link="{name(b)}"/>\n'
             f'    <axis xyz="{fmt(m.jnt_axis[j])}"/>\n'
             f'    <limit lower="{lo:.6g}" upper="{hi:.6g}" effort="100" velocity="2"/>\n')
        if short in MIMIC:
            s += f'    <mimic joint="robotiq_driver_joint" multiplier="{MIMIC[short]}" offset="0"/>\n'
        return s + "  </joint>"

    tcp = m.site("fr3_gripper_tcp")
    arm = open(os.path.join(HERE, "fr3_robotiq.urdf")).read()
    arm = arm[: arm.index("  <!-- Generated by make_fr3_robotiq_urdf.py")]
    for rigid, fname in ((False, "fr3_robotiq.urdf"), (True, "fr3_robotiq_rigid.urdf")):
        parts = [arm.rstrip(), f"  <!-- Generated by make_fr3_robotiq_urdf.py from ../fr3_robotiq_robot.xml. "
                 + ("Gripper rigid (open): controllers' model." if rigid else "Gripper articulated (mimic): Viser.") + " -->",
                 '  <joint name="fr3_gripper_joint" type="fixed">\n'
                 f'    {_flange(m, root)}\n'
                 '    <parent link="fr3_link8"/>\n    <child link="fr3_gripper"/>\n  </joint>',
                 '  <link name="fr3_gripper_tcp"/>\n  <joint name="fr3_gripper_tcp_joint" type="fixed">\n'
                 f'    <origin xyz="{fmt(m.site_pos[tcp.id])}" rpy="0 0 0"/>\n'
                 '    <parent link="fr3_gripper"/>\n    <child link="fr3_gripper_tcp"/>\n  </joint>']
        for b in bodies:
            parts.append(link_xml(b))
            if b != root:
                parts.append(joint_xml(b, rigid))
        parts.append("</robot>\n")
        open(os.path.join(HERE, fname), "w").write("\n".join(parts))
        print("wrote", fname)


def _flange(m, root):
    # fr3_gripper relative to fr3_link8 (= link7 + 0.107 z): rotation only.
    T = tf(m.body_pos[root], m.body_quat[root])
    T[:3, 3] -= np.array([0, 0, 0.107])
    return origin(T)


if __name__ == "__main__":
    main()
