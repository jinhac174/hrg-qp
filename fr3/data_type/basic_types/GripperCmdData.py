from fr3.data_type.BaseData import BaseData


class GripperCmdData(BaseData):
    """Gripper command (see lcm_type/lcm_type/gripper_cmd.lcm).

    Mirrors the real gripper API: MOVE(width, speed), GRASP(width, speed,
    force, eps_inner, eps_outer), STOP, HOMING. ``seq`` must change for every
    new command; receivers execute each seq once.
    """

    MOVE = 0
    GRASP = 1
    STOP = 2
    HOMING = 3

    def __init__(self, name: str = "gripper_cmd_data"):
        super().__init__(name)
        self.seq = 0
        self.mode = self.STOP
        self.width = 0.0
        self.speed = 0.1
        self.force = 0.0
        self.epsilon_inner = 0.005
        self.epsilon_outer = 0.005

    def set_data(
        self,
        t: float,
        seq: int,
        mode: int,
        width: float = 0.0,
        speed: float = 0.1,
        force: float = 0.0,
        epsilon_inner: float = 0.005,
        epsilon_outer: float = 0.005,
    ):
        self.timestamp = t
        self.seq = int(seq)
        self.mode = int(mode)
        self.width = float(width)
        self.speed = float(speed)
        self.force = float(force)
        self.epsilon_inner = float(epsilon_inner)
        self.epsilon_outer = float(epsilon_outer)

    def get_data(self):
        return (
            self.timestamp,
            self.seq,
            self.mode,
            self.width,
            self.speed,
            self.force,
            self.epsilon_inner,
            self.epsilon_outer,
        )
