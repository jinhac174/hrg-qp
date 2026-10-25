from fr3.data_type.BaseData import BaseData


class GripperStateData(BaseData):
    """Gripper state (see lcm_type/lcm_type/gripper_state.lcm)."""

    def __init__(self, name: str = "gripper_state_data"):
        super().__init__(name)
        self.width = 0.0
        self.max_width = 0.085  # Robotiq 2F-85
        self.is_grasped = False
        self.busy = False
        self.last_seq = -1
        self.last_success = False

    def set_data(
        self,
        t: float,
        width: float,
        max_width: float,
        is_grasped: bool,
        busy: bool,
        last_seq: int,
        last_success: bool,
    ):
        self.timestamp = t
        self.width = float(width)
        self.max_width = float(max_width)
        self.is_grasped = bool(is_grasped)
        self.busy = bool(busy)
        self.last_seq = int(last_seq)
        self.last_success = bool(last_success)

    def get_data(self):
        return (
            self.timestamp,
            self.width,
            self.max_width,
            self.is_grasped,
            self.busy,
            self.last_seq,
            self.last_success,
        )
