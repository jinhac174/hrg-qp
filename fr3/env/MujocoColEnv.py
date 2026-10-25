"""MuJoCo env that also publishes collision geometry for planning."""

import copy

from fr3.data_type.basic_types.ColInfoData import ColInfoData
from fr3.env.MujocoBaseEnv import MujocoBaseEnv
from fr3.utils.collision.wall_geoms import extract_static_col_geoms


class MujocoColEnv(MujocoBaseEnv):
    """Robot env that additionally publishes collision info for
    collision-aware control:

    - ``robot_col_info`` (intr): per-link primitives from the platform's
      ``get_col_info_data()`` (link-local offsets; controllers place them by FK).
    - ``static_col_info`` (extr): static scene geoms (``floor``, ``wall_*``,
      ``table`` / ``table_*``, ``obs_*``), auto-derived from the SAME MuJoCo
      scene you tune in the XML, so there is one source of truth. FCL has no
      plane, so the floor plane is exported as a large thin box whose top sits
      at the plane height.

    Both are republished every ``col_info_update_dt`` seconds. Static obstacles
    are world-fixed so their offset is constant, but re-sending keeps a late-
    joining controller supplied.
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self._static_col_info_channel = config["pub_manager"]["static_col_info_channel"]
        self._robot_col_info_channel = config["pub_manager"]["robot_col_info_channel"]

        self.static_col_info_data = ColInfoData(name=self._static_col_info_channel)
        self.robot_col_info_data = ColInfoData(name=self._robot_col_info_channel)

        # Static obstacles: derive from the scene (single source of truth).
        for geom in extract_static_col_geoms(self.mj_model):
            self.static_col_info_data.add_col_geom(**geom)

        # Robot primitives: per-link, from the platform.
        for dict_col_info in self.platform.get_col_info_data():
            self.robot_col_info_data.add_col_geom(**dict_col_info)

        self._col_info_update_dt = float(config.get("col_info_update_dt", 0.1))
        self._last_col_info_update_time = -1e9

    def reset(self) -> None:
        super().reset()
        # Republish right away after a reset (sim time restarts at 0).
        self._last_col_info_update_time = -1e9

    def _sync_data_from_sim(self, intr_pub_que_dict, extr_pub_que_dict, time_pub_que):
        super()._sync_data_from_sim(intr_pub_que_dict, extr_pub_que_dict, time_pub_que)
        t = self.mj_data.time
        if t - self._last_col_info_update_time > self._col_info_update_dt:
            self._last_col_info_update_time = t
            self.static_col_info_data.set_time(t)
            self.robot_col_info_data.set_time(t)
            extr_pub_que_dict[self._static_col_info_channel].put(
                copy.deepcopy(self.static_col_info_data)
            )
            intr_pub_que_dict[self._robot_col_info_channel].put(
                copy.deepcopy(self.robot_col_info_data)
            )
