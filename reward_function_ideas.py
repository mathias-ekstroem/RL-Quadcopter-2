import numpy as np


def get_reward_linalg(self):
    distance_to_target = np.linalg.norm(self.target_pos - self.sim.pose[:3])
    sum_acceleration = np.linalg.norm(self.sim.linear_accel)

    reward = (5. - distance_to_target) * 0.3 - sum_acceleration * 0.05

    return reward


def get_reward(self):
    """Uses current pose of sim to return reward."""

    # we need to punish moving in either x or y since we're interested in just gaining some altitude
    x_punish = abs(self.sim.pose[0] - self.target_pos[0])
    y_punish = abs(self.sim.pose[1] - self.target_pos[1])

    # give reward bigger the closer to target
    altitude_reward = 1 / (abs(self.target_pos[2] - self.sim.pose[2]) + 1)

    # reward_z = 1.0 - (self.target_pos[2] - self.sim.pose[2])
    # reward_z = 1 / (abs(self.target_pos[2] - self.sim.pose[2]) + 1)
    punish_rot1 = abs(self.sim.pose[3])
    punish_rot2 = abs(self.sim.pose[4])
    punish_rot3 = abs(self.sim.pose[5])
    # reward_vz = self.sim.v[2]
    reward_tf = self.sim.time
    # reward = reward_z + 0.1 * reward_tf - 0.1 * (punish_x + punish_y) - 0.1 * (
    #             punish_rot1 + punish_rot2 + punish_rot3)

    reward = 2 * altitude_reward - 0.01 * (x_punish + y_punish) + 0.1 * self.sim.time - 0.1 * (
            punish_rot1 + punish_rot2 + punish_rot3)

    if self.target_pos[2] == self.sim.pose[2]:
        print('giving the ultimate reward')
        reward = 100

    return reward
