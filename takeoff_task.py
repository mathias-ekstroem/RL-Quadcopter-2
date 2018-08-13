import numpy as np

from physics_sim import PhysicsSim


class TakeoffTask:
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward_linalg(self):
        distance_to_target = np.linalg.norm(self.target_pos - self.sim.pose[:3])
        sum_acceleration = np.linalg.norm(self.sim.linear_accel)

        reward = (5. - distance_to_target) * 0.3 - sum_acceleration * 0.05

        return reward

    def get_reward_ed(self):
        """Uses current pose of sim to return reward."""
        max_reward = 1
        min_reward = -1

        ed = (abs(self.sim.pose[:3] - self.target_pos)).sum()  # euclidian distance
        avd = (abs(self.sim.angular_v)).sum()  # angular v
        vd = (abs(self.sim.v)).sum()  # velocity

        reward = 1. - ed / 519. - avd / 20. - vd / 6000.

        reward = np.maximum(np.minimum(reward, max_reward), min_reward)
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

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward_ed()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
