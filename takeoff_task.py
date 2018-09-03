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

    # this one gave pretty good results
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

    def my_get_reward(self):
        # clip the rewards
        min_reward = -1
        max_reward = 1

        # right now this is handled by the distance to target variable
        # punish movement on the x-axis
        xy_punish_factor = -0.01
        # x_punish = xy_punish_factor * abs(self.target_pos[0] - self.sim.pose[0])
        # y_punish = xy_punish_factor * abs(self.target_pos[1] - self.sim.pose[1])

        rotation_punish_factor = -0.1
        rot_1_punish = rotation_punish_factor * abs(self.sim.pose[3])
        rot_2_punish = rotation_punish_factor * abs(self.sim.pose[4])
        rot_3_punish = rotation_punish_factor * abs(self.sim.pose[5])

        rotation_punish = sum([rot_1_punish, rot_2_punish, rot_3_punish])

        target_reward_factor = 0.8
        # try and change this function. The target does not only have to be on the z axis. We can just measure the
        # distance the target pose as a hole and give reward from that.
        # reward_dist_to_target = target_reward_factor * (1 / (abs(self.target_pos[2] - self.sim.pose[2]) + 1))
        reward_dist_to_target = target_reward_factor * (1 / (abs(self.target_pos - self.sim.pose[:3]) + 1)).sum()
        # get reward for staying in the air and continuing the simulation
        # time_reward_factor = 0.1
        # time_reward = time_reward_factor * self.sim.time

        # velocity punishment
        angular_v_punish_factor = -0.002
        angular_velocity_punish = angular_v_punish_factor * (abs(self.sim.angular_v)).sum()

        # punish high velocity
        velocity_punish_factor = -0.005
        velocity_punish = velocity_punish_factor * self.sim.v.sum()

        # reward = sum([x_punish, y_punish, rotation_punish, reward_dist_to_target])
        reward = sum([rotation_punish, reward_dist_to_target, angular_velocity_punish, velocity_punish])

        # make sure that the rewards are being clipped
        if reward > max_reward:
            return max_reward
        elif reward < min_reward:
            return min_reward

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.my_get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
