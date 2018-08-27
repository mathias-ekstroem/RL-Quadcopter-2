import csv
import sys

import numpy as np

from agents.agent import DDPG
from takeoff_task import TakeoffTask


def write_to_csv(stuff_write):
    with open(file_output, 'w') as file:
        row_writer = csv.writer(file)
        row_writer.writerow(stuff_write)


def main():
    labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
              'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
              'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
    file_output = 'data.txt'

    # write initial row
    with open(file_output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)

    num_episodes = 1000
    run_time = 10.
    target_pos = np.array([0., 0., 10.])  # takeoff and stay in place
    init_pose = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    init_velocities = np.array([0.0, 0.0, 0.0])
    init_angle_velocities = np.array([0.0, 0.0, 0.0])

    task = TakeoffTask(init_pose=init_pose, target_pos=target_pos, runtime=run_time)
    agent = DDPG(task)

    best_score = -np.inf

    results_list = []
    rewards_list = []

    for i_episode in range(1, num_episodes + 1):
        state = agent.reset_episode()  # start a new episode
        count = 0
        total_reward = 0

        results = {x: [] for x in labels}
        rewards = []

        while True:
            action = agent.act(state)  # noise is added for exploration
            next_state, reward, done = task.step(action)

            total_reward += reward
            rewards.append(reward)

            agent.step(action, reward, next_state, done)
            state = next_state

            to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(
                action)
            for ii in range(len(labels)):
                results[labels[ii]].append(to_write[ii])

            write_to_csv(to_write)

            count += 1
            if done:
                score = total_reward / float(count) if count else 0.0

                results_list.append(results)
                rewards_list.append(rewards)

                if score > best_score:
                    best_score = score

                # plot every 200 episodes

                if i_episode % 200 == 0:
                    print('i should be plotting something now.')
                    print('episode {}'.format(i_episode))

                print("\rEpisode = {:4d}, score = {:7.3f}, best_score = {:7.3f}, reward for episode = {}".format(
                    i_episode,
                    score,
                    best_score,
                    total_reward),
                      end="")  # [debug]
                break
        sys.stdout.flush()


if __name__ == '__main__':
    main()
