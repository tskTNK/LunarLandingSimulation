import gym
import torch
import time

import numpy as np
import scipy as sp
import scipy.sparse.linalg
from scipy.sparse import linalg

def demo_ddpg_lander(env, seed=None, render=False):
    env.seed(seed)
    total_reward = 0
    steps = 0
    s = env.reset()
    mu = torch.load('log/lander/models1/model_3100')
    mu = mu.eval()
    while True:

        s = torch.tensor(s)
        a = torch.squeeze(mu(s)).data.numpy()

        FPS = 50
        SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well
        VIEWPORT_W = 600
        VIEWPORT_H = 400

        gravity = 9.8/FPS/FPS # gravity is enhanced by scaling
        thrust_main_max = gravity/0.56
        thrust_side_max = thrust_main_max*0.095/0.7 # m/frame^2 # determined by test
        m_main_inv = thrust_main_max    # gravity*0.57
        m_side_inv = thrust_side_max    # gravity*0.225
        a_i_inv= 0.198/100 # rad/frame^2 # determined by test # not depend on SCALE
        align = 0.87   # 0.87 = sin30

        # target point set
        x_target = 0
        y_target = 0   # the landing point is 0
        Vx_target = 0
        Vy_target = 0
        theta_target = 0
        omega_target = 0

        if a < env.action_space2.low:
            a = env.action_space2.low
        elif a > env.action_space2.high:
            a = env.action_space2.high

        """
        if a < 1.0:
            a = 1.0
        elif a > 3:
            a = 3
        """

        a_float = float(a)
        y_target = s[1]*(VIEWPORT_H/SCALE/2)/a_float # 1.6 succeeds all the times

        X = np.array([ \
        [s[0]*(VIEWPORT_W/SCALE/2)-x_target], \
        [s[1]*(VIEWPORT_H/SCALE/2)-y_target], \
        [s[2]/(VIEWPORT_W/SCALE/2)-Vx_target], \
        [s[3]/(VIEWPORT_H/SCALE/2)-Vy_target], \
        [s[4]-theta_target], \
        [s[5]/20.0-omega_target]])

        # print("X {}\n".format(X))

        A = np.array([ \
        [0, 0, 1, 0, 0, 0], \
        [0, 0, 0, 1, 0, 0], \
        [0, 0, 0, 0, -1*gravity, 0], \
        [0, 0, 0, 0, 0, 0], \
        [0, 0, 0, 0, 0, 1], \
        [0, 0, 0, 0, 0, 0]])

        B = np.array([ \
        [0, 0], \
        [0, 0], \
        [0, m_side_inv*align], \
        [1*m_main_inv, 0], \
        [0, 0], \
        [0, -1*a_i_inv]])

        sigma = np.array([ \
        [0], \
        [0], \
        [0], \
        [-1*gravity], \
        [0], \
        [0]])

        # gravity compensation
        BTB = np.dot(B.T, B)
        u_sigma = -1*np.linalg.inv(BTB).dot(B.T).dot(sigma)

        # Design of LQR
        # Solve Riccati equation to find a optimal control input
        R = np.array([ \
        [1, 0], \
        [0, 1]])

        Q = np.array([ \
        [1, 0, 0, 0, 0, 0], \
        [0, 1, 0, 0, 0, 0], \
        [0, 0, 1, 0, 0, 0], \
        [0, 0, 0, 1, 0, 0], \
        [0, 0, 0, 0, 100, 0], \
        [0, 0, 0, 0, 0, 100]])

        # Solving Riccati equation
        P = sp.linalg.solve_continuous_are(A, B, Q, R)
        # print("P {}\n".format(P))

        # u = -KX
        # K = R-1*Rt*P
        K = np.linalg.inv(R).dot(B.T).dot(P)
        thrust = -1*np.dot(K, X) + u_sigma

        BK = np.dot(B, K)
        A_ = A - BK
        a_eig = np.linalg.eig(A_)
        a_sort = np.sort(a_eig[0])
        # print("eigen values {}\n".format(a_sort))

        # print("thrust {}\n".format(thrust))
        # thrust[0] = 0
        # thrust[1] = 1

        if s[1] < 0.3/SCALE:
            thrust[0] = 0
            thrust[1] = 0

        # conversion to compensate main thruster's tricky thrusting
        thrust[0] = thrust[0]/0.5-1.0

        a_updated = np.array([thrust[0], thrust[1]])
        a_updated = np.clip(a_updated, -1, +1)  #  if the value is less than 0.5, it's ignored
        # print("a_updated * {}\n".format(a_updated))

        # print("s:","{} {} {} {} {}".format(s[0], s[1], s[2], s[3], s[4]))
        # print("a {}\n".format(a), "actions:","{} {}".format(a_updated[0], a_updated[1]))

        # s, r, done, info = env.step(a)
        s, r, done, info = env.step(a_updated)
        total_reward += r

        if render:
            still_open = env.render()
            if still_open == False: break

        if steps % 1 == 0 or done:
        # if done:
            # print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            # print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            print("step {} reward {:+0.2f}".format(steps, total_reward),"observations:", " ".join(["{:+0.2f}".format(x) for x in s]), "actions:","{} {}".format(a_updated[0], a_updated[1]), "k:","{}".format(a_float))
        steps += 1
        if done: break

    return total_reward

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    time.sleep(3)

    for i in range(1):
        demo_ddpg_lander(env, seed=i, render=True)
