from networks import Actor, Critic
from noise import OrnsteinUhlenbeck
from buffer import Buffer
import torch
import numpy as np
import copy
import os

import scipy as sp
import scipy.sparse.linalg
from scipy.sparse import linalg


class DDPG():
    def __init__(self, env, log_dir, gamma=0.99, batch_size=64, sigma=0.2, batch_norm=True, merge_layer=2,
                 buffer_size=int(1e6), buffer_min=int(1e4), tau=1e-3, Q_wd=1e-2, num_episodes=1000):

        self.s_dim = env.reset().shape[0]
        # self.a_dim = env.action_space.shape[0]
        self.a_dim = env.action_space2.shape[0]
        # self.a_dim = 1

        self.env = env
        # self.mu = Actor(self.s_dim, self.a_dim, env.action_space, batch_norm=batch_norm)
        self.mu = Actor(self.s_dim, self.a_dim, env.action_space2, batch_norm=batch_norm)
        self.Q = Critic(self.s_dim, self.a_dim, batch_norm=batch_norm, merge_layer=merge_layer)
        self.targ_mu = copy.deepcopy(self.mu).eval()
        self.targ_Q = copy.deepcopy(self.Q).eval()
        self.noise = OrnsteinUhlenbeck(mu=torch.zeros(self.a_dim), sigma=sigma * torch.ones(self.a_dim))
        self.buffer = Buffer(buffer_size, self.s_dim, self.a_dim)
        self.buffer_min = buffer_min
        self.mse_fn = torch.nn.MSELoss()
        self.mu_optimizer = torch.optim.Adam(self.mu.parameters(), lr=1e-4)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=1e-3, weight_decay=Q_wd)

        self.gamma = gamma
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.tau = tau
        self.log_dir = log_dir

        self.fill_buffer()

    #updates the target network to slowly track the main network
    def track_network(self, target, main):
        with torch.no_grad():
            for pt, pm in zip(target.parameters(), main.parameters()):
                pt.data.copy_(self.tau*pm.data + (1-self.tau)*pt.data)

    # updates the target nets to slowly track the main ones
    def track_networks(self):
        self.track_network(self.targ_mu, self.mu)
        self.track_network(self.targ_Q, self.Q)

    def run_episode(self):
        done = False
        s = torch.tensor(self.env.reset().astype(np.float32), requires_grad=False)
        t = 0
        tot_r = 0
        while not done:

            self.mu = self.mu.eval()
            # a_ = torch.squeeze(self.mu(s)).detach().numpy()
            a = torch.squeeze(self.mu(s)).detach().numpy()
            # print("a {}\n".format(a))

            self.mu = self.mu.train()

            ac_noise = self.noise().detach().numpy()
            a = a + ac_noise
            # print("ac_noise {}\n".format(ac_noise))
            # print("a+ac_noise {}\n".format(a))

            if a < self.env.action_space2.low:
                a = self.env.action_space2.low
            elif a > self.env.action_space2.high:
                a = self.env.action_space2.high

            s = s.detach().numpy()

            a_updated = self.LQR(s, a)
            # s_p, r, done, _ = self.env.step(a)
            s_p, r, done, _ = self.env.step(a_updated)

            tot_r += r
            self.buffer.add_tuple(s,a,r,s_p,done)

            s_batch, a_batch, r_batch, s_p_batch, done_batch = self.buffer.sample(batch_size=self.batch_size)

            # update critic
            with torch.no_grad():
                q_p_pred = self.targ_Q(s_p_batch, self.targ_mu(s_p_batch))
                q_p_pred = torch.squeeze(q_p_pred)
                y = r_batch + (1.0 - done_batch)*self.gamma*q_p_pred
            self.Q_optimizer.zero_grad()
            q_pred = self.Q(s_batch, a_batch)
            q_pred = torch.squeeze(q_pred)
            #print(torch.mean(q_pred))
            Q_loss = self.mse_fn(q_pred, y)
            Q_loss.backward(retain_graph=False)
            self.Q_optimizer.step()

            # update actor
            self.mu_optimizer.zero_grad()
            q_pred_mu = self.Q(s_batch, self.mu(s_batch))
            q_pred_mu = torch.squeeze(q_pred_mu)
            #print(torch.mean(q_pred_mu))
            mu_loss = -torch.mean(q_pred_mu)
            # print(mu_loss)
            mu_loss.backward(retain_graph=False)
            #print(torch.sum(self.mu.layers[0].weight.grad))
            self.mu_optimizer.step()
            self.track_networks()

            s = torch.tensor(s_p.astype(np.float32), requires_grad=False)
            t += 1
        return tot_r, t

    def train(self):
        results = []
        for i in range(self.num_episodes):
            r, t = self.run_episode()
            print('{} reward: {:.2f}, length: {}'.format(i,r,t))
            results.append([r,t])

            if i % 10 == 0:
                torch.save(self.mu, self.log_dir +'/models/model_'+str(i))
        np.save(self.log_dir + '/results_train.npy', np.array(results))

    def train1(self):
        results = []
        for i in range(self.num_episodes):
            r, t = self.run_episode()
            print('{} reward: {:.2f}, length: {}'.format(i,r,t))
            results.append([r,t])

            if i % 10 == 0:
                torch.save(self.mu, self.log_dir +'/models1/model_'+str(i))
        np.save(self.log_dir + '/results_train1.npy', np.array(results))

    def train2(self):
        results = []
        for i in range(self.num_episodes):
            r, t = self.run_episode()
            print('{} reward: {:.2f}, length: {}'.format(i,r,t))
            results.append([r,t])

            if i % 10 == 0:
                torch.save(self.mu, self.log_dir +'/models2/model_'+str(i))
        np.save(self.log_dir + '/results_train2.npy', np.array(results))

    def train3(self):
        results = []
        for i in range(self.num_episodes):
            r, t = self.run_episode()
            print('{} reward: {:.2f}, length: {}'.format(i,r,t))
            results.append([r,t])

            if i % 10 == 0:
                torch.save(self.mu, self.log_dir +'/models3/model_'+str(i))
        np.save(self.log_dir + '/results_train3.npy', np.array(results))

    def eval_all(self, model_dir, num_eps=5):
        results = []

        for model_fname in sorted(os.listdir(model_dir), key=lambda x: int(x.split('_')[1])):
            print(model_fname)
            mu = torch.load(os.path.join(model_dir, model_fname))
            r,t = self.eval(num_eps=num_eps, mu=mu)
            results.append([r,t])
        np.save(self.log_dir+'/results_eval.npy', np.array(results))

    def eval_all1(self, model_dir, num_eps=5):
        results = []

        for model_fname in sorted(os.listdir(model_dir), key=lambda x: int(x.split('_')[1])):
            print(model_fname)
            mu = torch.load(os.path.join(model_dir, model_fname))
            r,t = self.eval(num_eps=num_eps, mu=mu)
            results.append([r,t])
        np.save(self.log_dir+'/results_eval1.npy', np.array(results))

    def eval_all2(self, model_dir, num_eps=5):
        results = []

        for model_fname in sorted(os.listdir(model_dir), key=lambda x: int(x.split('_')[1])):
            print(model_fname)
            mu = torch.load(os.path.join(model_dir, model_fname))
            r,t = self.eval(num_eps=num_eps, mu=mu)
            results.append([r,t])
        np.save(self.log_dir+'/results_eval2.npy', np.array(results))

    def eval_all3(self, model_dir, num_eps=5):
        results = []

        for model_fname in sorted(os.listdir(model_dir), key=lambda x: int(x.split('_')[1])):
            print(model_fname)
            mu = torch.load(os.path.join(model_dir, model_fname))
            r,t = self.eval(num_eps=num_eps, mu=mu)
            results.append([r,t])
        np.save(self.log_dir+'/results_eval3.npy', np.array(results))

    def eval(self, num_eps=10, mu=None):
        if mu == None:
            mu = self.mu

        results = []
        mu = mu.eval()
        for i in range(num_eps):
            r, t = self.run_eval_episode(mu=mu)
            results.append([r,t])
            print('{} reward: {:.2f}, length: {}'.format(i,r,t))
        return np.mean(results, axis=0)

    def run_eval_episode(self, mu=None):
        if mu == None:
            mu = self.mu
        done = False
        s = torch.tensor(self.env.reset().astype(np.float32), requires_grad=False)
        tot_r = t = 0
        while not done:
            a = mu(s).view(-1).detach().numpy()

            a_updated = self.LQR(s, a)
            # s_p, r, done, _ = self.env.step(a)
            s_p, r, done, _ = self.env.step(a_updated)

            tot_r += r
            t += 1
            s = torch.tensor(s_p.astype(np.float32), requires_grad=False)
        return tot_r, t

    def LQR(self, s, a):

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

        if a < self.env.action_space2.low:
            a = self.env.action_space2.low
        elif a > self.env.action_space2.high:
            a = self.env.action_space2.high

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
        # print("u_sigma {}\n".format(u_sigma))

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

        if self.env.continuous:
            a_updated = np.array([thrust[0], thrust[1]])
            # print("a_updated {}\n".format(a_updated))
            # a = (0.5, 0)
            a_updated = np.clip(a_updated, -1, +1)  #  if the value is less than 0.5, it's ignored
            # print("a_updated * {}\n".format(a_updated))
        else:
            print("please change to cts mode")

        return a_updated

    def fill_buffer(self):
        print('Filling buffer')
        s = torch.tensor(self.env.reset().astype(np.float32), requires_grad=False)

        temp_number = 0

        while self.buffer.size < self.buffer_min:

            # self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
            a = np.random.uniform(self.env.action_space2.low, self.env.action_space2.high, size=(self.a_dim))
            a_updated = self.LQR(s, a)

            if temp_number < 3:
                print("a {}\n".format(a), "actions:","{} {}".format(a_updated[0], a_updated[1]))
                # print("a_updated*** {}\n".format(a_updated))
                temp_number += 1

            # s_p, r, done, _ = self.env.step(a)
            s_p, r, done, _ = self.env.step(a_updated)

            if done:
                self.env.reset()

            self.buffer.add_tuple(s,a,r,s_p,done)
            s = s_p
