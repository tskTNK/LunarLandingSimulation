import gym
import torch
import time

def demo_ddpg_lander(env, seed=None, render=False):
    env.seed(seed)
    total_reward = 0
    steps = 0
    s = env.reset()
    mu = torch.load('log/lander/models3/model_7980')
    mu = mu.eval()
    while True:

        s = torch.tensor(s)
        a = torch.squeeze(mu(s)).data.numpy()
        s, r, done, info = env.step(a)
        total_reward += r

        if render:
            still_open = env.render()
            if still_open == False: break

        if steps % 1 == 0 or done:
        # if done:
            # print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            # print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            print("step {} reward {:+0.2f}".format(steps, total_reward),"observations:", " ".join(["{:+0.2f}".format(x) for x in s]), "actions:","{} {}".format(a[0], a[1]))
        steps += 1
        if done: break
    return total_reward


if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    time.sleep(3)

    for i in range(1):
        demo_ddpg_lander(env, seed=i, render=True)
