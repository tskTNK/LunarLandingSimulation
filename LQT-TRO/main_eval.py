import gym
from ddpg import DDPG

def main():
    env = gym.make('LunarLanderContinuous-v2')
    log_dir = 'log/lander'

    # env = gym.make('Pendulum-v0')
    # log_dir = 'log/pendulum'

    # paper settings
    # agent = DDPG(env, sigma=0.2, num_episodes=1000, buffer_size=1000000, batch_size=64,
    #              tau=1e-3, batch_norm=True, merge_layer=2)

    # did not work unless I merged action into critic at first layer
    # worked btter without batchnorm

    agent = DDPG(env, log_dir, sigma=0.2, num_episodes=20, buffer_size=1000000, batch_size=64,
                 tau=1e-3, batch_norm=False, merge_layer=0)
    # agent.train()
    # agent.eval_all(log_dir+'/models', num_eps=5)
    agent.eval_all1(log_dir+'/models1', num_eps=5)
    # agent.eval_all2(log_dir+'/models2', num_eps=5)
    # agent.eval_all3(log_dir+'/models3', num_eps=5)

if __name__ == '__main__':
    main()
