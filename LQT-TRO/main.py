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

    k = 4000

    """
    agent = DDPG(env, log_dir, sigma=0.2, num_episodes=k, buffer_size=1000000, batch_size=64,
                 tau=1e-3, batch_norm=False, merge_layer=0)
    print('training start')
    agent.train()
    """
    agent = DDPG(env, log_dir, sigma=0.2, num_episodes=k, buffer_size=1000000, batch_size=64,
                 tau=1e-2, batch_norm=False, merge_layer=0)
    print('training1 start')
    agent.train1()
    """
    agent = DDPG(env, log_dir, sigma=0.1, num_episodes=k, buffer_size=1000000, batch_size=64,
                 tau=1e-3, batch_norm=False, merge_layer=0)
    print('training2 start')
    agent.train2()

    agent = DDPG(env, log_dir, sigma=0.1, num_episodes=k, buffer_size=1000000, batch_size=64,
                 tau=1e-2, batch_norm=False, merge_layer=0)
    print('training3 start')
    agent.train3()
    """

    # agent.eval_all(log_dir+'/models', num_eps=5)

if __name__ == '__main__':
    main()
