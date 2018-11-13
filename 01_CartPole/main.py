import gym

from rl_agent import QLAgent

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    agent = QLAgent(env.observation_space, env.action_space.n)

    acc_reward = 0
    for i in range(10000):
        state = env.reset()

        done = False

        while not done:
            if i > 1500:
                env.render()

            action = agent.predict_action(state)

            next_state, reward, done, info = env.step(action)
            agent.update_table(state, next_state, action, reward)

            state = next_state
            acc_reward += reward

        agent.decrease_exploration()

        if i % 100 == 0 and i > 0:
            print("Simulations {}-{} ended with {} average score".format(i - 100, i, round(acc_reward / 100)))
            acc_reward = 0
