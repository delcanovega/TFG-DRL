import gym

from rl_agent import QLAgent

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    agent = QLAgent(env.observation_space, env.action_space.n)

    average_reward = 0
    for i in range(100000):
        state = env.reset()
        #if i > 1000:
        #    env.render()

        epochs = 0
        done = False

        while not done:
            action = agent.predict_action(state)

            next_state, reward, done, info = env.step(action)
            agent.update_table(state, next_state, action, reward)

            state = next_state
            epochs += 1
            average_reward += reward

        if i % 100 == 0:
            print("Simulation {} ended with {} average score".format(i, round(average_reward / 100)))
            average_reward = 0
