import numpy as np

class TestAgentModel:
    """
    A simple test model for portfolio optimization.
    This agent does not learn and simply takes random actions.
    Implements a minimal interface:
      - train(episodes): Runs episodes taking random actions
      - predict(state): Returns a random action
      - test(env, ...): (optional) Runs a full rollout in the given environment
    """
    def __init__(self, env, device="cpu", **kwargs):
        self.env = env
        self.device = device
        self.episodes_trained = 0

    def train(self, episodes=10):
        print("Starting training for TestAgentModel (random actions)...")
        for ep in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = self.env.action_space.sample()  # random action
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
            print(f"Episode {ep + 1}: Total Reward = {total_reward:.2f}")
            self.episodes_trained += 1

    def predict(self, state):
        # Simply return a random action from the environment's action space.
        return self.env.action_space.sample()

    def test(self, env, policy=None, online_training_period=10, lr=None, optimizer=None):
        print("Testing TestAgentModel (random actions)...")
        state = env.reset()
        done = False
        while not done:
            action = self.predict(state)
            state, reward, done, _ = env.step(action)
        print("Test episode complete.")
