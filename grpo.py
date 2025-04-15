import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- GRPO Policy ---
class GRPOPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[64, 64]):
        """
        A simple feed-forward actor-only network.
        It outputs a mean action vector and uses a learnable log_std
        for the standard deviation.
        """
        super(GRPOPolicy, self).__init__()
        layers = []
        last_dim = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.ReLU())
            last_dim = hidden
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, state):
        mean = self.model(state)
        std = torch.exp(self.log_std)
        return mean, std

# --- Utility: Compute group advantages ---
def compute_group_advantages(trajectories, gamma=0.99):
    """
    Computes discounted returns for each trajectory.
    For each trajectory, it computes the discounted sum of rewards.
    Returns a flat numpy array containing all advantages.
    """
    advantages = []
    for traj in trajectories:
        G = 0
        traj_returns = []
        for (_, _, reward, _) in reversed(traj):
            G = reward + gamma * G
            traj_returns.insert(0, G)
        advantages.append(np.array(traj_returns))
    return np.concatenate(advantages)

# --- GRPO Agent ---
class GRPOAgent:
    def __init__(
        self, 
        env, 
        policy_kwargs=None, 
        lr=0.001, 
        gamma=0.99, 
        group_size=4,
        epsilon=0.15,
        beta=0.0005,
        device="cpu"
    ):
        """
        Initializes the GRPO agent.
        
        Args:
            env: The environment (e.g. PortfolioOptimizationEnv).
            policy_kwargs: Dictionary for policy network parameters.
            lr: Learning rate.
            gamma: Discount factor.
            group_size: Number of trajectories to group for advantage estimation.
            epsilon: Clipping parameter.
            beta: KL penalty coefficient.
            device: "cpu" or "cuda".
        """
        self.env = env
        self.gamma = gamma
        self.group_size = group_size
        self.epsilon = epsilon
        self.beta = beta
        # Compute flattened observation dimension
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = env.action_space.shape[0]
        if policy_kwargs is None:
            policy_kwargs = {}
        self.policy = GRPOPolicy(input_dim=obs_dim, output_dim=act_dim, **policy_kwargs).to(device)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr)
        self.device = device

    def select_action(self, state):
        """
        Given a state, samples an action from a Normal distribution based on the policy.
        The state is flattened before being passed to the network.
        Returns (action, log_prob).
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        # flatten the state tensor
        state_tensor = state_tensor.reshape(1, -1)
        mean, std = self.policy(state_tensor)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1)
        return action.squeeze(0).detach(), log_prob.squeeze(0).detach()

    def collect_trajectory(self, max_steps):
        """
        Collects one trajectory from the environment.
        Returns a list of tuples: (state, action, reward, log_prob).
        """
        state = self.env.reset()
        trajectory = []
        steps = 0
        done = False
        while not done and steps < max_steps:
            action, log_prob = self.select_action(state)
            # Convert action to numpy array for env.step
            next_state, reward, done, _ = self.env.step(action.cpu().numpy())
            trajectory.append((state, action, reward, log_prob))
            state = next_state
            steps += 1
        return trajectory

    def train(self, total_timesteps, max_steps_per_ep=None):
        """
        Trains the GRPO policy over total_timesteps.
        Trajectories are collected in groups and the policy is updated with one gradient step per group.
        """
        if max_steps_per_ep is None:
            max_steps_per_ep = getattr(self.env, '_max_episode_steps', 1000)
        trajectories = []
        timesteps = 0
        while timesteps < total_timesteps:
            traj = self.collect_trajectory(max_steps=max_steps_per_ep)
            trajectories.append(traj)
            timesteps += len(traj)
            if len(trajectories) >= self.group_size:
                self.gradient_ascent(trajectories)
                trajectories = []

    def gradient_ascent(self, trajectories):
        """
        Performs one policy update using all samples in the collected trajectories.
        Implements the GRPO-style update with clipping and a KL penalty term.
        """
        advantages = compute_group_advantages(trajectories, gamma=self.gamma)
        adv_tensor = torch.FloatTensor(advantages).to(self.device)
        mean_advantage = adv_tensor.mean()

        # Freeze current policy as old policy
        old_policy = GRPOPolicy(
            input_dim=int(np.prod(self.env.observation_space.shape)),
            output_dim=self.env.action_space.shape[0]
        ).to(self.device)
        old_policy.load_state_dict(self.policy.state_dict())

        loss_terms = []
        kl_terms = []

        for traj in trajectories:
            for (state, action, _, _) in traj:
                state_tensor = torch.FloatTensor(state).to(self.device)
                state_tensor = state_tensor.reshape(1, -1)
                with torch.no_grad():
                    mean_old, std_old = old_policy(state_tensor)
                    dist_old = torch.distributions.Normal(mean_old, std_old)
                    log_prob_old = dist_old.log_prob(action.unsqueeze(0)).sum(dim=1)
                    prob_old = torch.exp(log_prob_old)
                mean_new, std_new = self.policy(state_tensor)
                dist_new = torch.distributions.Normal(mean_new, std_new)
                log_prob_new = dist_new.log_prob(action.unsqueeze(0)).sum(dim=1)
                prob_new = torch.exp(log_prob_new)
                ratio = prob_new / (prob_old + 1e-8)
                unclipped = ratio * mean_advantage
                clipped = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * mean_advantage
                loss_sample = -torch.min(unclipped, clipped)
                loss_terms.append(loss_sample)
                ratio_kl = prob_old / (prob_new + 1e-8)
                kl = (ratio_kl - torch.log(ratio_kl) - 1).mean()
                kl_terms.append(kl)
        loss_policy = torch.stack(loss_terms).mean()
        kl_penalty = torch.stack(kl_terms).mean()
        total_loss = loss_policy + self.beta * kl_penalty
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        print(f"GRPO update: Loss = {total_loss.item():.4f}, Policy Loss = {loss_policy.item():.4f}, KL = {kl_penalty.item():.4f}")

    def predict(self, state):
        """
        For evaluation, returns the mean action from the policy.
        The input state is flattened before being passed to the network.
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        state_tensor = state_tensor.reshape(1, -1)
        mean, _ = self.policy(state_tensor)
        return mean.squeeze(0).detach().cpu().numpy()
