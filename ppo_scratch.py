import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Normal


# ==========================================================
# PPO PARAMETERS
# ==========================================================

STATE_SIZE = 23
ACTION_SIZE = 4

GAMMA = 0.99
GAE_LAMBDA = 0.95

CLIP_EPSILON = 0.2

ACTOR_LR = 3e-4
CRITIC_LR = 1e-3


# ==========================================================
# ACTOR NETWORK
# ==========================================================

class Actor(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(STATE_SIZE, 64)
        self.layer2 = nn.Linear(64, 64)

        # Mean of the Gaussian policy
        self.mean_layer = nn.Linear(64, ACTION_SIZE)

        # For now I use one trainable standard deviation
        # value for each action dimension
        self.log_std = nn.Parameter(
            torch.zeros(ACTION_SIZE)
        )


    def forward(self, state):

        x = torch.tanh(self.layer1(state))
        x = torch.tanh(self.layer2(x))

        mean = self.mean_layer(x)

        std = torch.exp(self.log_std)

        return mean, std


    def sample_action(self, state):

        mean, std = self.forward(state)

        # Gaussian policy:
        #
        # a ~ pi(a | s)

        distribution = Normal(mean, std)

        raw_action = distribution.rsample()

        # Action should stay between -1 and 1
        action = torch.tanh(raw_action)

        # First calculate Gaussian log probability
        log_prob = distribution.log_prob(raw_action)

        # Because tanh changes the distribution,
        # this correction term is needed
        log_prob = log_prob - torch.log(
            1 - action.pow(2) + 1e-6
        )

        log_prob = log_prob.sum(dim=-1)

        return action, log_prob


    def evaluate_action(self, state, action):

        mean, std = self.forward(state)

        distribution = Normal(mean, std)

        # We stored the action after tanh.
        # To calculate the probability again,
        # approximately reverse tanh.

        action = torch.clamp(
            action,
            -0.999999,
            0.999999
        )

        raw_action = torch.atanh(action)

        log_prob = distribution.log_prob(raw_action)

        log_prob = log_prob - torch.log(
            1 - action.pow(2) + 1e-6
        )

        log_prob = log_prob.sum(dim=-1)

        return log_prob


# ==========================================================
# CRITIC NETWORK
# ==========================================================

class Critic(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(STATE_SIZE, 64)
        self.layer2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 1)


    def forward(self, state):

        x = torch.tanh(self.layer1(state))
        x = torch.tanh(self.layer2(x))

        value = self.output_layer(x)

        return value


# ==========================================================
# PPO AGENT
# ==========================================================

class PPOAgent:

    def __init__(self):

        self.actor = Actor()
        self.critic = Critic()

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=ACTOR_LR
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=CRITIC_LR
        )


    def choose_action(self, state):

        state = torch.tensor(
            state,
            dtype=torch.float32
        )

        with torch.no_grad():

            action, log_prob = self.actor.sample_action(state)

            value = self.critic(state)


        return (
            action.numpy(),
            log_prob.item(),
            value.item()
        )


    # ======================================================
    # GENERALIZED ADVANTAGE ESTIMATION
    # ======================================================

    def calculate_advantages(
        self,
        rewards,
        values,
        dones,
        last_value=0.0
    ):

        advantages = []

        advantage = 0.0
        next_value = last_value


        # GAE is calculated backwards
        for t in reversed(range(len(rewards))):

            # Temporal Difference error:
            #
            # delta_t =
            # r_t
            # + gamma * V(s_t+1)
            # - V(s_t)

            delta = (
                rewards[t]
                + GAMMA
                * next_value
                * (1 - dones[t])
                - values[t]
            )


            # Generalized Advantage Estimation:
            #
            # A_t =
            # delta_t
            # + gamma * lambda * A_t+1

            advantage = (
                delta
                + GAMMA
                * GAE_LAMBDA
                * (1 - dones[t])
                * advantage
            )


            advantages.insert(
                0,
                advantage
            )


            next_value = values[t]


        advantages = torch.tensor(
            advantages,
            dtype=torch.float32
        )


        values_tensor = torch.tensor(
            values,
            dtype=torch.float32
        )


        # Return:
        #
        # R_t = A_t + V(s_t)

        returns = (
            advantages
            + values_tensor
        )


        return advantages, returns


    # ======================================================
    # PPO UPDATE
    # ======================================================

    def update(
        self,
        states,
        actions,
        old_log_probs,
        advantages,
        returns
    ):

        states = torch.tensor(
            states,
            dtype=torch.float32
        )

        actions = torch.tensor(
            actions,
            dtype=torch.float32
        )

        old_log_probs = torch.tensor(
            old_log_probs,
            dtype=torch.float32
        )


        # Advantage normalization
        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std() + 1e-8
        )


        # --------------------------------------------------
        # ACTOR UPDATE
        # --------------------------------------------------

        new_log_probs = self.actor.evaluate_action(
            states,
            actions
        )


        # Probability ratio:
        #
        # r_t(theta) =
        # pi_new(a_t | s_t)
        # -----------------
        # pi_old(a_t | s_t)
        #
        # With log probabilities:
        #
        # exp(log pi_new - log pi_old)

        ratio = torch.exp(
            new_log_probs
            - old_log_probs
        )


        # Normal policy objective
        objective1 = (
            ratio
            * advantages
        )


        # PPO clipping:
        #
        # clip(
        #   r_t,
        #   1 - epsilon,
        #   1 + epsilon
        # )

        clipped_ratio = torch.clamp(
            ratio,
            1 - CLIP_EPSILON,
            1 + CLIP_EPSILON
        )


        objective2 = (
            clipped_ratio
            * advantages
        )


        # PPO clipped surrogate loss:
        #
        # L =
        # - min(
        #     r_t A_t,
        #     clip(r_t) A_t
        #   )

        actor_loss = -torch.min(
            objective1,
            objective2
        ).mean()


        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        self.actor_optimizer.step()


        # --------------------------------------------------
        # CRITIC UPDATE
        # --------------------------------------------------

        predicted_values = self.critic(
            states
        ).squeeze(-1)


        # Value function loss:
        #
        # L_v =
        # (R_t - V(s_t))^2

        critic_loss = (
            (returns - predicted_values) ** 2
        ).mean()


        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        self.critic_optimizer.step()


        return (
            actor_loss.item(),
            critic_loss.item()
        )


# ==========================================================
# TEST MODEL CREATION ONLY
# ==========================================================

if __name__ == "__main__":

    agent = PPOAgent()

    print("PPO agent created.")

    print("\nActor:")
    print(agent.actor)

    print("\nCritic:")
    print(agent.critic)

    print("\nTraining is not started yet.")