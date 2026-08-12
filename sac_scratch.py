import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Normal


# ==========================================================
# SAC PARAMETERS
# ==========================================================

STATE_SIZE = 23
ACTION_SIZE = 4

GAMMA = 0.99

TAU = 0.005

ALPHA = 0.2

LEARNING_RATE = 3e-4

BUFFER_SIZE = 100000


# ==========================================================
# REPLAY BUFFER
# ==========================================================

class ReplayBuffer:

    def __init__(self, max_size):

        self.max_size = max_size

        self.buffer = []


    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        experience = (
            state,
            action,
            reward,
            next_state,
            done
        )


        # Simple implementation:
        # remove the oldest experience
        # when the buffer is full

        if len(self.buffer) >= self.max_size:

            self.buffer.pop(0)


        self.buffer.append(
            experience
        )


    def sample(self, batch_size):

        batch = random.sample(
            self.buffer,
            batch_size
        )


        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []


        for experience in batch:

            state, action, reward, next_state, done = experience

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)


        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )


    def size(self):

        return len(self.buffer)


# ==========================================================
# SAC ACTOR
# ==========================================================

class Actor(nn.Module):

    def __init__(self):

        super().__init__()

        self.layer1 = nn.Linear(
            STATE_SIZE,
            64
        )

        self.layer2 = nn.Linear(
            64,
            64
        )


        # Gaussian policy mean
        self.mean_layer = nn.Linear(
            64,
            ACTION_SIZE
        )


        # Gaussian policy standard deviation
        self.log_std_layer = nn.Linear(
            64,
            ACTION_SIZE
        )


    def forward(self, state):

        x = torch.relu(
            self.layer1(state)
        )

        x = torch.relu(
            self.layer2(x)
        )


        mean = self.mean_layer(x)

        log_std = self.log_std_layer(x)


        # Avoid too small or too large std
        log_std = torch.clamp(
            log_std,
            -20,
            2
        )


        std = torch.exp(
            log_std
        )


        return mean, std


    def sample_action(self, state):

        mean, std = self.forward(
            state
        )


        distribution = Normal(
            mean,
            std
        )


        # Reparameterization trick:
        #
        # z = mean + std * epsilon
        #
        # rsample() allows the gradient
        # to go through the sampled action.

        z = distribution.rsample()


        # Keep actions inside [-1, 1]
        action = torch.tanh(z)


        # log pi(a | s)
        log_prob = distribution.log_prob(z)


        # Correction because of tanh
        log_prob = log_prob - torch.log(
            1 - action.pow(2) + 1e-6
        )


        log_prob = log_prob.sum(
            dim=-1,
            keepdim=True
        )


        return action, log_prob


# ==========================================================
# Q FUNCTION
# ==========================================================

class Critic(nn.Module):

    def __init__(self):

        super().__init__()


        # Q receives both state and action
        input_size = (
            STATE_SIZE
            + ACTION_SIZE
        )


        self.layer1 = nn.Linear(
            input_size,
            64
        )

        self.layer2 = nn.Linear(
            64,
            64
        )

        self.output_layer = nn.Linear(
            64,
            1
        )


    def forward(
        self,
        state,
        action
    ):

        # Concatenate state and action:
        #
        # Q(s, a)

        x = torch.cat(
            [state, action],
            dim=-1
        )


        x = torch.relu(
            self.layer1(x)
        )

        x = torch.relu(
            self.layer2(x)
        )


        q_value = self.output_layer(
            x
        )


        return q_value


# ==========================================================
# SAC AGENT
# ==========================================================

class SACAgent:

    def __init__(self):

        self.actor = Actor()


        # SAC uses two Q networks
        self.critic1 = Critic()

        self.critic2 = Critic()


        # Target networks
        self.target_critic1 = Critic()

        self.target_critic2 = Critic()


        # Initially target and normal
        # networks have the same parameters

        self.target_critic1.load_state_dict(
            self.critic1.state_dict()
        )


        self.target_critic2.load_state_dict(
            self.critic2.state_dict()
        )


        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=LEARNING_RATE
        )


        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(),
            lr=LEARNING_RATE
        )


        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(),
            lr=LEARNING_RATE
        )


        self.replay_buffer = ReplayBuffer(
            BUFFER_SIZE
        )


    # ======================================================
    # SELECT ACTION
    # ======================================================

    def choose_action(
        self,
        state
    ):

        state = torch.tensor(
            state,
            dtype=torch.float32
        )

        state = state.unsqueeze(0)


        with torch.no_grad():

            action, _ = self.actor.sample_action(
                state
            )


        action = action.squeeze(0)


        return action.numpy()


    # ======================================================
    # UPDATE NETWORKS
    # ======================================================

    def update(
        self,
        batch_size=256
    ):

        # Cannot train until enough
        # samples exist

        if self.replay_buffer.size() < batch_size:

            return None


        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.replay_buffer.sample(
            batch_size
        )


        states = torch.tensor(
            states,
            dtype=torch.float32
        )


        actions = torch.tensor(
            actions,
            dtype=torch.float32
        )


        rewards = torch.tensor(
            rewards,
            dtype=torch.float32
        ).unsqueeze(1)


        next_states = torch.tensor(
            next_states,
            dtype=torch.float32
        )


        dones = torch.tensor(
            dones,
            dtype=torch.float32
        ).unsqueeze(1)


        # ==================================================
        # CALCULATE BELLMAN TARGET
        # ==================================================

        with torch.no_grad():

            next_actions, next_log_probs = (
                self.actor.sample_action(
                    next_states
                )
            )


            next_q1 = self.target_critic1(
                next_states,
                next_actions
            )


            next_q2 = self.target_critic2(
                next_states,
                next_actions
            )


            # SAC uses the smaller Q value
            #
            # min(Q1, Q2)

            min_next_q = torch.min(
                next_q1,
                next_q2
            )


            # Entropy regularized value:
            #
            # Q - alpha * log(pi)

            next_value = (
                min_next_q
                - ALPHA
                * next_log_probs
            )


            # Bellman target:
            #
            # y =
            # r
            # +
            # gamma
            # * (1-done)
            # * [
            #     min(Q1', Q2')
            #     - alpha log pi
            #   ]

            target_q = (
                rewards
                + GAMMA
                * (1 - dones)
                * next_value
            )


        # ==================================================
        # UPDATE CRITIC 1
        # ==================================================

        current_q1 = self.critic1(
            states,
            actions
        )


        critic1_loss = (
            (current_q1 - target_q) ** 2
        ).mean()


        self.critic1_optimizer.zero_grad()

        critic1_loss.backward()

        self.critic1_optimizer.step()


        # ==================================================
        # UPDATE CRITIC 2
        # ==================================================

        current_q2 = self.critic2(
            states,
            actions
        )


        critic2_loss = (
            (current_q2 - target_q) ** 2
        ).mean()


        self.critic2_optimizer.zero_grad()

        critic2_loss.backward()

        self.critic2_optimizer.step()


        # ==================================================
        # UPDATE ACTOR
        # ==================================================

        new_actions, log_probs = (
            self.actor.sample_action(
                states
            )
        )


        q1 = self.critic1(
            states,
            new_actions
        )


        q2 = self.critic2(
            states,
            new_actions
        )


        min_q = torch.min(
            q1,
            q2
        )


        # SAC policy objective:
        #
        # L_actor =
        #
        # alpha log(pi(a|s))
        # -
        # Q(s,a)

        actor_loss = (
            ALPHA
            * log_probs
            - min_q
        ).mean()


        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        self.actor_optimizer.step()


        # ==================================================
        # UPDATE TARGET NETWORKS
        # ==================================================

        self.soft_update(
            self.critic1,
            self.target_critic1
        )


        self.soft_update(
            self.critic2,
            self.target_critic2
        )


        return (
            actor_loss.item(),
            critic1_loss.item(),
            critic2_loss.item()
        )


    # ======================================================
    # SOFT TARGET UPDATE
    # ======================================================

    def soft_update(
        self,
        critic,
        target_critic
    ):

        for target_param, param in zip(
            target_critic.parameters(),
            critic.parameters()
        ):

            # Soft update:
            #
            # theta_target =
            #
            # tau * theta
            #
            # +
            #
            # (1 - tau)
            # * theta_target

            target_param.data.copy_(

                TAU
                * param.data

                +

                (1 - TAU)
                * target_param.data
            )


# ==========================================================
# TEST MODEL CREATION ONLY
# ==========================================================

if __name__ == "__main__":

    agent = SACAgent()

    print("SAC agent created.")

    print("\nActor:")
    print(agent.actor)

    print("\nCritic 1:")
    print(agent.critic1)

    print("\nCritic 2:")
    print(agent.critic2)

    print("\nReplay buffer size:")
    print(agent.replay_buffer.size())

    print("\nTraining is not started yet.")