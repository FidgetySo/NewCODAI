import sys
import random
import numpy as np

from tqdm import tqdm
from agent import Agent


from collections import deque

class DDQN:
    """ Deep Q-Learning Main Algorithm
    """

    def __init__(self, action_dim, state_dim):
        """ Initialization
        """
        # Environment and DDQN parameters
        self.with_per = False
        self.action_dim = action_dim
        self.state_dim = (4,) + state_dim
        #
        self.lr = 2.5e-4
        self.gamma = 0.95
        self.epsilon = 0.8
        self.epsilon_decay = 0.99
        self.target_update_counter = 1
        #
        if(len(state_dim) < 3):
            self.tau = 1e-2
        else:
            self.tau = 1.0
        # Create actor and critic networks
        self.agent = Agent(self.state_dim, action_dim, self.lr, self.tau, True)
        # Memory Buffer for Experience Replay
        self.replay_memory=deque(maxlen=10_000)

    def policy_action(self, s):
        """ Apply an espilon-greedy policy to pick next action
        """
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        else:
            return np.argmax(self.agent.predict(s)[0])
    def train(self, old_state, state, reward, done):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory)< 1_000:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, 64)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.agent.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.agent.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + 0.99 * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.agent.fit(X, y)

        # Update target network counter every episode
        if done:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter >= 5:
            MODEL_FILE='models/MODEL.h5'
            self.agent.load_weights(MODEL_FILE)
            self.target_update_counter = 0

    def memorize(self, transition):
        """ Store experience in memory buffer
        """

        td_error = 0
        self.replay_memory.append(transition)

    def save_weights(self, path):
        self.agent.save(path)

    def load_weights(self, path):
        self.agent.load_weights(path)