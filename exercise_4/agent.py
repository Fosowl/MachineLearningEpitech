import numpy as np

alpha = 0.3
gamma = 0.9

class Agent(object):

    """Docstring for Agent."""

    def __init__(self, initial_position, known_rewards) -> None:
        self.position = initial_position
        self.known_rewards = known_rewards
        self.Q = np.random.rand(len(self.known_rewards), 3)

    def move(self, action, world_size) -> None:
        """
        the world is 1 dimensional
        """
        if action == "left":
            if self.position > 0:
                self.position -= 1
        elif action == "right":
            if self.position < world_size - 1:
                self.position += 1
        elif action == "none":
            pass
        else:
            raise ValueError("incorrect action")
        
    def update_Q(self, action, reward) -> None:
        """
        update the Q table
        """
        index_action = ["left", "right", "none"].index(action)
        reward = self.known_rewards[self.position]
        self.Q[self.position, index_action] = reward 
