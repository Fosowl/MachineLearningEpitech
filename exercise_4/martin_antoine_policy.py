import numpy as np
from agent import Agent
import random

epsilon = 0.2

def martin_antoine_policy(agent: Agent) -> str:
    """
    Policy of the agent
    return "left", "right", or "none"
    """
    actions = ["left", "right", "none"]
    action = "none"
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        action = actions[np.argmax(agent.Q[agent.position])]
        assert action in actions
        return action
