import numpy as np
from agent import Agent
import random

epsilon = 0.2

def martin_antoine_policy(agent: Agent) -> str:
    """
    Policy de l'agent
    La policy de l'agent utilise une Q-table pour se souvenir des récompenses.
    Une valeur epsilon de 0.2 est utilisée pour déterminer si l'agent doit explorer ou exploiter.
    L'agent explore donc 20% du temps son environnement.
    La Q-table est déclarée dans le constructeur de l'agent, dans agent.py
    """
    actions = ["left", "right", "none"]
    action = "none"
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        action = actions[np.argmax(agent.Q[agent.position])]
        assert action in actions
        return action
