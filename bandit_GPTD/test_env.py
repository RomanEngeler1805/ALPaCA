import numpy as np
import matplotlib.pyplot as plt
from bandit_environment import bandit_environment

env = bandit_environment(5)

print(env.state)
print(env.mu)
print(env.L)
print(env.theta)
print(env.phase)
print(env._psi(env.state, env.phase))

print(env._step(2))