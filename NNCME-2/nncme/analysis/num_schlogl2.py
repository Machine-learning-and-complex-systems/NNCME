import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

Para=1
c1 = 2.676
c2 = 0.040
c3 = 108.102
c4 = 37.881
na = 1
nb = 1 * Para
V = 1
theta = 0.2
n_states = 101
target_state = 1
initial_state = 50


Q = np.zeros((n_states, n_states))

for x in range(n_states):
    outgoing = 0.0
    

    if x >= 2:
        rate_R1 = (c1 * na * x * (x-1)) / V
        if x+1 < n_states:
            Q[x, x+1] += rate_R1
            outgoing += rate_R1
    

    if x >= 3:
        rate_R2 = (c2 * x * (x-1) * (x-2)) / (V**2)
        Q[x, x-1] += rate_R2
        outgoing += rate_R2
    

    rate_R3 = c3 * nb * V
    if x+1 < n_states:
        Q[x, x+1] += rate_R3
        outgoing += rate_R3
    

    rate_R4 = c4 * x
    if x >= 1:
        Q[x, x-1] += rate_R4
        outgoing += rate_R4
    

    Q[x, x] = -outgoing


Q_abs = Q.copy()
Q_abs[target_state, :] = 0.0
Q_abs[target_state, target_state] = 0.0


p0 = np.zeros(n_states)
p0[initial_state] = 1.0


trans_matrix = expm(Q_abs * theta)


probability = p0.dot(trans_matrix)[target_state]

print(f"Exact transition probability p({target_state}|{initial_state}, t ≤ {theta}): {probability:.3e}")


plt.figure(figsize=(10, 6))
plt.plot(range(n_states), p0.dot(trans_matrix), label="Probability Distribution", color="blue")
plt.axvline(x=target_state, color="red", linestyle="--", label=f"Target State (x={target_state})")
plt.axvline(x=initial_state, color="green", linestyle="--", label=f"Initial State (x={initial_state})")


plt.xlabel("State (x)")
plt.ylabel("Probability")
plt.title(f"Probability Distribution at t = {theta}")
plt.legend()
plt.grid(True)

plt.ylim(0,0.1)
plt.xlim(0, 80)
plt.show()
