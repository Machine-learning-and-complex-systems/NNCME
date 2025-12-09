#BirthDeath-absorb-exact
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
plt.rc('font', size=14)

k1 = 0.1 #1.0
k2 = 0.01 #0.025
theta = 300
theta_values = np.linspace(0, theta+1, 50)
n_states = 40
target_state = 3
initialD='uniform'
initial_state = 1
uniform=[5,25]


Q = np.zeros((n_states, n_states))

for x in range(n_states):
    outgoing = 0.0

    if x < n_states - 1:
        Q[x, x+1] = k1
        outgoing += k1

    if x > 0:
        Q[x, x-1] = k2 * x
        outgoing += k2 * x

    Q[x, x] = -outgoing


Q_abs = Q.copy()
Q_abs[target_state, :] = 0.0
Q_abs[target_state, target_state] = 0.0

if initialD=='delta':

    p0 = np.zeros(n_states)
    p0[initial_state] = 1.0

if initialD=='uniform':

    p0 = np.zeros(n_states)
    a,b=uniform
    p0[a:b+1] = 1.0 / (b+1 - a)


trans_matrix = expm(Q_abs * theta)


p_theta = p0.dot(trans_matrix)
probability = p_theta[target_state]

print(f"T={theta}, prob(X={target_state}): {probability:.3e}")
plt.figure(figsize=(5,4))
plt.bar(range(n_states),p_theta)
plt.xlabel("X")
plt.ylabel("Probability")
plt.title(f"T={theta}")
plt.show()

prob_list = []

for theta in theta_values:

    trans_matrix = expm(Q_abs * theta)
    p_theta = p0.dot(trans_matrix)

    probability = p_theta[target_state]
    prob_list.append(probability)


plt.figure(figsize=(5,4))
plt.plot(theta_values, prob_list, marker=None,color='grey')
plt.xlabel("T")
plt.ylabel("Probability")
plt.title("X={} to X={}".format(initial_state,target_state))
plt.grid(True)
# plt.ylim(-0.05,0.65)
plt.show()
# %%
with open('data_path.txt', 'r') as file:
    data_file = file.read().strip()
absorb = np.load('{}/absorb.npy'.format(data_file), allow_pickle=True).item()


plt.figure(figsize=(5, 4))
plt.plot(theta_values, prob_list,label='Exact')
plt.plot(absorb['jiayu_time'], absorb['jiayu_prob'],label='NNCME')
plt.xlabel("Time")
plt.ylabel("absorb_prob")
plt.title("X={} to X={}".format(initial_state,target_state))
plt.legend()
plt.grid(True)
plt.show()

    
    
    
    
