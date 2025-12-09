# Schlogl-absorb-Exact-Compare
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
theta = 1
n_states = 101
target_state = 1
initial_state = 50
theta_values = np.linspace(0, theta, 100)


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

plt.ylim(0,0.05)
plt.xlim(0, 80)
plt.show()

prob_list = []

for theta in theta_values:
    p0 = np.zeros(n_states)
    p0[initial_state] = 1.0
    trans_matrix = expm(Q_abs * theta)
    p_theta = p0.dot(trans_matrix)

    probability = p_theta[target_state]
    prob_list.append(probability)


plt.figure(figsize=(6,4))
plt.plot(theta_values, prob_list, marker='o',color='grey')
plt.xlabel("T")
plt.ylabel("Probability")
plt.title("X={} to X={}".format(initial_state,target_state))
plt.grid(True)
# plt.ylim(-0.05,0.4)
plt.show()

# %%
# absorb4 = np.load("E:\\Downloads\\absorb (10).npy", allow_pickle=True).item()
# jiayu_time4 = absorb4["jiayu_time"]
# jiayu_prob4 = absorb4["jiayu_prob"]
Sites=4
absorb4 = np.load("SSA_absorb.npy", allow_pickle=True).item()
absorb4 = np.load(f'E:\\Downloads\\SSA_absorb_Sites{Sites}_times10000_T1.0_prob.npy', allow_pickle=True).item()
absorb4 = np.load('E:\\Downloads\\SSA_absorb_Sites4_times50000_T3.0_prob.npy', allow_pickle=True).item()
# traj = np.load('E:\\Downloads\\SSA_absorb_Sites1_times100_T1.0_traj.npy', allow_pickle=True)
# print(traj.shape)
# (100, 100, 1)
# plt.hist(traj[:,-1,0])
SSA_time = absorb4["SSA_time"]
SSA_prob = absorb4["SSA_prob"]

plt.rc('font', size=14)
plt.figure(figsize=(6,4))
# plt.plot(theta_values, prob_list,color='grey',label='exact')
# plt.plot(jiayu_time4, jiayu_prob4, label='NG-random-lr0.1')
plt.plot(SSA_time, SSA_prob, label='SSA')
plt.xlabel("T")
plt.ylabel("Probability")
plt.title(f"Schlogl-Sites{Sites}  X={initial_state} to X={target_state}")
plt.grid(True)
plt.legend()
# plt.xlim(-0.005,0.4)
# plt.ylim(-0.001,0.01)
# ax_inset = plt.gca().inset_axes([0.75, 0.1, 0.2, 0.2])  # [left, bottom, width, height]
# # ax_inset.set_yscale("log")
# ax_inset.plot(theta_values, prob_list,color='grey')
# ax_inset.set_xlim(-0.005,0.2)
# ax_inset.set_ylim(-0.001,0.01)
# ax_inset.set_xticks([0,0.2])
# ax_inset.tick_params(axis='both', labelsize=10)
# # ax_inset.grid(True, linestyle="--")
plt.show()


# %%
absorb1 = np.load("E:\\Downloads\\absorb (7).npy", allow_pickle=True).item()
jiayu_time1 = absorb1["jiayu_time"]
jiayu_prob1 = absorb1["jiayu_prob"]

absorb2 = np.load("E:\\Downloads\\absorb (8).npy", allow_pickle=True).item()
jiayu_time2 = absorb2["jiayu_time"]
jiayu_prob2 = absorb2["jiayu_prob"]
    
absorb3 = np.load("E:\\Downloads\\absorb (9).npy", allow_pickle=True).item()
jiayu_time3 = absorb3["jiayu_time"]
jiayu_prob3 = absorb3["jiayu_prob"]

absorb4 = np.load("E:\\Downloads\\absorb (12).npy", allow_pickle=True).item()
jiayu_time4 = absorb4["jiayu_time"]
jiayu_prob4 = absorb4["jiayu_prob"]


plt.rc('font', size=14)
plt.figure(figsize=(6,4))
plt.plot(theta_values, prob_list,color='grey',label='exact')
plt.plot(jiayu_time1, jiayu_prob1, label='NG-default')
plt.plot(jiayu_time2, jiayu_prob2, label='NG-random')
plt.plot(jiayu_time3, jiayu_prob3, label='SGD-random')
plt.plot(jiayu_time4, jiayu_prob4, label='NG-default-new')
plt.xlabel("T")
plt.ylabel("Probability")
plt.title("Schlogl-Sites1  X={} to X={}".format(initial_state,target_state))
plt.grid(True)
plt.legend()
# plt.xlim(-0.005,0.4)
# plt.ylim(-0.001,0.05)
ax_inset = plt.gca().inset_axes([0.75, 0.1, 0.2, 0.2])  # [left, bottom, width, height]
# ax_inset.set_yscale("log")
ax_inset.plot(theta_values, prob_list,color='grey')
ax_inset.plot(jiayu_time1, jiayu_prob1)
ax_inset.plot(jiayu_time2, jiayu_prob2)
ax_inset.plot(jiayu_time3, jiayu_prob3)
ax_inset.plot(jiayu_time4, jiayu_prob4)
ax_inset.set_xlim(-0.005,0.2)
ax_inset.set_ylim(-0.001,0.01)
ax_inset.set_xticks([0,0.2])
ax_inset.tick_params(axis='both', labelsize=10)
# ax_inset.grid(True, linestyle="--")
plt.show()


















    
