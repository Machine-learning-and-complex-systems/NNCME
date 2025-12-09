import time, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from multiprocessing import Pool, cpu_count, freeze_support
from tqdm import tqdm
plt.rc('font', size=14)
def adaptive_ylim(y_max):
    """Adaptive ylim operation.
    """


    if y_max >= 0.8: return 1.0
    elif y_max >= 0.4: return 0.8
    elif y_max >= 0.1: return 0.4
    else: return 0.1


Sites = 2
tfinal = 1
times = 10000
nx = 50
nproc = 4
c1, c2, c3, c4 = 2.676, 0.040, 108.102, 37.881
d = 8.2207
num_points = 100
uniform_time_points = np.linspace(0, tfinal, num_points)

absorb_state = [0, 15]
absorb_min_val, absorb_max_val = absorb_state
absorb_min = np.full(Sites, absorb_min_val)
absorb_max = np.full(Sites, absorb_max_val)

modify = 1
if modify==True:
    filename = f"SSA_absorb/SSA_modify_absorb{absorb_state}_Sites{Sites}_times{times}_T{tfinal}.npz"
else:
    filename = f"SSA_absorb/SSA_absorb{absorb_state}_Sites{Sites}_times{times}_T{tfinal}.npz"


L, K = 1, 4
L_total, K_total = L * Sites, K * Sites
Left = np.array([2, 3, 0, 1])
Right = np.array([3, 2, 1, 0])
update1L = np.zeros((L_total, K_total), dtype=int)
update1R = np.zeros((L_total, K_total), dtype=int)
for i in range(Sites):
    update1L[i*L:(i+1)*L, i*K:(i+1)*K] = Left
    update1R[i*L:(i+1)*L, i*K:(i+1)*K] = Right
update1 = update1R - update1L

if Sites > 1:
    update2L = np.zeros((L_total, 1+(Sites-2)*2+1), dtype=int)
    update2R = np.zeros((L_total, 1+(Sites-2)*2+1), dtype=int)
    update2L[0,0] = 1; update2L[-L,-1] = 1
    update2R[L,0] = 1; update2R[-L*2,-1] = 1
    for i in range(1, Sites-1):
        update2L[i*L,i*2-1] = 1
        update2L[i*L,i*2] = 1
        update2R[(i-1)*L,i*2-1] = 1
        update2R[(i+1)*L,i*2] = 1
    update2 = update2R - update2L
    update = np.concatenate((update1, update2), axis=1)
else:
    update = update1
V = update.T



def single_sim(_):
    """Single sim operation.

    

    Args:

        _: Parameter forwarded to single_sim.

    

    Returns:

        Result produced by single_sim.

    """


    X = np.full(Sites, nx)
    t = 0
    traj = [[t] + list(X)]
    while t < tfinal:
        prop = np.zeros(K * Sites + 1 + (Sites - 2) * 2 + 1)
        if modify==True:
            if (X==absorb_max).any():#(X<=absorb_max).all():
                for i in range(Sites):
                    k, ii = i * K, i * L
                    prop[k]     = 0 #c1 * X[ii] * (X[ii]-1)# X[ii] ** 2
                    prop[k + 1] = c2 * X[ii] * (X[ii]-1) * (X[ii]-2) #X[ii] ** 3
                    prop[k + 2] = 0 #c3
                    prop[k + 3] = c4 * X[ii]
            else:
                for i in range(Sites):
                    k, ii = i * K, i * L
                    prop[k]     = c1 * X[ii] * (X[ii]-1)# X[ii] ** 2
                    prop[k + 1] = c2 * X[ii] * (X[ii]-1) * (X[ii]-2) #X[ii] ** 3
                    prop[k + 2] = c3
                    prop[k + 3] = c4 * X[ii]
                if Sites>1:
                    prop[K_total] = d * X[0] 
                    prop[-1] = d * X[-1]
                    for i in range(1, Sites - 1):
                        prop[K_total + i*2 - 1] = d * X[i*L]
                        prop[K_total + i*2]     = d * X[i*L]
        else:
            if 1-((X<=absorb_max).all()):
                for i in range(Sites):
                    k, ii = i * K, i * L
                    prop[k]     = c1 * X[ii] * (X[ii]-1)# X[ii] ** 2
                    prop[k + 1] = c2 * X[ii] * (X[ii]-1) * (X[ii]-2) #X[ii] ** 3
                    prop[k + 2] = c3
                    prop[k + 3] = c4 * X[ii]
                if Sites>1:
                    prop[K_total] = d * X[0] 
                    prop[-1] = d * X[-1]
                    for i in range(1, Sites - 1):
                        prop[K_total + i*2 - 1] = d * X[i*L]
                        prop[K_total + i*2]     = d * X[i*L]

        asum = np.sum(prop)
        if asum == 0: break
        tau = np.log(1.0 / np.random.rand()) / asum
        r = np.random.rand() * asum
        j = np.searchsorted(np.cumsum(prop), r)
        X0 = X.copy()
        X = X + V[j]
        t = t + tau
        # if modify==True:
        #     if (X0 <= absorb_max).all() and (X > X0).any() and (X > absorb_max).any():
        #         break
        # else:
        #     if (X0 <= absorb_max).all():
        #         break
        if (X < 0).any(): break
        traj.append([t] + list(X))

    traj = np.array(traj)
    interp_result = np.zeros((num_points, Sites))
    t_vals = traj[:, 0]
    for s in range(Sites):
        x_vals = traj[:, s + 1]
        j = 0
        for i, t_target in enumerate(uniform_time_points):
            while j + 1 < len(t_vals) and t_vals[j + 1] <= t_target:
                j += 1
            interp_result[i, s] = x_vals[j]
    return interp_result

# %%
interp_result=single_sim(times)
# %%

def main():
    """Main operation.

    

    Returns:

        Result produced by main.

    """


    if os.path.exists(filename):
        data = np.load(filename)
        time_points = data['time']
        SampleSum = data['results']
        absorb_prob = data['absorb_prob']
        print('file exists: ', filename)
    else:
        with Pool(processes=nproc) as pool:
            results_list = list(tqdm(pool.imap(single_sim, range(times)), total=times))
        SampleSum = np.stack(results_list)
        time_points = uniform_time_points

        absorb_min = np.full(Sites, absorb_min_val)
        absorb_max = np.full(Sites, absorb_max_val)
        within_absorb = ((SampleSum >= absorb_min) & (SampleSum <= absorb_max)).all(axis=2)
        absorb_prob = within_absorb.mean(axis=0)

        np.savez(filename, time=time_points, results=SampleSum, absorb_prob=absorb_prob)
        print('Saved:', filename)

    return SampleSum, absorb_prob, time_points

    

if __name__ == '__main__':
    freeze_support()
    SampleSum, absorb_prob, time_points = main()
    

    t = 1
    index = np.argmin(np.abs(time_points - t))
    numplot = Sites + 1
    L_label = [f'X{i}' for i in range(Sites)]
    colors = ['C0','C1','C2','C3']
    fig, axes = plt.subplots(1, numplot)
    fig.tight_layout()

    for i in range(Sites):
        ax = plt.subplot(1, numplot, i+1)
        arr = SampleSum[:, index, i]
        lim = 85
        hist_vals, _, _ = plt.hist(arr, density=True, color=colors[i], bins=lim, range=(0, lim), label=L_label[i])
        plt.ylabel('Prob')
        plt.xlabel('Count')
        plt.title(f't={t}')
        ymax = max(hist_vals) if len(hist_vals) > 0 else 0
        plt.ylim(0, adaptive_ylim(ymax))
        plt.xticks(np.arange(0,81,15))
        plt.legend()

    if Sites > 1:
        ax = plt.subplot(1, numplot, numplot)
        arr1 = SampleSum[:, index, 0]
        arr2 = SampleSum[:, index, 1]
        plt.hist2d(arr1, arr2, bins=[40, 40], density=True,
                   norm=mpl.colors.LogNorm(vmin=1e-4, vmax=1e-1))
        plt.xlabel('X0'); plt.ylabel('X1')
        plt.title(f'Joint Dist t={t}')
        ax.set_facecolor([68/255, 1/255, 80/255])
        plt.xlim(0, lim); plt.ylim(0, lim)
        plt.xticks(np.arange(0,81,15))

    fig.set_size_inches(numplot * 5, 5)
    plt.tight_layout()
    plt.show()


    absorb_min_val = 0
    absorb_max_val = 15
    absorb_min = np.full(Sites, absorb_min_val)
    absorb_max = np.full(Sites, absorb_max_val)


    within_absorb = ((SampleSum >= absorb_min) & (SampleSum <= absorb_max)).all(axis=2)


    absorb_prob = within_absorb.mean(axis=0)


    plt.figure(figsize=(5,4))
    plt.plot(time_points, absorb_prob, color='grey', lw=2)
    plt.xlabel("Time")
    plt.ylabel("Absorption Probability")
    plt.title("Probability inside absorption region")
    plt.grid()
    plt.ylim([0,1])
    plt.tight_layout()
    plt.show()
