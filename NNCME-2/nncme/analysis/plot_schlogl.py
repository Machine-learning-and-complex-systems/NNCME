import numpy as np
from nncme.args import args
import matplotlib.pyplot as plt
from matplotlib import cm
import math

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl
from matplotlib import cm
jet = cm.get_cmap('jet')
Number=15
jet_12_colors = jet(np.linspace(0, 1, Number))
color1=jet_12_colors[0,:]
color2=jet_12_colors[4,:]
color3=jet_12_colors[12,:]
color4=jet_12_colors[8,:]
color=[color1,color2,color3,color4]
jet= jet(np.linspace(0, 1, 16))


with open('data_path.txt', 'r') as file:
    data_file = file.read().strip()
AdaptiveT=0
data = np.load('{}/Data.npy'.format(data_file), allow_pickle=True).item()
# data=np.load('{}/Data.npz'.format(data_file), allow_pickle=True)#np.load('{}_img/Data.npz'.format(args.out_filename), allow_pickle=True) 
# print(list(data))

argsSave = data['argsSave']
args.Tstep=argsSave[0]#1001
args.delta_t=argsSave[1]#0.05
args.L=argsSave[2]
args.print_step= argsSave[6]

SampleSum=data['SampleSum']
delta_T= data['delta_TSum']
if AdaptiveT: TimePoins=np.cumsum(delta_T)[np.arange(SampleSum.shape[0])*args.print_step]
else: TimePoins=np.cumsum(delta_T)[np.arange(SampleSum.shape[0])*args.print_step]*args.delta_t

Sites=args.L
X1,X2,X3=0,1,2
color=['C0','C1','C2','C3','C4','C5','C6','C7','C8']
colors = [f'C{i}' for i in range(Sites)]  # Adjusting colors dynamically
plotlist = np.unique(np.floor(np.linspace(1, Sites, 4) - 1)).astype(int)
legend = [f'X{i+1}' for i in plotlist]#['bA', 'cA', 'cB']#['a','b','c']#['A', 'B', 'C']#['a', 'b', 'c', 'A', 'B', 'C', 'bA', 'cA', 'cB']

out_filename = 'Schlogl/Schlogl_Sites'+str(args.L)+'_times10000_T1.0'#'Schlogl_nb0.9_Sites1_times1000_T1'#'Schlogl_nx50_Sites3_times10000_T1'
out_filename = f'SSA_Schlogl_2d/Schlogl2D_1x{Sites}_times100000_T1.0_nb1.0'
data=np.load(out_filename+'.npz')
# print(list(data)) 
#times = data['arr_0'] 
#comput_time=data['arr_1']  
#time_points = data['arr_3']
#species_total=data['arr_4']

times = data['times'] 
comput_time=data['comput_time']  
time_points = data['time_points']
species_total=data['species']
    
Gstep=time_points[1]



fig, axes = plt.subplots(1, 3, figsize=(32, 8),dpi=100)
# plt.figure(num=None,  dpi=300, edgecolor='k', linewidth=8)

for ax in axes.flatten():
    ax.tick_params(axis='x', labelsize=48)
    ax.tick_params(axis='y', labelsize=48)
    ax.set_xlabel('',fontsize=48)
    ax.set_ylabel('',fontsize=48)

###curve------------------------
plt.rc('font', size=48)
markersize0=8
step=2


ax=axes[0]
i=0
for Species in plotlist:#[1,2,5]:#range(0,16):#
    ax.plot(time_points,np.mean(species_total[:,Species,:],1),linewidth=5,color=color[i],label=legend[i])
    ax.plot(TimePoins[::step],np.mean(SampleSum[:,:,Species][::step],axis=1),
              marker='o',linestyle = 'None',markersize=markersize0,color=color[i])
    i=i+1

# plt.plot(time_points,np.mean(Gy_total,0),linewidth=6,color=color2,label='$G_y$')
# plt.plot(TimePoins[::step],np.mean(SampleSum[:,:,1][::step],axis=1),
#           marker='o',linestyle = 'None',color=color2,markersize=markersize0)
ax.set_xlabel("Time (s)")
ax.legend(numpoints=2,handletextpad=0.2,fontsize=45,loc='upper right')
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.set_title('Average Count')
# plt.show()
# plt.savefig('FigToggleSwitch\ToggleSwitch_panela1.svg', bbox_inches="tight", dpi=300)


#####mean-----------------------------------
legendsize=50
T=min(max(TimePoins),max(time_points))
NumPoints=10
markersize0=20
transparency=1-np.linspace(0, 0.99,NumPoints)
a=max(np.mean(SampleSum[:,:,X1],axis=1))+1
Range=[0,a]

ax=axes[1]
# Range=1#math.ceil(max(np.mean(Gx_total,0)))
i=0
for Species in plotlist:#[1,2,5]:
    k=0
    for tt in np.linspace(0,T,NumPoints):
        indexa=np.argmin(np.abs(time_points - tt))#int(tt/Gstep)
        indexb=np.argmin(np.abs(TimePoins - tt))#[int(tt/np.max(TimePoins)*TimePoins.shape[0])]
        # print(indexa, indexb)
        if tt==0:
            ax.plot(np.mean(species_total[indexa,Species,:]),np.mean(SampleSum[:,:,Species],axis=1)[indexb],
                      alpha=transparency[k], marker='o',linestyle = 'None',color=color[i],markersize=markersize0,label=legend[i])
        else:
            ax.plot(np.mean(species_total[indexa,Species,:]),np.mean(SampleSum[:,:,Species],axis=1)[indexb],
                      alpha=transparency[k], marker='o',linestyle = 'None',color=color[i],markersize=markersize0)
        k=k+1
    i=i+1    
ax.plot(Range,Range,linewidth=4,color='black')

ax.set_xlabel('Gillespie')
ax.set_ylabel('VAN')
ax.set_title('Mean')
ax.legend(fontsize=legendsize,loc='best')
# plt.xticks([0.4,0.6,0.8])
# plt.yticks([0,30])
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
# plt.legend(fontsize=legendsize,loc='best',handletextpad=0.2)
# plt.savefig('FigToggleSwitch_newSample\ToggleSwitch_panelb11.svg', bbox_inches="tight", dpi=300)

#####std-----------------------------------
legendsize=50
T=min(max(TimePoins),max(time_points))
NumTimePoints=10
markersize0=20
transparency=1-np.linspace(0, 0.99, NumPoints)
a=max(np.std(SampleSum[:,:,X1],axis=1))+1
Range=[0,a]

ax=axes[2]
# Range=1#math.ceil(max(np.mean(Gx_total,0)))
i=0
for Species in plotlist:#[1,2,5]:
    k=0
    for tt in np.linspace(0,T,NumPoints):
        indexa=np.argmin(np.abs(time_points - tt))#int(tt/Gstep)
        indexb=np.argmin(np.abs(TimePoins - tt))#[int(tt/np.max(TimePoins)*TimePoins.shape[0])]
        if tt==0:
            ax.plot(np.std(species_total[indexa,Species,:]),np.std(SampleSum[:,:,Species],axis=1)[indexb],
                      alpha=transparency[k], marker='o',linestyle = 'None',color=color[i],markersize=markersize0,label=legend[i])
        else:
            ax.plot(np.std(species_total[indexa,Species,:]),np.std(SampleSum[:,:,Species],axis=1)[indexb],
                      alpha=transparency[k], marker='o',linestyle = 'None',color=color[i],markersize=markersize0)
        k=k+1
    i=i+1    
ax.plot(Range,Range,linewidth=4,color='black')

ax.set_xlabel('Gillespie')
ax.set_ylabel('VAN')
ax.set_title('Std')
# ax.set_xlim(Range)
# ax.set_ylim(Range)
ax.legend(fontsize=legendsize,loc='best')
# plt.yticks([0,30])
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
# plt.legend(fontsize=legendsize,loc='best',handletextpad=0.2)
# plt.savefig('FigToggleSwitch_newSample\ToggleSwitch_panelb11.svg', bbox_inches="tight", dpi=300)

plt.subplots_adjust(hspace=0.4,wspace=0.3,bottom=0.1)
plt.savefig('{}/PlotSchlogl_1.jpg'.format(data_file), bbox_inches="tight", dpi=300)


def hellinger(p, q):
    """Hellinger operation.

    

    Args:

        p: Parameter forwarded to hellinger.

        q: Parameter forwarded to hellinger.

    

    Returns:

        Result produced by hellinger.

    """


    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
# Distribution---------------------------
plt.rc('font', size=38)
legendsize = 34
probrange = [0.06] * Sites
Range = [80] * Sites  # Adjust range for all species
Tmax = max(TimePoins)
Tlist = [Tmax/10,Tmax/2,Tmax]#[int(Tmax/10),int(Tmax/2),Tmax-1]#[90, 450, 900]
n_row = len(plotlist)  # Number of rows corresponds to Sites
n_col = len(Tlist)  # Number of columns corresponds to time points

fig, axes = plt.subplots(n_row, n_col, figsize=(9 * n_col, 8 * n_row), dpi=300)

for j, T in enumerate(Tlist):
    indexa=np.argmin(np.abs(time_points - T))#int(T/Gstep)
    indexb=np.argmin(np.abs(TimePoins - T))#[int(tt/np.max(TimePoins)*TimePoins.shape[0])]
    print(time_points[indexa], TimePoins[indexb])

    for k, i in enumerate(plotlist):
        if len(plotlist)>1:
            ax = axes[k, j]  # Select the correct subplot
        else:
            ax = axes[k]  # Select the correct subplot
        m1_G = species_total[indexa, i, :]
        m1_V = SampleSum[:, :, i][indexb]

        sample1 = m1_G#np.random.normal(loc=0, scale=1, size=1000)
        sample2 = m1_V#np.random.normal(loc=2, scale=1.5, size=1000)
        

        bins = np.linspace(min(sample1.min(), sample2.min()), max(sample1.max(), sample2.max()), 50)
        hist1, _ = np.histogram(sample1, bins=bins, density=True)
        hist2, _ = np.histogram(sample2, bins=bins, density=True)
        

        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        

        hellinger_distance = hellinger(hist1, hist2)
        hd = round(hellinger_distance, 3)
        print(f'Hellinger Distance: {hellinger_distance}')
        
        weights1 = np.ones_like(m1_G) / float(len(m1_G))
        weights2 = np.ones_like(m1_V) / float(len(m1_V))
        ax.hist([m1_G, m1_V], bins=Range[k], range=(0, Range[k]), weights=[weights1, weights2], color=['darkgrey', color[k]], alpha=0.7)
        
        ax.set_xlabel(legend[k])
        ax.set_ylabel('Probability')
        ax.set_title(f"$t=$ {T:.2f}")
        ax.set_xlim(right=Range[k])
        ax.set_ylim(top=probrange[k])
        ax.legend(['Gillespie', 'VAN'], fontsize=legendsize, title='$D_{HD}=$'+str(hd), title_fontsize=30,loc='upper left')
        for spine in ax.spines.values():
            spine.set_linewidth(2)

plt.subplots_adjust(hspace=0.5,wspace=0.3,bottom=0.1)
# plt.show()
plt.savefig('{}/PlotSchlogl_2.jpg'.format(data_file), bbox_inches="tight", dpi=300)

# Distribution---------------------------
fig, axes = plt.subplots(n_row, n_col, figsize=(9*n_col, 8*n_row), dpi=300)

for j, T in enumerate(Tlist):
    indexa = np.argmin(np.abs(time_points - T))
    print(time_points[indexa], TimePoins[indexa])

    for k, i in enumerate(plotlist):
        if len(plotlist)>1:
            ax = axes[k, j]  # Select the correct subplot
        else:
            ax = axes[k]  # Select the correct subplot
        m1_G = species_total[indexa, i, :]

        sample1 = m1_G


        bins = np.linspace(min(sample1), max(sample1), 50)
        hist1, _ = np.histogram(sample1, bins=bins, density=True)


        hist1 = hist1 / np.sum(hist1)


        weights1 = np.ones_like(m1_G) / float(len(m1_G))
        ax.hist(m1_G, bins=Range[k], range=(0, Range[k]), weights=weights1, color='darkgrey', alpha=0.7)

        ax.set_xlabel(legend[k])
        ax.set_ylabel('Probability')
        ax.set_title("$t=$" + str(T))
        ax.set_xlim(right=Range[k])
        ax.set_ylim(top=probrange[k])
        for spine in ax.spines.values():
            spine.set_linewidth(2)

plt.subplots_adjust(hspace=0.5, wspace=0.3, bottom=0.1)
plt.savefig('{}/PlotSchlogl_m1_G.jpg'.format(data_file), bbox_inches="tight", dpi=300)

# Distribution---------------------------
fig, axes = plt.subplots(n_row, n_col, figsize=(9*n_col, 8*n_row), dpi=300)

for j, T in enumerate(Tlist):
    indexb = np.argmin(np.abs(TimePoins - T))
    print(TimePoins[indexb])

    for k, i in enumerate(plotlist):
        if len(plotlist)>1:
            ax = axes[k, j]  # Select the correct subplot
        else:
            ax = axes[k]  # Select the correct subplot
        m1_V = SampleSum[:, :, i][indexb]

        sample2 = m1_V


        bins = np.linspace(min(sample2), max(sample2), 50)
        hist2, _ = np.histogram(sample2, bins=bins, density=True)


        hist2 = hist2 / np.sum(hist2)


        weights2 = np.ones_like(m1_V) / float(len(m1_V))
        ax.hist(m1_V, bins=Range[k], range=(0, Range[k]), weights=weights2, color=color[k], alpha=0.7)

        ax.set_xlabel(legend[k])
        ax.set_ylabel('Probability')
        ax.set_title("$t=$" + str(T))
        ax.set_xlim(right=Range[k])
        ax.set_ylim(top=probrange[k])
        for spine in ax.spines.values():
            spine.set_linewidth(2)

plt.subplots_adjust(hspace=0.5, wspace=0.3, bottom=0.1)
plt.savefig('{}/PlotSchlogl_m1_V.jpg'.format(data_file), bbox_inches="tight", dpi=300)


#hist2d-----------------------------
from matplotlib.colors import LogNorm
plt.rc('font', size=24)
Tmax = max(TimePoins)
Tlist = [0.01,0.14,0.44,0.75]#[Tmax/10,Tmax/2,Tmax]#[int(Tmax/10),int(Tmax/2),Tmax-1]#[90, 450, 900]
newcmp='inferno'#'viridis'
n_row = 1
n_col = len(Tlist)
k1=0#plotlist[2]
k2=1#plotlist[-1]
print("hist2d:",k1,k2)
print(legend)
fig, axes = plt.subplots(n_row, n_col, figsize=(10*n_col, 9*n_row), dpi=300)
for j, T0 in enumerate(Tlist):
    ax = axes[j] 
    indexb=np.argmin(np.abs(TimePoins - T0))#[int(tt/np.max(TimePoins)*TimePoins.shape[0])]
    T=indexb#round(T0/np.max(TimePoins)*TimePoins.shape[0])
    print(T,TimePoins[T])
    bins=[35,35]#[int(max(SampleSum[T,:,X2])),int(max(SampleSum[T,:,X3]))]
    # ax.set_facecolor([68/255,1/255,80/255])
    h=ax.hist2d(SampleSum[T,:,k1],SampleSum[T,:,k2],bins=bins,cmap=newcmp,density=True,vmax=1e-3)#norm=mpl.colors.LogNorm(vmin=8e-5,vmax=2e-3))
    # plt.colorbar(h[3],label='Probability',ax=ax)#pad=0.1,ticks=[0,0.05,0.10,0.15],fraction=0.1,orientation='horizontal')
    ax.set_facecolor([0/255,0/255,0/255])
    ax.set_xlabel("X"+str(k1+1))
    ax.set_ylabel("X"+str(k2+1))
    ax.set_title("$t=$"+str(T0))
    ax.set_xlim((0,70))
    ax.set_ylim((0,70))
    # plt.xticks([])
    # fig.set_size_inches(11,8)
    # fig.set_size_inches(2,8)
    # plt.savefig('FigToggleSwitch\ToggleSwitch_paneld_T'+str(T0)+'.svg', bbox_inches="tight", dpi=300)
    # plt.show()


fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.78])
plt.colorbar(h[3], cax=cbar_ax,label='Probability')


# plt.subplots_adjust(hspace=0.5,wspace=0.3,bottom=0.1)
# plt.show()
plt.savefig('{}/PlotSchlogl_3.jpg'.format(data_file), bbox_inches="tight", dpi=300)

#%% heatmap with log-probability via KDE -----------------------------
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', size=58)
title_size=62
tick_size=58

Tmax = max(TimePoins)
Tlist = [Tmax/100,Tmax/5,Tmax]#[0.05, 0.5, 1]#[0.01, 0.03, 0.06]#
k1, k2 = Sites-2, Sites-1
M = 70
newcmp = 'viridis'
norm = LogNorm(vmin=5e-7, vmax=5e-4) 
# norm = Normalize(vmin=0, vmax=0.0003)
print("kde logp-style plot (LogNorm cmap):", k1, k2)
print("Legend:", legend)

fig, axes = plt.subplots(1, len(Tlist), figsize=(9 * len(Tlist), 8), dpi=400)

x = np.linspace(0, M, M)
y = np.linspace(0, M, M)
X, Y = np.meshgrid(x, y)
grid_points = np.vstack([X.ravel(), Y.ravel()])

for j, T0 in enumerate(Tlist):
    ax = axes[j]
    T = np.argmin(np.abs(TimePoins - T0))
    print(f"T index {T}, time = {TimePoins[T]:.3f}")
    

    data = np.vstack([SampleSum[T, :, k1], SampleSum[T, :, k2]])  # shape (2, N)
    kde = gaussian_kde(data, bw_method=0.2)
    prob_density = kde(grid_points).reshape(X.shape)


    im = ax.imshow(prob_density.T, extent=[0, M, 0, M], origin='lower',
                   cmap=newcmp, norm=norm, aspect='equal')

    ax.set_facecolor([68/255,1/255,80/255])
    ax.set_xlabel(f'$X_{k1+1}$',fontsize=tick_size)
    if j ==0:
        ax.set_ylabel(f'VAN\n$X_{k2+1}$',fontsize=tick_size)
    ax.set_title(f"$t={T0:.2f}$",fontsize=title_size)
    ax.set_xlim(0, M)
    ax.set_ylim(0, M)
    # ax.set_yticks([0,35,70])
    # ax.set_xticks([0,35,70])
    ax.tick_params(labelsize=tick_size-5)


fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im, cax=cbar_ax, label="Probability (log scale)")
# plt.suptitle('VAN')
plt.subplots_adjust(wspace=0.25)
plt.savefig('{}/Plot_kde_lognorm_V.jpg'.format(data_file), bbox_inches="tight", dpi=300)
# % Gillespie -----------------------------
# Tlist = [0.01, 0.03, 0.06]
plt.rc('font', size=58)
title_size=62
tick_size=58

fig, axes = plt.subplots(1, len(Tlist), figsize=(9 * len(Tlist), 8), dpi=400)

x = np.linspace(0, M, M)
y = np.linspace(0, M, M)
X, Y = np.meshgrid(x, y)
grid_points = np.vstack([X.ravel(), Y.ravel()])

for j, T0 in enumerate(Tlist):
    ax = axes[j]
    T = np.argmin(np.abs(time_points - T0))
    print(f"T index {T}, time = {time_points[T]:.3f}")
    

    data = np.vstack([species_total[T, k1, :], species_total[T, k2, :]])  # shape (2, N)
    kde = gaussian_kde(data, bw_method=0.2)
    prob_density = kde(grid_points).reshape(X.shape)


    im = ax.imshow(prob_density.T, extent=[0, M, 0, M], origin='lower',
                   cmap=newcmp, norm=norm, aspect='equal')

    ax.set_facecolor([68/255,1/255,80/255])
    ax.set_xlabel(f'$X_{k1+1}$',fontsize=tick_size)
    if j ==0:
        ax.set_ylabel(f'Gillespie\n$X_{k2+1}$',fontsize=tick_size)
    ax.set_title(f"$t={T0:.2f}$",fontsize=title_size)
    ax.set_xlim(0, M)
    ax.set_ylim(0, M)
    # ax.set_yticks([0,35,70])
    # ax.set_xticks([0,35,70])
    ax.tick_params(labelsize=tick_size-5)


fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im, cax=cbar_ax, label="Probability (log scale)")
# plt.suptitle('Gillespie')
plt.subplots_adjust(wspace=0.25)
plt.savefig('{}/Plot_kde_lognorm_G.jpg'.format(data_file), bbox_inches="tight", dpi=300)



