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
legend=['MKP3','K','Kpp']

species_num=16

out_filename = 'MAPK_times1000_T10000.0'
data=np.load(out_filename+'.npz')
# print(list(data)) 
times = data['arr_0']   
time_points = data['arr_1']
species_total=data['arr_2']
Gstep=time_points[1]
with open('data_path.txt', 'r') as file:
    data_file = 'out\\MAPK\\MAPK_L16_S1_M10_T500001_dt0.01_batch2000_samplingdefault\\nd1_nw8_NADE_NatGrad_lr0.5_epoch5_LossReverseKL1_IniDistdelta_Para1_bias_cg1\\out_img'#file.read().strip()
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
print(species_total.shape,SampleSum.shape)
print(time_points[-1])
plt.rc('font', size=16)
# for i in range(0,species_num):
#     plt.figure()
#     m1_G=species_total[:,i,:].flatten()
#     print(i,max(m1_G))
#     weights1 = np.ones_like(m1_G) / float(len(m1_G))
#     plt.hist(m1_G,range=(0,10),bins=10,weights=weights1)
#     plt.title(i)
#     plt.show()
# # for i in range(0,species_num):
#     # plt.figure()
#     # m1_G=SampleSum[-1,:,i]
#     # weights1 = np.ones_like(m1_G) / float(len(m1_G))
#     # plt.hist(m1_G,range=(0,10),bins=10,weights=weights1)
#     # plt.title(i)
#     plt.close()


fig, axes = plt.subplots(1, 3, figsize=(32, 8),dpi=100)
# plt.figure(num=None,  dpi=400, edgecolor='k', linewidth=8)

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
for Species in [1,2,5]:#range(0,16):#
    ax.plot(time_points,np.mean(species_total[:,Species,:],1),linewidth=5,color=color[i],label=legend[i])
    ax.plot(TimePoins[::step],np.mean(SampleSum[:,:,Species][::step],axis=1),
              marker='o',linestyle = 'None',markersize=markersize0,color=color[i])
    i=i+1

# plt.plot(time_points,np.mean(Gy_total,0),linewidth=6,color=color2,label='$G_y$')
# plt.plot(TimePoins[::step],np.mean(SampleSum[:,:,1][::step],axis=1),
#           marker='o',linestyle = 'None',color=color2,markersize=markersize0)
ax.set_xlabel("Time (s)")
ax.legend(numpoints=2,handletextpad=0.2,fontsize=40,loc='upper right')
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.set_title('Average Count')
# plt.show()
# plt.savefig('FigToggleSwitch\ToggleSwitch_panela1.svg', bbox_inches="tight", dpi=400)


#####mean-----------------------------------
legendsize=50
T=min(int(max(TimePoins)),int(max(time_points)))
NumPoints=10
markersize0=20
transparency=1-np.linspace(0, 0.99,NumPoints)
Range=[0,3]

ax=axes[1]
# Range=1#math.ceil(max(np.mean(Gx_total,0)))
i=0
for Species in [1,2,5]:
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
# plt.xticks([0.4,0.6,0.8])
# plt.yticks([0,30])
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
# plt.legend(fontsize=legendsize,loc='best',handletextpad=0.2)
# plt.savefig('FigToggleSwitch_newSample\ToggleSwitch_panelb11.svg', bbox_inches="tight", dpi=400)

#####std-----------------------------------
legendsize=50
T=min(int(max(TimePoins)),int(max(time_points)))
NumTimePoints=10
markersize0=20
transparency=1-np.linspace(0, 0.99, NumPoints)
a=max(np.std(SampleSum[:,:,5],axis=1))+0.1
Range=[0,a]

ax=axes[2]
# Range=1#math.ceil(max(np.mean(Gx_total,0)))
i=0
for Species in [1,2,5]:
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
# plt.xticks([0.4,0.6,0.8])
# plt.yticks([0,30])
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
# plt.legend(fontsize=legendsize,loc='best',handletextpad=0.2)
# plt.savefig('FigToggleSwitch_newSample\ToggleSwitch_panelb11.svg', bbox_inches="tight", dpi=400)

plt.subplots_adjust(hspace=0.4,wspace=0.3,bottom=0.1)
plt.savefig('{}/PlotMAPK_1.jpg'.format(data_file), bbox_inches="tight", dpi=400)

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
probrange = [1, 0.8, 1]
Range = [5, 10, 10]
Tmax = min(int(max(TimePoins)),int(max(time_points)))
Tlist = [Tmax/10,Tmax/2,Tmax]#[int(Tmax/10),int(Tmax/2),Tmax-1]#[90, 450, 900]
n_row = len(Tlist)
n_col = 3
fig, axes = plt.subplots(n_row, n_col, figsize=(9*n_col, 8*n_row), dpi=400)
# fig.tight_layout(pad=1.0)

for j, T in enumerate(Tlist):
    indexa=np.argmin(np.abs(time_points - T))#int(T/Gstep)
    indexb=np.argmin(np.abs(TimePoins - T))#[int(tt/np.max(TimePoins)*TimePoins.shape[0])]
    # print(indexa,'???')
    print(time_points[indexa], TimePoins[indexb])

    for k, i in enumerate([1, 2, 5]):
        ax = axes[j, k]  # Select the correct subplot
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
        ax.hist([m1_G, m1_V], bins=Range[k], range=(0, Range[k]), weights=[weights1, weights2], color=['darkgrey', color[k]], alpha=0.7, orientation='horizontal')
        
        ax.set_ylabel(legend[k])
        ax.set_xlabel('Probability')
        ax.set_title("$t=$" + str(T))
        ax.set_ylim(top=Range[k])
        ax.set_xlim(right=1)#probrange[k])
        ax.legend(['Gillespie', 'VAN'], fontsize=legendsize,title='$D_{HD}=$'+str(hd),title_fontsize=30)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

plt.subplots_adjust(hspace=0.5,wspace=0.3,bottom=0.1)
plt.show()
plt.savefig('{}/PlotMAPK_2.jpg'.format(data_file), bbox_inches="tight", dpi=400)

# hist2d---------------------------------
newcmp='viridis'

plt.rc('font', size=38)
n_row = 1
n_col = len(Tlist)
fig, axes = plt.subplots(n_row, n_col, figsize=(10*n_col, 8*n_row), dpi=400)

# T0=9000
# for T0 in range(0,10000,1000):
for j, T0 in enumerate(Tlist):
    ax = axes[j] 
    indexb=np.argmin(np.abs(TimePoins - T0))#[int(tt/np.max(TimePoins)*TimePoins.shape[0])]
    T=indexb#round(T0/np.max(TimePoins)*TimePoins.shape[0])
    print(T,TimePoins[T])
    max1=10
    max2=10
    bins=[int(max(SampleSum[T,:,2])),int(max(SampleSum[T,:,5]))]
    ax.set_facecolor([68/255,1/255,80/255])
    h=ax.hist2d(SampleSum[T,:,2],SampleSum[T,:,5],bins=bins,cmap=newcmp,density=True,norm=mpl.colors.LogNorm(vmin=1e-4,vmax=2e-1))
    # plt.colorbar(h[3], ax=ax,label='Probability')#ax.set_colorbar(label='Probability')
    # plt.colorbar(orientation='vertical',ticks=[0,0.005,0.01])#pad=0.1,ticks=[0,0.05,0.10,0.15],fraction=0.1,orientation='horizontal')
    ax.set_xlabel("$K$")
    ax.set_ylabel("$Kpp$")
    ax.set_title("$t=$"+str(T0))
    # plt.xticks([])
    ax.set_xlim(right=max1)
    ax.set_ylim(top=max2)


fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.78])
plt.colorbar(h[3], cax=cbar_ax,label='Probability')


# from mpl_toolkits.axes_grid1 import make_axes_locatable
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.2)


# plt.colorbar(h[3], cax=cax)

plt.subplots_adjust(hspace=0.5,wspace=0.3,bottom=0.1)
plt.show()
plt.savefig('{}/PlotMAPK_3.jpg'.format(data_file), bbox_inches="tight", dpi=400)


