import numpy as np
import biocircuits
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

species_num=9
Para=2

def FFL_propensity(
    propensities, population, t, rbA,fbA,rcA,fcA,rcB,fcB,sA,dA,sB,sBk1,dB,sC,sCk2,sCk3,dC
):
    """Ffl propensity operation.
    """


    c, C, cB, cA, b, B, bA, a, A = population

    propensities[0] = rbA * b * A
    propensities[1] = fbA * bA
    propensities[2] = rcA * c * A
    propensities[3] = fcA * cA
    propensities[4] = rcB * c * B
    propensities[5] = fcB * cB
    propensities[6] = sA * a
    propensities[7] = dA * A
    propensities[8] = sB * b
    propensities[9] = sBk1 * bA
    propensities[10] = dB * B
    propensities[11] = sC * c
    propensities[12] = sCk2 * cB
    propensities[13] = sCk3 * cA
    propensities[14] = dC * C
    
FFL_update = np.array(
    [   # c, C, cB, cA, b, B, bA, a, A
        [0, 0, 0, 0, -1, 0, 1, 0, -1],
        [0, 0, 0, 0, 1, 0, -1, 0, 1],
        [-1, 0, 0, 1, 0, 0, 0, 0, -1],
        [1, 0, 0, -1, 0, 0, 0, 0, 1],
        [-1, 0, 1, 0, 0, -1, 0, 0, 0],
        [1, 0, -1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=int)
  
  
sA=sB=sC=10
dA=dB=dC=1
rbA=rcA=rcB=0.005
fbA=fcA=fcB=0.1

if Para==1:
# bimodality in both proteins B and C
    k1=3.0
    k2=0.5
    k3=5.0
if Para==2:
#  tri-modality in protein C and bimodality in protein B
    k1=0.1
    k2=2.75
    k3=5.0

sBk1=sB*k1
sCk3=sC*k3
sCk2=sC*k2
# ku2 = 9.0 *2 # To test the wrong implementation on master equation 

FFL_args = (rbA,fbA,rcA,fcA,rcB,fcB,sA,dA,sB,sBk1,dB,sC,sCk2,sCk3,dC)

# State with 10 copies of everything, nothing bound to operators
#FFL_pop_0 = np.array([10, 10, 10, 10, 10, 10, 0, 0, 0], dtype=int)
#FFL_pop_0 = np.array([11, 11, 11, 11, 11, 11, 2, 2, 2], dtype=int) # follow VAN's learnt initial number
FFL_pop_0 = np.zeros(shape=species_num,dtype=int)#np.array([0, 0, 0,0, 0, 0,0, 0, 0], dtype=int) # follow VAN's learnt initial number
FFL_pop_0[0]=1
FFL_pop_0[4]=1
FFL_pop_0[7]=1
T=50#5e3#200#30#100
FFL_time_points = np.linspace(0, T, 100)#(0, 40000, 2001)#(0, 10000, 501)##(0,2000,126)#(0, 80000, 4001)


######################

Run=0
times=10000#1000#00


out_filename = 'FFLo_times'+str(times)+'_T'+str(T)
if Run==1:
    
    time1=time.time()
    for jiadeyu in range(times):
         
        # Perform the Gillespie simulation
        pop = biocircuits.gillespie_ssa(
            FFL_propensity,
            FFL_update,
            FFL_pop_0,
            FFL_time_points,
            args=FFL_args,
        )
        
        if jiadeyu==0:species_total=pop[0,:,:]
        else:species_total=np.dstack((species_total,pop[0,:,:]))
        
        print(jiadeyu)
        # print(pop[0,:,0])
        if jiadeyu<3:
            plt.rc('font', size=20)
            plt.figure()
            plt.plot(FFL_time_points,pop[0,:,3],label='A')
            plt.plot(FFL_time_points,pop[0,:,4],label='B')
            plt.plot(FFL_time_points,pop[0,:,5],label='C')
            plt.xlabel("Time (s)")
            plt.ylabel("copy number")
            plt.grid()
            plt.legend()
            # plt.ylim((-0.5,5.5))
            plt.show()
    time2=time.time()
    comput_time=(time2-time1)/60
    print('Comput Time (min)',comput_time)

    np.savez('{}'.format(out_filename),np.array(times),np.array(comput_time),np.array(FFL_time_points),species_total)    
else:
    data=np.load(out_filename+'.npz')
    print(list(data))    
    FFL_time_points = data['arr_2']
    species_total = data['arr_3']

c, C, cB, cA, b, B, bA, a, A=0, 1, 2, 3, 4, 5, 6, 7, 8
plotlist=[A,B,C]#[bA,cA,cB]
legend=['c', 'C', 'cB', 'cA', 'b', 'B', 'bA', 'a', 'A']#['a', 'b', 'c', 'A', 'B', 'C', 'bA', 'cA', 'cB']
colors=['C0','C1','C2','C3','C4','C5','C6','C7','C8']
print(species_total.shape)
plt.rc('font', size=16)
plt.figure(num=None,  dpi=400, edgecolor='k')
fig, axes = plt.subplots(2,1)
fig.tight_layout()
ax = plt.subplot(1,1, 1)

for Species in plotlist:#[A,B,C]:#np.arange(species_num):#[1,2,5]:#np.arange(16):
    if Species==A:
        plt.errorbar(FFL_time_points, np.mean(species_total[:,Species,:],1),  yerr=np.std(species_total[:,Species,:],1),label='A')    
    if Species==B:
        plt.errorbar(FFL_time_points, np.mean(species_total[:,Species,:],1),  yerr=np.std(species_total[:,Species,:],1),label='B')    
    if Species==C:
        plt.errorbar(FFL_time_points, np.mean(species_total[:,Species,:],1),  yerr=np.std(species_total[:,Species,:],1),label='C')    
    if Species!=A and Species!=B and Species!=C:
        plt.errorbar(FFL_time_points, np.mean(species_total[:,Species,:],1),  yerr=np.std(species_total[:,Species,:],1),label=legend[Species])    

# plt.ylim((-1,4))
# plt.yticks([0,1,2,3])
plt.xlabel('Time (s)')
plt.ylabel('#')
plt.legend()
plt.title('Gillespie-times'+str(times))
fig.set_size_inches(9, 8)

# for i in range(0,species_num):
#     plt.figure()
#     m1_G=species_total[-1,i,:].flatten()
#     weights1 = np.ones_like(m1_G) / float(len(m1_G))
#     plt.hist(m1_G,range=(0,80),bins=80,weights=weights1)
#     plt.title(i)
#     plt.show()
# # for i in range(0,species_num):
#     # plt.figure()
#     # m1_G=SampleSum[-1,:,i]
#     # weights1 = np.ones_like(m1_G) / float(len(m1_G))
#     # plt.hist(m1_G,range=(0,10),bins=10,weights=weights1)
#     # plt.title(i)
#     plt.close()

# ###curve------------------------

# fig, axes = plt.subplots(1, 1, figsize=(9, 8),dpi=100)
# # plt.figure(num=None,  dpi=400, edgecolor='k', linewidth=8)
# plt.rc('font', size=48)
# markersize0=8
# step=2


# ax=axes
# i=0
# for Species in [A,B,C]:#range(0,16):#
#     ax.plot(FFL_time_points,np.mean(species_total[:,Species,:],1),linewidth=5,color=colors[i],label=legend[i])
#     # ax.plot(TimePoins[::step],np.mean(SampleSum[:,:,Species][::step],axis=1),
#     #           marker='o',linestyle = 'None',markersize=markersize0,color=color[i])
#     i=i+1

# # plt.plot(time_points,np.mean(Gy_total,0),linewidth=6,color=color2,label='$G_y$')
# # plt.plot(TimePoins[::step],np.mean(SampleSum[:,:,1][::step],axis=1),
# #           marker='o',linestyle = 'None',color=color2,markersize=markersize0)
# ax.set_xlabel("Time (s)")
# ax.legend(numpoints=2,handletextpad=0.2,fontsize=40,loc='upper right')
# ax.spines['bottom'].set_linewidth(1.5)
# ax.spines['top'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['right'].set_linewidth(1.5)
# ax.set_title('Average Count')
# plt.show()
# # plt.savefig('Fig\FFL_curve.svg', bbox_inches="tight", dpi=400)


#Dis-------------------------------
plt.rc('font', size=38)
legendsize = 34
probrange = [0.5,0.5,0.5]#[0.4, 0.3, 0.1]
Range = [30,30,30]#[30, 25, 80]
Tmax = int(max(FFL_time_points))
Tlist = [int(Tmax/10),int(Tmax/2),Tmax-1]#[0.5,2,4,8]#[int(Tmax/10),int(Tmax/2),Tmax-1]#[90, 450, 900]
n_row = len(Tlist)
n_col = 3
fig, axes = plt.subplots(n_row, n_col, figsize=(9*n_col, 8*n_row), dpi=400)
# fig.tight_layout(pad=1.0)

for j, T in enumerate(Tlist):
    indexa=np.argmin(np.abs(FFL_time_points - T))#int(T/Gstep)
    # indexb=np.argmin(np.abs(TimePoins - T))#[int(tt/np.max(TimePoins)*TimePoins.shape[0])]
    # print(time_points[indexa])#, TimePoins[indexb])
    if T>2:
      probrange = [0.5,0.25,0.12]#[0.4, 0.3, 0.1]
      Range = [30,30,80]#[30, 25, 80]  
    for k, i in enumerate([A, B, C]):
        ax = axes[j, k]  # Select the correct subplot
        m1_G = species_total[indexa, i, :]
        # m1_V = SampleSum[:, :, i][indexb]

        weights1 = np.ones_like(m1_G) / float(len(m1_G))
        # weights2 = np.ones_like(m1_V) / float(len(m1_V))
        # ax.hist([m1_G, m1_V], bins=Range[k], range=(0, Range[k]), weights=[weights1, weights2], color=['darkgrey', color[k]], alpha=0.7, orientation='horizontal')
        ax.hist(m1_G, bins=Range[k], range=(0, Range[k]), weights=weights1, color=colors[k], alpha=0.7, orientation='horizontal')
        ax.set_ylabel(legend[i])
        ax.set_xlabel('Probability')
        ax.set_title("$t=$" + str(T))
        ax.set_ylim(top=Range[k])
        ax.set_xlim(right=probrange[k])
        ax.legend(['Gillespie'], fontsize=legendsize)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

plt.subplots_adjust(hspace=0.5,wspace=0.3,bottom=0.1)
# plt.show()
plt.savefig('Fig/FFL_dis.jpg', bbox_inches="tight", dpi=400)


#hist2d-----------------------------
from matplotlib.colors import LogNorm
plt.rc('font', size=24)
Tmax = int(max(FFL_time_points))
Tlist = [1,int(Tmax/10),int(Tmax/2),Tmax-1]#[90, 450, 900]
newcmp='viridis'
n_row = 1
n_col = len(Tlist)
fig, axes = plt.subplots(n_row, n_col, figsize=(8*n_col, 10*n_row), dpi=400)
for j, T0 in enumerate(Tlist):
    indexa=np.argmin(np.abs(FFL_time_points - T0))#[int(tt/np.max(TimePoins)*TimePoins.shape[0])]
    T=indexa#round(T0/np.max(TimePoins)*TimePoins.shape[0])
    bins=[int(max(species_total[T,B,:])),int(max(species_total[T,C,:]))]
    # plt.figure(num=None,  dpi=400, edgecolor='k', linewidth=4)
    ax = axes[j]#plt.subplot(1,1, 1,facecolor=[68/255,1/255,80/255])
    # SampleSum[T,10,0]=50;SampleSum[T,10,1]=50;
    h=ax.hist2d(species_total[T,B,:],species_total[T,C,:],bins=bins,cmap=newcmp,density=True,norm=mpl.colors.LogNorm(vmax=1e-2,vmin=1e-3))
    # plt.colorbar(h[3],label='Probability',ax=ax)#pad=0.1,ticks=[0,0.05,0.10,0.15],fraction=0.1,orientation='horizontal')
    ax.set_facecolor([68/255,1/255,80/255])
    ax.set_xlabel("$B$")
    ax.set_ylabel("$C$")
    ax.set_title("$t=$"+str(T0))
    ax.set_xlim(right=20)
    ax.set_ylim(top=65)
    # plt.xticks([])
    # fig.set_size_inches(11,8)
    # fig.set_size_inches(2,8)
    # plt.savefig('FigToggleSwitch\ToggleSwitch_paneld_T'+str(T0)+'.svg', bbox_inches="tight", dpi=400)
    # plt.show()


fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.78])
plt.colorbar(h[3], cax=cbar_ax,label='Probability')

plt.subplots_adjust(hspace=0.5,wspace=0.3,bottom=0.1)
# plt.show()
plt.savefig('Fig/FFL_hist2d.jpg', bbox_inches="tight", dpi=400)
