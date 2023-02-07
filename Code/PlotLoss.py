import numpy as np
from args import args
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl


jet = cm.get_cmap('jet')
Number=15
jet_12_colors = jet(np.linspace(0, 1, Number))
viridis=cm.get_cmap('viridis')
viridis_colors=viridis(np.linspace(0, 1, 5))

color1=jet_12_colors[0,:]
color2=jet_12_colors[4,:]
color3=jet_12_colors[12,:]
color4='#A6A6A6'
coloryel='yellow'


AdaptiveT=0
Name= 'DataToggleSwitch/Data-ToggleSwitch_oldSample_M81_T8001_delta1_1'
data=np.load(str(Name)+'.npz', allow_pickle=True) 
print(list(data))

argsSave = data['arr_1']
args.Tstep=argsSave[0]#1001
args.delta_t=argsSave[1]#0.005
args.L=argsSave[2]
args.print_step= argsSave[6]


Lossmean=data['arr_7']
Lossstd=data['arr_8']

SampleSum=data['arr_5']
delta_T= data['arr_6']
if AdaptiveT: TimePoins=np.cumsum(delta_T)[np.arange(SampleSum.shape[0])*args.print_step]
else: TimePoins=np.cumsum(delta_T)[np.arange(SampleSum.shape[0])*args.print_step]*args.delta_t


# LossMean----------------------------------------
plt.rc('font', size=30)
step=0
linew=3
plt.figure(num=None,  dpi=400, edgecolor='k', linewidth=8)
fig, axes = plt.subplots(1,1)
fig.tight_layout()
plt.plot(range(len(Lossmean[step]))[::4],[abs(i) for i in Lossmean[step]][::4],
            label='Time step '+str(step+1),lw=linew,color=color1)

axes.add_patch(mpl.patches.Rectangle((4900,0),100,100,color=coloryel,zorder=-1))
axes = plt.gca()
axes.set_yscale('log')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper center')
plt.xticks([0,2500,5000])
plt.ylim([1e-7,1e2])
fig.set_size_inches(9, 8)
# plt.show()
plt.savefig('FigToggleSwitch\ToggleSwitch_LossShade1.svg', bbox_inches="tight", dpi=400)

plt.figure(num=None,  dpi=400, edgecolor='k', linewidth=8)
plt.tight_layout()
fig, axes = plt.subplots(1,1)
fig.tight_layout()
plt.rc('font', size=30)
step=4000
plt.plot(range(len(Lossmean[step])),[abs(i) for i in Lossmean[step]],
          label='Time step '+str(step+1),lw=linew,color=color2)

step=8000
plt.plot(range(len(Lossmean[step])),[abs(i) for i in Lossmean[step]],
          label='Time step '+str(step+1),lw=linew,color=color3)

axes.add_patch(mpl.patches.Rectangle((90,0),10,100,color=coloryel,zorder=-1,alpha=0.7))
axes = plt.gca()
axes.set_yscale('log')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([1e-7,1e2])
fig.set_size_inches(9, 8)
# plt.show()
plt.savefig('FigToggleSwitch\ToggleSwitch_LossShade2.svg', bbox_inches="tight", dpi=400)

finalLossmean=[]
finalLossstd=[]
for i in range(len(Lossmean)):
    finalLossmean.append(abs(Lossmean[i][-1]))
    finalLossstd.append(Lossstd[i][-1])

plt.rc('font', size=30)
plt.figure(num=None,  dpi=400, edgecolor='k', linewidth=8)
fig, axes = plt.subplots(1,1)
fig.tight_layout()
plt.plot(range(len(Lossmean))[::10],finalLossmean[::10],
          label='Converged Loss',lw=3,color=color4)
axes = plt.gca()
axes.set_yscale('log')
# plt.xticks([0,2000,4000,6000,8000])
plt.ylabel('Loss')
plt.xlabel('Time step')
# plt.legend()
plt.ylim([1e-8,1e0])
fig.set_size_inches(21.5, 8)
plt.savefig('FigToggleSwitch\ToggleSwitch_LossShade3.svg', bbox_inches="tight", dpi=400)


# LossVar-------------------------------------------
plt.rc('font', size=30)
step=0
linew=3
plt.figure(num=None,  dpi=400, edgecolor='k', linewidth=8)
fig, axes = plt.subplots(1,1)
fig.tight_layout()
plt.plot(range(len(Lossstd[step]))[::4],[i**2 for i in Lossstd[step]][::4],
            label='Time step '+str(step+1),lw=linew,color=color1)
axes.add_patch(mpl.patches.Rectangle((4900,0),100,100,color=coloryel,zorder=-1))
axes = plt.gca()
axes.set_yscale('log')
plt.ylabel('Loss Var')
plt.xlabel('Epoch')
plt.legend(loc='upper center')
plt.xticks([0,2500,5000])
plt.ylim([1e-7,1e2])
fig.set_size_inches(9, 8)
# plt.show()
plt.savefig('FigToggleSwitch\ToggleSwitch_LossVar1.svg', bbox_inches="tight", dpi=400)

plt.figure(num=None,  dpi=400, edgecolor='k', linewidth=8)
plt.tight_layout()
fig, axes = plt.subplots(1,1)
fig.tight_layout()
plt.rc('font', size=30)
linew=3
step=4000
plt.plot(range(len(Lossstd[step])),[i**2 for i in Lossstd[step]],
          label='Time step '+str(step+1),lw=linew,color=color2)
# step=5000
# plt.plot(range(len(Lossmean[step])),[abs(i) for i in Lossmean[step]],label='Time step '+str(step+1),lw=linew,color=color3)
step=8000
plt.plot(range(len(Lossstd[step])),[i**2 for i in Lossstd[step]],
          label='Time step '+str(step+1),lw=linew,color=color3)
axes.add_patch(mpl.patches.Rectangle((90,0),10,100,color=coloryel,zorder=-1,alpha=0.7))
axes = plt.gca()
axes.set_yscale('log')
plt.ylabel('Loss Var')
plt.xlabel('Epoch')
plt.legend()#(handlelength=1.5)#borderpad=0.7)
plt.ylim([1e-7,1e2])
fig.set_size_inches(9, 8)
# plt.show()
plt.savefig('FigToggleSwitch\ToggleSwitch_LossVar2.svg', bbox_inches="tight", dpi=400)

finalLoss=[]
for i in range(len(Lossstd)):
    finalLoss.append(Lossstd[i][-1]**2)
plt.rc('font', size=30)
plt.figure(num=None,  dpi=400, edgecolor='k', linewidth=8)
fig, axes = plt.subplots(1,1)
fig.tight_layout()
plt.plot(range(len(Lossstd))[::10],finalLoss[::10],
          label='Converged Loss',lw=3,color=color4)
axes = plt.gca()
axes.set_yscale('log')
# plt.xticks([0,2000,4000,6000,8000])
plt.ylabel('Loss Var')
plt.xlabel('Time step')
# plt.legend()
plt.ylim([1e-8,1e0])
fig.set_size_inches(21.5, 8)
# plt.show()
plt.savefig('FigToggleSwitch\ToggleSwitch_LossVar3.svg', bbox_inches="tight", dpi=400)
   