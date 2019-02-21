import sys
import os
import numpy as np
import re
import fnmatch
import matplotlib
import pylab as pl
pl.style.use('ggplot')


gpu_logfiles = [
    'out.profile.a_-35',
    'out.profile.a_-30',
    'out.profile.a_-20'
]
gpu_part_cnt = []

for f in gpu_logfiles:
    data = np.loadtxt(f)
    gpu_part_cnt.append(data[:,1])

cpu_logfiles = [
    'CPU_simulations_to_match/out.cpu.a-35',
    'CPU_simulations_to_match/out.cpu.a-30',
    'CPU_simulations_to_match/out.cpu.a-20'
]
cpu_part_cnt = []

for f in cpu_logfiles:
    data = np.loadtxt(f,skiprows=4)
    cpu_part_cnt.append(data[:,2])

# Verify data integrity
[gpu_part_cnt,cpu_part_cnt] = [np.array(i) for i in [gpu_part_cnt,cpu_part_cnt]]
if (gpu_part_cnt.shape != cpu_part_cnt.shape):
    raise Exception("Data Incomplete: Check data integrity.")


# PLOT -------------------------------------------------------------------------
MARKERSIZE = 5
LINEWIDTH = 1
FONTSIZE = 7
colorset = ['#E24A33','cadetblue','#777777']     # ggplot colorset

# take out the tails
[gpu_part_cnt,cpu_part_cnt] = [ i[:,1:-1] for i in [gpu_part_cnt,cpu_part_cnt]]
xaxis = np.linspace(0,1,num=len(gpu_part_cnt[0]))
text = ['$A_{SL}=-35$','$A_{SL}=-30$','$A_{SL}=-20$']

fig0, axes = pl.subplots(3,1,figsize=(4.0,4.0),sharex=True)
fig0.subplots_adjust(left=0.14,bottom=0.14,right=0.95,top=0.95)
for i,(g,c) in enumerate(zip(gpu_part_cnt,cpu_part_cnt)):
    ax = axes[i]
    ax.plot(xaxis,g,'o-',label='gpu',c=colorset[1],ms=MARKERSIZE,linewidth=LINEWIDTH)
    ax.plot(xaxis,c,'*--',label='cpu',c=colorset[0],ms=MARKERSIZE,linewidth=LINEWIDTH)
    ax.set_ylim([300,600])
    ax.set_ylabel('Particle count',fontsize=FONTSIZE)
    ax.annotate(text[i],(0.37,350), fontsize=FONTSIZE*1.4)
    ax.xaxis.set_tick_params(labelsize=FONTSIZE)
    ax.yaxis.set_tick_params(labelsize=FONTSIZE)
    if (i==0):
        ax.legend(loc='lower left',fontsize=FONTSIZE)
    if (i==2):
        ax.set_xlabel(r'$y/h$',fontsize=FONTSIZE)

fig0.savefig('Verf_stat_contact.eps')


pl.ion()
pl.show()
