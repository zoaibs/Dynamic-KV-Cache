"""All quantitative plots derive only from saved real benchmark observations."""
import os
from scripts.resources import ROOT
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.runtime/matplotlib'))
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

NAMES=['Native BF16','Uniform 8','Uniform 4','Dynamic']
COLORS=['#607187','#4C91B8','#6BB0A0','#CD864F']

def save(fig,name):
    fig.savefig(ROOT/'results/figures'/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(ROOT/'results/figures'/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)

def main():
    data=json.loads((ROOT/'results/metrics.json').read_text())
    rows=data['metrics']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','figure.facecolor':'white'})
    x=np.arange(4)
    fig,ax=plt.subplots(figsize=(7,3.8))
    payload=np.array([r['cpu_payload_bytes_mean']/2**20 for r in rows])
    metadata=np.array([r['cpu_metadata_bytes_mean']/2**20 for r in rows])
    ax.bar(x,payload,color=COLORS,label='KV payload')
    ax.bar(x,metadata,bottom=payload,color='#C7CDD3',label='Quantization + mapping metadata')
    ax.set(xticks=x,xticklabels=NAMES,ylabel='CPU cache storage (MiB)',
           title='Persistent temporal KV cache, before QA')
    ax.legend(fontsize=8);save(fig,'cache_bytes')
    fig,ax=plt.subplots(figsize=(7,3.8))
    ax.bar(x,[r['frames_per_second'] for r in rows],color=COLORS,
           yerr=[r['frames_per_second_repeat_std'] for r in rows],capsize=4)
    ax.set(xticks=x,xticklabels=NAMES,ylabel='Sampled video frames / second',
           title='CPU end-to-end throughput (decode through answer)')
    save(fig,'throughput')
    fig,ax=plt.subplots(figsize=(7,3.8))
    acc=np.array([r['accuracy_percent'] for r in rows])
    errors=np.array([[r['accuracy_wilson95_low'] for r in rows],
                     [r['accuracy_wilson95_high'] for r in rows]])
    ax.bar(x,acc,color=COLORS,yerr=np.stack((acc-errors[0],errors[1]-acc)),capsize=4)
    ax.set(xticks=x,xticklabels=NAMES,ylabel='Exact multiple-choice accuracy (%)',ylim=(0,100),
           title='8 unique MVBench action-antonym videos; 95% Wilson intervals')
    save(fig,'task_accuracy')
    fig,ax=plt.subplots(figsize=(7,3.8))
    for i,r in enumerate(rows):
        ax.scatter(r['cpu_cache_bytes_mean']/2**20,r['accuracy_percent'],s=100,color=COLORS[i],label=NAMES[i])
    ax.set(xlabel='Persistent CPU cache storage (MiB)',ylabel='Accuracy (%)',ylim=(0,100),
           title='Memory versus accuracy on the same 8 videos')
    ax.legend(fontsize=8);save(fig,'memory_accuracy')
    fig,ax=plt.subplots(figsize=(7,3.2))
    ax.set(xlim=(-.5,3.5),ylim=(0,1),xticks=x,xticklabels=NAMES,
           ylabel='GPU allocated / reserved bytes',title='GPU VRAM: not measured')
    ax.text(1.5,.55,'Unavailable: CPU-only runtime\nNo zero-valued GPU measurements are implied',
            ha='center',va='center',bbox=dict(facecolor='#F2F4F6',edgecolor='#C7CDD3'))
    ax.set_yticks([]);save(fig,'gpu_vram_unavailable')
    fig,ax=plt.subplots(figsize=(7,3.8))
    ax.bar(x,[r['process_peak_rss_bytes_mean']/2**30 for r in rows],color=COLORS)
    ax.set(xticks=x,xticklabels=NAMES,ylabel='Mean request peak process RSS (GiB)',
           title='CPU process memory (10 ms sampling; includes loaded weights)')
    save(fig,'cpu_process_memory')
    timeline=list(csv.DictReader((ROOT/'results/timeline.csv').open()))
    fig,axes=plt.subplots(2,1,figsize=(9,5),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    ids=np.arange(len(timeline))
    scores=[float(t['score']) for t in timeline]
    axes[0].plot(ids,scores,marker='o',color='#4C91B8')
    axes[0].axhline(.30,ls='--',color='#888',lw=1,label='Initial thresholds')
    axes[0].axhline(.65,ls='--',color='#888',lw=1)
    axes[0].set(ylabel='Information score S',ylim=(0,1),title='Dynamic decisions on actual sampled frames')
    axes[1].step(ids,[int(t['bits']) for t in timeline],where='mid',color='#CD864F')
    axes[1].set(ylabel='Storage bits',yticks=[4,8,16],ylim=(2,18),xlabel='Frame observation (4 per video; state resets between videos)')
    for ax in axes:
        for i in range(4,len(timeline),4): ax.axvline(i-.5,color='#CCC',lw=.8)
    fig.tight_layout();save(fig,'precision_timeline')
    fig,ax=plt.subplots(figsize=(7,3.8))
    bottom=np.zeros(4)
    for key,label,color in [('score_ms_mean','OpenCV + scheduling','#4C91B8'),
                             ('pack_ms_mean','Packing','#CD864F'),
                             ('unpack_ms_mean','Unpacking','#6BB0A0')]:
        values=np.array([r[key] for r in rows])
        ax.bar(x,values,bottom=bottom,label=label,color=color);bottom+=values
    ax.set(xticks=x,xticklabels=NAMES,ylabel='Mean overhead per complete QA request (ms)',
           title='Measured scheduler and cache conversion overhead')
    ax.legend(fontsize=8);save(fig,'overhead')
    fig,axes=plt.subplots(1,2,figsize=(8,3.0))
    axes[0].bar(x,[r['cpu_cache_bytes_mean']/2**20 for r in rows],color=COLORS)
    axes[0].set(ylabel='Persistent CPU cache (MiB)',title='Cache storage')
    axes[1].bar(x,[r['frames_per_second'] for r in rows],color=COLORS,
                yerr=[r['frames_per_second_repeat_std'] for r in rows],capsize=3)
    axes[1].set(ylabel='Sampled frames / second',title='CPU end-to-end throughput')
    for ax in axes: ax.set_xticks(x,['BF16','8-bit','4-bit','Dynamic'],rotation=15)
    fig.tight_layout();save(fig,'paper_overview')
    fig,ax=plt.subplots(figsize=(8,1.8))
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(cx,cy,label,width=.19):
        ax.add_patch(FancyBboxPatch((cx-width/2,cy-.16),width,.32,
                     boxstyle='round,pad=.01',facecolor='#EEF2F5',edgecolor='#526B7B'))
        ax.text(cx,cy,label,ha='center',va='center',fontsize=9)
    def arrow(a,b):
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#526B7B',lw=1.2))
    box(.12,.30,'New video\nframe')
    box(.37,.30,'Pretrained\nvision + decoder')
    box(.63,.30,'Packed K/V\nCPU cache')
    box(.88,.30,'Reconstruct\nhistory for QA')
    box(.37,.80,'Canny, flow,\ndifference')
    box(.63,.80,'Score chooses\n4 / 8 / 16 bits')
    arrow((.22,.30),(.27,.30));arrow((.47,.30),(.53,.30));arrow((.73,.30),(.78,.30))
    arrow((.12,.46),(.27,.80));arrow((.47,.80),(.53,.80));arrow((.63,.64),(.63,.47))
    ax.annotate('',xy=(.37,.12),xytext=(.88,.12),arrowprops=dict(arrowstyle='->',
                connectionstyle='arc3,rad=-.15',color='#CD864F'))
    ax.text(.63,-.10,'Reused by subsequent decoder calls',ha='center',va='top',fontsize=8,color='#8A5730')
    save(fig,'architecture')

if __name__=='__main__':
    main()
