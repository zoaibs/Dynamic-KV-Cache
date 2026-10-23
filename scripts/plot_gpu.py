"""Publication figures from completed GPU measurements only."""
import os,json
from scripts.resources import ROOT
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.runtime/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

LABELS={'native':'Native BF16','fused16':'Engine BF16','4':'Uniform 4','8':'Uniform 8',
        'dynamic':'Tuned policy','original':'Original policy','random':'Random policy'}
COLORS={'native':'#61768A','fused16':'#969FAA','4':'#73A694','8':'#4E93B2',
        'dynamic':'#D3874F','original':'#A57F9E','random':'#BAAE7E'}

def save(fig,name):
    fig.savefig(ROOT/f'results/figures/{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(ROOT/f'results/figures/{name}.svg',bbox_inches='tight');plt.close(fig)

def main():
    d=json.loads((ROOT/'results/gpu_metrics.json').read_text())
    quality=d['quality'];timing=d['timing']
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    fig,ax=plt.subplots(figsize=(9,2));ax.set(xlim=(0,10),ylim=(0,2));ax.axis('off')
    boxes=[(.1,1.05,1.9,'CPU decode +\nframe features','#DBE6ED'),
           (2.55,1.05,1.9,'CUDA vision\nbatch up to 8','#D7E8EE'),
           (5.,1.05,2.,'Causal decoder\nchronological appends','#D8E9DF'),
           (7.7,1.05,2.,'Candidate QA\nshared video prefix','#EEDFCC'),
           (.1,.05,1.9,'Frozen policy\n4 / 8 / 16 bits','#EEDFCC'),
           (4.7,.05,2.6,'CUDA packed KV pools\nused + allocated accounting','#D8E9DF')]
    for x,y,w,label,color in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,.67,boxstyle='round,pad=.03',facecolor=color,edgecolor='#637681'))
        ax.text(x+w/2,y+.335,label,ha='center',va='center',fontsize=9)
    for a,b in [((2.05,1.38),(2.5,1.38)),((4.5,1.38),(4.95,1.38)),((7.05,1.38),(7.65,1.38)),
                ((1.05,1.02),(1.05,.75)),((2.05,.38),(4.65,.38))]:
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#526571','lw':1.3})
    ax.annotate('',xy=(6.,1.02),xytext=(6.,.75),arrowprops={'arrowstyle':'<->','color':'#526571','lw':1.3})
    ax.text(7.4,.35,'Unpack tiles inside attention',fontsize=8,color='#526571')
    save(fig,'gpu_architecture')
    fig,ax=plt.subplots(figsize=(8,4))
    x=np.arange(len(quality));acc=np.array([r['accuracy_percent'] for r in quality])
    yerr=np.array([acc-[r['accuracy_ci_low'] for r in quality],[r['accuracy_ci_high'] for r in quality]-acc])
    ax.bar(x,acc,color=[COLORS[r['mode']] for r in quality],yerr=yerr,capsize=3)
    ax.axhline(quality[0]['no_video_accuracy_percent'],color='#555',ls='--',lw=1,label='No-video control')
    ax.set(xticks=x,xticklabels=[LABELS[r['mode']] for r in quality],ylim=(0,100),ylabel='Held-out accuracy (%)',
           title='72 held-out videos; 95% task-stratified video bootstrap')
    ax.tick_params(axis='x',rotation=20);ax.legend(fontsize=8);save(fig,'gpu_quality')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for mode in ['native','fused16','4','8','dynamic']:
        rs=[r for r in timing if r['mode']==mode]
        xs=[r['requested_frames'] for r in rs]
        axes[0].errorbar(xs,[r['frames_per_second'] for r in rs],
             yerr=[r['frames_per_second_repeat_std'] for r in rs],marker='o',color=COLORS[mode],label=LABELS[mode])
        axes[1].plot(xs,[r['gpu_cache_allocated_bytes_mean']/2**20 for r in rs],marker='o',color=COLORS[mode],label=LABELS[mode])
    axes[0].set(xlabel='Frames per video',ylabel='End-to-end sampled frames/s',title='GPU throughput')
    axes[1].set(xlabel='Frames per video',ylabel='Allocated cache capacity (MiB)',title='Physical GPU cache buffers')
    axes[0].legend(fontsize=7);fig.tight_layout();save(fig,'gpu_summary')
    fig,ax=plt.subplots(figsize=(7,4))
    for r in quality:
        ax.errorbar(r['gpu_cache_allocated_bytes_mean']/2**20,r['accuracy_percent'],
            yerr=[[r['accuracy_percent']-r['accuracy_ci_low']],[r['accuracy_ci_high']-r['accuracy_percent']]],
            fmt='o',color=COLORS[r['mode']],label=LABELS[r['mode']],capsize=2)
    ax.set(xlabel='Allocated GPU cache capacity, up to 16 frames (MiB)',ylabel='Held-out accuracy (%)',
           title='Actual capacity versus quality');ax.legend(fontsize=8);save(fig,'gpu_memory_accuracy')
    long=[r for r in timing if r['requested_frames']==64]
    fig,ax=plt.subplots(figsize=(7,3.8));x=np.arange(len(long));bottom=np.zeros(len(long))
    for key,label,color in [('preparation_seconds','CPU preparation + feature wait','#A8BAC7'),
                            ('vision_seconds','Vision encoder','#6C96B0'),
                            ('decoder_seconds','Frame decoder/cache','#73A694'),
                            ('qa_seconds','Candidate QA','#D3874F')]:
        vals=np.array([r[key] for r in long]);ax.bar(x,vals,bottom=bottom,label=label,color=color);bottom+=vals
    ax.set(xticks=x,xticklabels=[LABELS[r['mode']] for r in long],ylabel='Seconds / complete request',title='64-frame latency components')
    ax.legend(fontsize=8);save(fig,'gpu_latency_components')
    fig,ax=plt.subplots(figsize=(7,3.8));w=.35
    ax.bar(x-w/2,[r['gpu_peak_allocated_bytes_mean']/2**30 for r in long],w,label='Whole request peak allocated')
    ax.bar(x+w/2,[r['gpu_decoder_peak_allocated_bytes_mean']/2**30 for r in long],w,label='Decoder + QA peak allocated')
    ax.set(xticks=x,xticklabels=[LABELS[r['mode']] for r in long],ylabel='PyTorch allocated VRAM (GiB)',title='GPU peak memory: scopes kept separate')
    ax.legend(fontsize=8);save(fig,'gpu_vram')
    validation=json.loads((ROOT/'results/scheduler_validation.json').read_text())['summary']
    fig,ax=plt.subplots(figsize=(10,4))
    ax.bar(np.arange(len(validation)),[100*r['accuracy'] for r in validation],color='#6C96B0')
    ax.set(xticks=np.arange(len(validation)),xticklabels=[r['policy']['name'] for r in validation],
           ylim=(0,100),ylabel='Validation accuracy (%)',title='Policy search on 36 validation videos; no test tuning')
    ax.tick_params(axis='x',rotation=45);save(fig,'gpu_policy_search')
    records=[json.loads(s) for s in (ROOT/'results/raw/gpu_test_quality.jsonl').read_text().splitlines()]
    r=next(r for r in records if r['mode']=='dynamic')
    fig,axes=plt.subplots(2,1,figsize=(8,3.7),sharex=True)
    t=np.arange(len(r['bits']))
    axes[0].plot(t,r['information_scores'],marker='o',color=COLORS['8']);axes[0].set(ylabel='Frozen policy score',ylim=(0,1),title=f'Held-out trace: {r["sample_id"]}')
    axes[1].step(t,r['bits'],where='mid',color=COLORS['dynamic']);axes[1].set(yticks=[4,8,16],ylim=(2,18),ylabel='Storage bits',xlabel='Sampled frame index')
    fig.tight_layout();save(fig,'gpu_timeline')

if __name__=='__main__':main()
