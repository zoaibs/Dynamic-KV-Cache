"""Cross-check the delivered GPU experiments, physical bytes and review evidence."""
import json,math,csv,statistics as st
from collections import Counter
from datetime import datetime
from pathlib import Path
from scripts.resources import ROOT,CAP,workspace_bytes,sha256
from src.policies import Policy

def read(path):return json.loads((ROOT/path).read_text())
def jsonl(path):return [json.loads(x) for x in (ROOT/path).read_text().splitlines()]

def main():
    selected=read('configs/gpu_selected.json');config_hash=sha256(ROOT/'configs/gpu_selected.json')
    assert selected['frozen_before_test']
    assert selected['policy']==read('results/scheduler_validation.json')['selected']['policy']
    assert sha256(ROOT/'data/validation_v3.json')==selected['validation_manifest_sha256']
    assert sha256(ROOT/'data/test_v3.json')==selected['test_manifest_sha256']
    validation=read('data/validation_v3.json')['samples'];heldout=read('data/test_v3.json')['samples']
    legacy=read('data/manifest.json')['samples']
    assert len(validation)==36 and len(heldout)==72
    for key in ['path','sha256']:
        assert not ({s[key] for s in validation}&{s[key] for s in heldout})
        assert not ({s[key] for s in validation+heldout}&{s[key] for s in legacy})
    for s in validation+heldout:assert sha256(ROOT/s['path'])==s['sha256']
    quality=jsonl('results/raw/gpu_test_quality.jsonl');timing=jsonl('results/raw/gpu_timing.jsonl')
    modes=['native','fused16','4','8','dynamic','original','random']
    assert len(quality)==504 and len({(r['sample_id'],r['mode']) for r in quality})==504
    assert len(timing)==180 and len({(r['sample_id'],r['mode'],r['repeat'],r['requested_frames']) for r in timing})==180
    for phase in ['quality','timing']:
        config=read(f'results/raw/gpu_{phase}_config.json')
        assert config['selected_sha256']==config_hash and config['selected']==selected
        status=read(f'results/raw/gpu_{"test_quality" if phase=="quality" else "timing"}.status.json')
        assert status['status']=='success'
        frozen=datetime.fromisoformat(selected['policy_frozen_at_utc']).timestamp()
        assert frozen<=(ROOT/f'results/raw/gpu_{phase}_config.json').stat().st_mtime
    ids={s['id'] for s in heldout}
    for mode in modes:assert {r['sample_id'] for r in quality if r['mode']==mode}==ids
    by_native={r['sample_id']:r for r in quality if r['mode']=='native'}
    assert all(1<=r['actual_frames']<=selected['quality_frames'] for r in quality)
    assert all(r['actual_frames']==by_native[r['sample_id']]['actual_frames'] for r in quality)
    for frames in [4,16,32,64]:
        for mode in modes[:5]:
            rs=[r for r in timing if r['mode']==mode and r['requested_frames']==frames]
            assert len(rs)==9 and {r['repeat'] for r in rs}=={0,1,2}
            assert all(r['actual_frames']==frames and r['latency_seconds']>0 for r in rs)
    by_id={s['id']:s for s in heldout}
    native={r['sample_id']:r for r in quality if r['mode']=='native'}
    validation_bits={r['sample_id']:r['bits'] for r in jsonl('results/raw/scheduler_validation.jsonl')
                     if r['policy']==selected['policy']['name']}
    count=Counter()
    for r in quality+timing:
        a=r['cache'];used=a['used_tokens'];frames=r['frame_map']
        assert a['cpu_cache_payload_bytes']==0
        assert a['gpu_cache_used_bytes']==a['payload_bytes']+a['quantization_metadata_bytes']+a['token_mapping_bytes']
        assert a['gpu_cache_allocated_bytes']>=a['gpu_cache_used_bytes']>0
        assert used==frames[-1]['end']<=8192
        assert all(f['visual_tokens']==64 for f in frames)
        assert all(x['end']==y['start'] for x,y in zip(frames,frames[1:]))
        assert len(frames)==len(r['bits'])==r['actual_frames']
        assert len({f['source_index'] for f in frames})==len(frames)
        if r['mode']=='native':assert a['gpu_cache_used_bytes']==32*2*5*64*2*used
        else:
            tokens={4:0,8:0,16:frames[0]['start']}
            for f in frames:tokens[f['bits']]+=f['end']-f['start']
            assert {int(k):v for k,v in a['precision_token_counts'].items()}==tokens
            assert a['payload_bytes']==32*2*5*64*sum(tokens[b]*b/8 for b in tokens)
            assert a['quantization_metadata_bytes']==32*2*5*8*(tokens[4]+tokens[8])
            assert a['token_mapping_bytes']==used*5
            caps={int(k):v for k,v in a['precision_pool_capacities'].items()}
            assert a['gpu_cache_allocated_bytes']==32*2*5*(40*caps[4]+72*caps[8]+128*caps[16])+8192*5
        if r['mode'] in ('dynamic','original'):
            policy=Policy(**selected['policy']) if r['mode']=='dynamic' else Policy('original')
            bits,scores=policy.assign(r['raw_features'])
            assert bits==r['bits']
            assert all(abs(x-y)<1e-10 for x,y in zip(scores,r['information_scores']))
        if 'answers' in r:
            s=by_id[r['sample_id']];answers=r['answers']
            assert len(answers)==len(s['candidates'])
            assert r['no_video']==native[r['sample_id']]['no_video']
            correct=0
            for shift,answer in enumerate(answers):
                cs=s['candidates'][shift:]+s['candidates'][:shift]
                assert all(math.isfinite(x) for x in answer['scores'])
                assert answer['prediction']==max(range(len(cs)),key=lambda i:answer['scores'][i])
                assert answer['predicted_text']==cs[answer['prediction']]
                assert answer['correct']==(answer['predicted_text']==s['answer'])
                correct+=answer['correct']
            assert r['video_mean_accuracy']==correct/len(answers)
            if r['mode']=='dynamic':count.update(r['bits'])
        else:
            if r['mode']=='dynamic':
                assert r['features_overlap_vision'] is True
                if r['requested_frames']==16:assert r['bits']==validation_bits[r['sample_id']]
            for key in ['gpu_peak_allocated_bytes','gpu_peak_reserved_bytes','gpu_decoder_peak_allocated_bytes']:
                assert r[key]>0
    assert len(count)>=2
    metrics=read('results/gpu_metrics.json')
    assert len(metrics['quality'])==7 and len(metrics['timing'])==20
    for summary in metrics['quality']:
        rs=[r for r in quality if r['mode']==summary['mode']]
        assert abs(summary['accuracy_percent']-100*st.mean(r['video_mean_accuracy'] for r in rs))<1e-10
        assert summary['gpu_cache_used_bytes_mean']==st.mean(r['cache']['gpu_cache_used_bytes'] for r in rs)
        assert summary['gpu_cache_allocated_bytes_mean']==st.mean(r['cache']['gpu_cache_allocated_bytes'] for r in rs)
    for summary in metrics['timing']:
        rs=[r for r in timing if r['mode']==summary['mode'] and r['requested_frames']==summary['requested_frames']]
        fps=[sum(r['actual_frames'] for r in rs if r['repeat']==rep)/
             sum(r['latency_seconds'] for r in rs if r['repeat']==rep) for rep in range(3)]
        assert abs(summary['latency_seconds']-st.mean(r['latency_seconds'] for r in rs))<1e-10
        assert abs(summary['frames_per_second']-st.mean(fps))<1e-10
    for name,key in [('gpu_quality.csv','quality'),('gpu_timing.csv','timing')]:
        with (ROOT/'results'/name).open(newline='') as f:rows=list(csv.DictReader(f))
        assert len(rows)==len(metrics[key])
        for row,summary in zip(rows,metrics[key]):
            assert row['mode']==summary['mode']
            field='accuracy_percent' if key=='quality' else 'frames_per_second'
            assert float(row[field])==summary[field]
    assert read('results/raw/gpu_tests_final.status.json')['status']=='success'
    assert '40 passed' in (ROOT/'results/raw/gpu_tests_final.log').read_text()
    assert read('results/raw/gpu_kernel_correctness.json')['status']=='passed'
    assert read('results/raw/gpu_cli_smoke.status.json')['status']=='success'
    assert '"cpu_cache_payload_bytes": 0' in (ROOT/'results/raw/gpu_cli_smoke.log').read_text()
    pdf_checks=read('results/pdf_checks.json')
    assert pdf_checks['visual_review'].startswith('passed')
    assert pdf_checks['pdf_sha256']==sha256(ROOT/'paper/paper.pdf')
    import pymupdf
    pdf=pymupdf.open(ROOT/'paper/paper.pdf')
    assert 4<=len(pdf)<=6 and len(pdf)==pdf_checks['pages']
    assert all(len(p.get_text().strip())>100 for p in pdf)
    required=['README.md','FINAL_SUMMARY.md','requirements_gpu.txt','paper/paper.md','paper/paper.pdf',
              'results/gpu_metrics.json','results/gpu_quality.csv','results/gpu_timing.csv',
              'results/figures/gpu_summary.png','results/figures/gpu_quality.png',
              'results/figures/gpu_memory_accuracy.png','results/figures/gpu_timeline.png',
              'archives/cpu_v1.zip','results/profiles/summary.json']
    assert all((ROOT/p).is_file() and (ROOT/p).stat().st_size for p in required)
    profile=read('results/profiles/summary.json')
    assert profile['native']['aten_mm_count']>profile['dynamic']['aten_mm_count']>0
    assert profile['native']['aten_cat_count']>profile['dynamic']['aten_cat_count']>0
    provenance=read('results/gpu_provenance.json')
    assert provenance['selected_sha256']==config_hash
    assert sha256(ROOT/selected['model_path']/'model.safetensors')==provenance['checkpoint_sha256']
    assert all(sha256(ROOT/p)==h for p,h in provenance['source_sha256'].items())
    assert provenance['security_settings_changed'] is False
    size=workspace_bytes();assert size<CAP
    report=dict(status='passed',unit_tests=40,quality_cache_streams=504,quality_videos=72,
        quality_rotated_questions_per_mode=240,timed_requests=180,
        quality_actual_frames_min=min(r['actual_frames'] for r in quality),
        quality_actual_frames_max=max(r['actual_frames'] for r in quality),
        real_gpu_pack_and_causal_attention='passed',selected_config_sha256=config_hash,
        split_hashes_disjoint=True,per_frame_policy_decisions='passed',physical_byte_accounting='passed',
        policy_frozen_before_test=True,checkpoint_sha256=provenance['checkpoint_sha256'],
        inference_cli='passed',pdf_visual_review='passed',paper_pages=len(pdf),
        dynamic_heldout_frame_precision_counts=dict(count),workspace_bytes=size,cap_bytes=CAP)
    (ROOT/'results/gpu_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
