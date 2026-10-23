"""Final cross-artifact checks over real logs, storage, tests and report files."""
import json
from pathlib import Path
from scripts.resources import ROOT, workspace_bytes, CAP, sha256

def main():
    data=json.loads((ROOT/'results/metrics.json').read_text())
    records=[json.loads(s) for s in (ROOT/'results/raw/benchmark.jsonl').read_text().splitlines()]
    assert len(records)==96
    signatures={(r['mode'],r['sample_id'],r['repeat']) for r in records}
    assert len(signatures)==96
    samples={r['sample_id'] for r in records}
    assert len(samples)==8
    for mode in ['native','8','4','dynamic']:
        for repeat in range(3):
            assert {r['sample_id'] for r in records if r['mode']==mode and r['repeat']==repeat}==samples
    for r in records:
        assert r['gpu_peak_allocated_bytes'] is None and r['gpu_peak_reserved_bytes'] is None
        assert r['cpu_cache_bytes']==r['cpu_payload_bytes']+r['cpu_metadata_bytes']
        assert r['gpu_cache_bytes']==r['disk_cache_bytes']==0
        assert r['frames']==4 and r['latency_seconds']>0
        assert sum(x['visual_tokens'] for x in r['trace'])==256
        if r['mode']!='native':
            assert r['cpu_metadata_bytes']>0
    counts=data['dynamic_precision_counts']
    assert sum(counts.values())==32 and sum(v>0 for v in counts.values())>=2
    test_log=(ROOT/'results/raw/unit_tests_final.log').read_text()
    assert '25 passed' in test_log
    integration=json.loads((ROOT/'results/raw/integration.json').read_text())
    assert integration['native']['logits']==integration['16']['answer']['logits']
    assert {t['bits'] for t in integration['mixed_integration_test']['trace']}=={4,8,16}
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    for sample in manifest['samples']:
        assert sha256(ROOT/sample['path'])==sample['sha256']
    required=['README.md','FINAL_SUMMARY.md','requirements.txt','results/metrics.csv',
              'paper/paper.md','paper/paper.pdf','results/provenance.json',
              'results/figures/cache_bytes.png','results/figures/task_accuracy.png',
              'results/figures/throughput.png','results/figures/memory_accuracy.png',
              'results/figures/precision_timeline.png','results/figures/gpu_vram_unavailable.png']
    assert all((ROOT/p).is_file() and (ROOT/p).stat().st_size>0 for p in required)
    import pymupdf
    pdf=pymupdf.open(ROOT/'paper/paper.pdf')
    assert 4<=len(pdf)<=6, f'Paper page count: {len(pdf)}'
    assert all(len(page.get_text().strip())>100 for page in pdf)
    pdf_checks=json.loads((ROOT/'results/pdf_checks.json').read_text())
    assert pdf_checks['visual_review'].startswith('passed')
    smoke_status=json.loads((ROOT/'results/raw/inference_final_smoke.status.json').read_text())
    assert smoke_status['status']=='success'
    assert workspace_bytes()<CAP
    report=dict(status='passed',timed_requests=len(records),unique_videos=len(samples),
                unit_tests=25,real_model_integration='passed',paper_pages=len(pdf),
                dynamic_precision_counts=counts,workspace_bytes=workspace_bytes(),
                pdf_visual_review='passed',inference_cli='passed',
                cap_bytes=CAP,optional_methods_implemented=False,gpu_benchmark_performed=False)
    (ROOT/'results/final_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
