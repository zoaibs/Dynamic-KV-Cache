"""Print measured cache memory rows from the completed controlled experiment."""
import json
from scripts.resources import ROOT

def main():
    for line in (ROOT/'results/raw/benchmark.jsonl').read_text().splitlines():
        r=json.loads(line)
        print(json.dumps({k:r[k] for k in ('mode','sample_id','repeat','cpu_payload_bytes',
              'cpu_metadata_bytes','cpu_cache_bytes','gpu_cache_bytes','disk_cache_bytes',
              'process_peak_rss_bytes','gpu_peak_allocated_bytes','gpu_peak_reserved_bytes')}))

if __name__=='__main__':
    main()
