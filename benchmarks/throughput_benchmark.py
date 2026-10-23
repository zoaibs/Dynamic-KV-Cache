"""Print measured end-to-end request time/throughput from the same experiment."""
import json
from scripts.resources import ROOT

def main():
    for line in (ROOT/'results/raw/benchmark.jsonl').read_text().splitlines():
        r=json.loads(line)
        print(json.dumps({k:r[k] for k in ('mode','sample_id','repeat','latency_seconds',
              'frames_per_second','qa_per_second','score_seconds','pack_seconds','unpack_seconds')}))

if __name__=='__main__':
    main()
