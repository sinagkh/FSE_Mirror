"""Memory-bounded runtime for unchanged backdoor stages beside stopped jobs."""
import argparse
import json
import runpy
import sys
from pathlib import Path
import torch
from mirror.cases.backdoor.common import ROOT; from mirror.cases.backdoor.common import OUT; from mirror.cases.backdoor.common import command; from mirror.cases.backdoor.common import log; from mirror.cases.backdoor.common import sha

PAUSE = OUT/'priority_backdoor_20260927/pause_verified.json'
CAP = int(4.5*1024**3)
TARGETS = {'train':'par_extension_train.py','evaluate':'par_extension_evaluate.py'}


def guard():
    receipt = json.loads(PAUSE.read_text())
    assert receipt['verified']
    for entry in receipt['paused_processes']:
        path = Path('/proc')/str(entry['pid'])/'stat'
        if not path.exists():
            raise RuntimeError('Paused process disappeared; inspect saved state and queues before continuing: '+str(entry['pid']))
        stat = path.read_text(); parts=stat[stat.rfind(')')+2:].split()
        assert int(parts[19])==entry['start_ticks'], 'PID reused'
        assert parts[0]=='T', ('Optional job resumed; no competing stage launched',entry['pid'],parts[0])
    torch.set_num_threads(4)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(CAP/total)
    free,_=torch.cuda.mem_get_info()
    assert free > CAP+512*1024**2, ('Insufficient GPU memory while preserving stopped jobs',free,CAP)
    log('priority_backdoor_gpu_stage',free_bytes=free,torch_allocation_cap_bytes=CAP,
        runtime_sha256=sha(__file__),pause_receipt_sha256=sha(PAUSE))


if __name__=='__main__':
    command()
    parser=argparse.ArgumentParser();parser.add_argument('target',choices=TARGETS)
    args,rest=parser.parse_known_args()
    guard()
    script=ROOT/'mirror/cases/backdoor'/TARGETS[args.target]
    sys.argv=[str(script),*rest]
    try:
        runpy.run_path(str(script),run_name='__main__')
    finally:
        log('priority_backdoor_gpu_stage_end',target=args.target,args=rest,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
