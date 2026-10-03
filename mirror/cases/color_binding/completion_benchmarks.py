"""Shared full-benchmark features for Phase-C bases; no subset/score selection."""
import argparse
import gc
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import log
from mirror.cases.color_binding.completion_data import OUT as ROOTOUT; from mirror.cases.color_binding.completion_data import configure
from mirror.cases.color_binding.routing_relative_pilot import SUGAR
from mirror.core.encoders import load_subject; from mirror.core.encoders import DEFAULT_REGISTRY

OUT=ROOTOUT/'benchmark_features'


def sugar(model):
    configure();dest=OUT/model/'sugarcrepe'
    if (dest/'complete.json').exists():return
    if torch.cuda.mem_get_info()[0]<8*1024**3:raise RuntimeError('GPU busy')
    idx=read(SUGAR/'indices.json');scorer=load_subject(model,device='cuda')
    names=idx['names'];prompts=idx['prompts'];images=[]
    # Identical official captions and the earlier benchmark's FP16 encoder precision.
    with torch.autocast('cuda',dtype=torch.float16):
        for start in range(0,len(names),64):
            ims=[Image.open(ROOT/'clip/data/coco/val2017'/s).convert('RGB') for s in names[start:start+64]]
            images.append(scorer.encode_images(ims,batch_size=64).numpy())
        texts=scorer.encode_texts(prompts,batch_size=256).numpy()
    dest.mkdir(parents=True,exist_ok=False)
    with (dest/'features.npz').open('xb') as f:np.savez_compressed(f,images=np.concatenate(images),texts=texts)
    dump(dest/'indices.json',idx)
    dump(dest/'complete.json',dict(model=model,files={str(p):sha(p) for p in dest.iterdir() if p.is_file()},
        official_reference_sha256=sha(SUGAR/'frozen_seed0.csv'),registry_sha256=sha(DEFAULT_REGISTRY),
        image_count=len(names),text_count=len(prompts),encoder_precision='FP16 autocast; unit FP32 features',no_scores=True))
    del scorer;gc.collect();torch.cuda.empty_cache();print('SUGAR_FEATURES_READY',model,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);a=ap.parse_args();log(OUT,'start',model=a.model)
    try:sugar(a.model)
    except BaseException as e:log(OUT,'failed',error=repr(e));raise
    log(OUT,'complete',model=a.model)


if __name__=='__main__':main()
