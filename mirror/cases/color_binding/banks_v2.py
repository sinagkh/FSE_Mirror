"""Score-blind visibility-aware source banks. V1 remains unchanged.

Only annotations and pixels enter selection. No torch, scorer, adapters or saved
model scores are imported. All policies are written before data enumeration.
"""
import argparse
from collections import Counter; from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from mirror.core.io import ROOT; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import log; from mirror.core.io import code_hashes
from mirror.cases.color_binding.banks import ANN; from mirror.cases.color_binding.banks import COCO; from mirror.cases.color_binding.banks import BG_NOUNS; from mirror.cases.color_binding.banks import TRAIN_NOUNS; from mirror.cases.color_binding.banks import TRAIN_PAIRS; from mirror.cases.color_binding.banks import annotations; from mirror.cases.color_binding.banks import decode; from mirror.cases.color_binding.banks import order; from mirror.cases.color_binding.banks import partition; from mirror.cases.color_binding.banks import _panel
from mirror.cases.color_binding.generators import hsv_recolor_visible; from mirror.cases.color_binding.generators import hsv_recolor; from mirror.cases.color_binding.generators import luminance_recolor; from mirror.cases.color_binding.generators import tint; from mirror.cases.color_binding.generators import edit_checks; from mirror.cases.color_binding.generators import pixel_hash; from mirror.cases.color_binding.generators import context

OUT=ROOT/"clip/interbind_banks_v2_20260922"
LIMITS={"development":(3,12),"calibration":(10,40),"pilot":(25,120),"reserve":(25,120)}


@lru_cache(maxsize=2)
def annotations_for(split):
    if split=="val2017":return annotations()
    raw=read(COCO/f"annotations/instances_{split}.json")
    images={r["id"]:r for r in raw["images"]};anns={r["id"]:r for r in raw["annotations"]}
    cats={r["id"]:r["name"] for r in raw["categories"]};byimage=defaultdict(list)
    for a in anns.values():byimage[a["image_id"]].append(a)
    return images,anns,cats,byimage


def square_crop(boxes,width,height):
    x0=min(b[0] for b in boxes);y0=min(b[1] for b in boxes)
    x1=max(b[0]+b[2] for b in boxes);y1=max(b[1]+b[3] for b in boxes)
    side=int(np.ceil(max(x1-x0,y1-y0)*1.25))
    if side>min(width,height):return None
    left=int(np.clip(round((x0+x1-side)/2),0,width-side))
    top=int(np.clip(round((y0+y1-side)/2),0,height-side))
    return [left,top,left+side,top+side]


def crop_array(image,box):
    x0,y0,x1,y1=box
    return image[y0:y1,x0:x1].copy()


def instance_quality(ann,im,rgb):
    h,w=im["height"],im["width"]
    if ann.get("iscrowd",0):return None,"crowd"
    x,y,bw,bh=ann["bbox"]
    if not .02<=ann["area"]/(h*w)<=.60:return None,"source_area"
    if min(bw,bh)<24:return None,"small_box"
    if min(x,y,w-x-bw,h-y-bh)<1:return None,"border_contact"
    mask=decode(ann,h,w)
    count,_,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    if count<2:return None,"empty_mask"
    connected=float(stats[1:,cv2.CC_STAT_AREA].max()/mask.sum())
    if connected<.75:return None,"fragmented_mask"
    core=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    if core.sum()<64:return None,"thin_mask"
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    focus=float(cv2.Laplacian(gray,cv2.CV_64F)[core].var())
    if focus<20:return None,"blurred_mask"
    return dict(mask=mask,connected_fraction=connected,laplacian_variance=focus),None


def crop_quality(mask,ann,crop):
    m=crop_array(mask,crop);side=crop[2]-crop[0]
    retained=float(m.sum()/mask.sum())
    fraction=float(m.mean())
    box_short_px=224*min(ann["bbox"][2:])/side
    return dict(retained=retained,area_fraction=fraction,min_bbox_at_224=box_short_px),bool(
        retained>=.99 and .02<=fraction<=.60 and box_short_px>=40)


def _enumerate_image(task):
    split,iid=task
    cv2.setNumThreads(1)
    images,_,cats,byimage=annotations_for(split);im=images[iid]
    group=partition(iid)
    if group=="donor":return [],{}
    path=COCO/split/im["file_name"]
    with Image.open(path) as f:rgb=np.array(f.convert("RGB"))
    rejected=Counter();byclass=defaultdict(list)
    for a in byimage[iid]:byclass[cats[a["category_id"]]].append(a)
    eligible={}
    for noun,aa in byclass.items():
        substantial=[a for a in aa if a["area"]/(im["width"]*im["height"])>=.005]
        if len(substantial)!=1:rejected["nonunique_class_instance"]+=1;continue
        # Largest instance is decided first; never fall back to a smaller instance.
        ann=max(aa,key=lambda a:(a["area"],-a["id"]))
        q,reason=instance_quality(ann,im,rgb)
        if reason:rejected[reason]+=1;continue
        eligible[noun]=(ann,q)
    source_sha=None
    def make_source(noun):
        nonlocal source_sha
        if source_sha is None:source_sha=sha(path)
        ann,q=eligible[noun]
        return dict(image_id=iid,ann_id=ann["id"],noun=noun,coco_split=split,file=im["file_name"],
            width=im["width"],height=im["height"],sha256=source_sha,
            visibility={k:v for k,v in q.items() if k!="mask"})
    result=[]
    for noun in sorted(set(BG_NOUNS)&set(eligible),key=lambda n:order("v2_bg",iid,n)):
        ann,q=eligible[noun];crop=square_crop([ann["bbox"]],im["width"],im["height"])
        if crop is None:rejected["no_square_crop"]+=1;continue
        qc,valid=crop_quality(q["mask"],ann,crop)
        if not valid:rejected["crop_visibility"]+=1;continue
        result.append(dict(anchor_id=f"ib2_{group}_background_{iid}",family="background",partition=group,
            objects=[noun],sources=[make_source(noun)],source_ids=[iid],crop_xyxy=crop,crop_quality=[qc]))
        break
    pairs=itertools.combinations(sorted(eligible),2)
    for pair in sorted(pairs,key=lambda p:order("v2_pair",iid,p)):
        aa=[eligible[n][0] for n in pair];qq=[eligible[n][1] for n in pair]
        if np.any(qq[0]["mask"]&qq[1]["mask"]):rejected["mask_overlap"]+=1;continue
        crop=square_crop([a["bbox"] for a in aa],im["width"],im["height"])
        if crop is None:rejected["no_square_crop"]+=1;continue
        checks=[crop_quality(q["mask"],a,crop) for a,q in zip(aa,qq)]
        if not all(c[1] for c in checks):rejected["crop_visibility"]+=1;continue
        seen=sum(n in TRAIN_NOUNS for n in pair)
        result.append(dict(anchor_id=f"ib2_{group}_routing_{iid}",family="routing",partition=group,
            objects=list(pair),sources=[make_source(n) for n in pair],source_ids=[iid],crop_xyxy=crop,
            crop_quality=[c[0] for c in checks],seen_class_stratum={0:"neither_seen",1:"one_seen",2:"both_seen"}[seen],
            pair_seen=any(set(pair)==set(p) for p in TRAIN_PAIRS)))
        break
    return result,dict(rejected)


def prepare(out,workers,split="val2017"):
    images,_,cats,byimage=annotations_for(split)
    exclusion_file=ROOT/"clip/interbind_source_exclusions_20260922/exclusions.json"
    excluded=set(read(exclusion_file)["source_ids"]) if split=="train2017" else set()
    dump(out/"protocol.json",dict(version="visibility-aware-v2",source_split=split,source_annotations_sha256=sha(COCO/f"annotations/instances_{split}.json"),
        exclusions=dict(path=str(exclusion_file),sha256=sha(exclusion_file),n=len(excluded)) if excluded else None,
        model_scores_loaded=False,source_partition="Unchanged banks.partition(COCO image_id); v1 partitions retained",
        background_nouns=BG_NOUNS,routing_nouns=sorted(cats.values()),train_routing_nouns=sorted(TRAIN_NOUNS),
        routing_scope="All COCO classes, reported by seen-class strata; expansion beyond original nouns is a declared transfer test",
        source_rules=dict(area=[.02,.60],min_bbox=24,min_border=1,substantial_same_class_count=1,
            substantial_area=.005,largest_instance_only=True,min_largest_component=.75,min_core_pixels=64,min_laplacian_variance=20),
        crop_rules=dict(square_union_expansion=1.25,require_fit_without_padding=True,min_retention=.99,
            mask_area=[.02,.60],min_object_bbox_at_224=40),
        limits=LIMITS,quota_units="background per noun, routing per seen-class stratum; at most one anchor/image/family",
        selection="Fixed hash order before outcomes; no model-based selection or filtering",
        value_candidate=dict(formula="V'=.35+.65V",saturation_floor=.70,absolute_lightness_changes=True),
        source_history="New edited outcomes have not been scored. Val2017 has earlier public-benchmark exposure; train2017 excludes known historical manifests but foundation pretraining membership is unknown.",
        code_hashes=code_hashes()))
    rows=[];rejected=Counter()
    ids=sorted(set(images)-excluded,key=lambda i:order("image",i))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index,(rr,bad) in enumerate(pool.map(_enumerate_image,[(split,i) for i in ids],chunksize=10)):
            rows+=rr;rejected.update(bad)
            if (index+1)%500==0:print("V2 ENUMERATE",index+1,"/",len(ids),flush=True)
    donorids=sorted([i for i in images if i not in excluded and partition(i)=="donor"],key=lambda i:order("v2_donor",i))
    donor_classes={i:{cats[a["category_id"]] for a in byimage[i]} for i in donorids}
    used=set();counts={};files={}
    for group,(nb,nr) in LIMITS.items():
        for family in ["background","routing"]:
            seen=Counter();kept=[]
            for row in rows:
                if row["partition"]!=group or row["family"]!=family:continue
                key=row["objects"][0] if family=="background" else row["seen_class_stratum"]
                if seen[key]>=(nb if family=="background" else nr):continue
                if family=="background":
                    donor=next((i for i in donorids if i not in used and not set(row["objects"])&donor_classes[i]),None)
                    if donor is None:raise RuntimeError("Insufficient distinct donors")
                    used.add(donor);im=images[donor]
                    row["donor"]=dict(image_id=donor,file=im["file_name"],coco_split=split,sha256=sha(COCO/split/im["file_name"]),
                        absence_evidence="No annotated class instance; unannotated presence remains unknown")
                seen[key]+=1;kept.append(row)
            path=out/f"banks/{group}_{family}.jsonl";jsonl(path,kept);files[str(path)]=sha(path)
            counts[f"{group}/{family}"]=dict(n=len(kept),strata=dict(seen))
    dump(out/"freeze.json",dict(protocol_sha256=sha(out/"protocol.json"),files=files,counts=counts,
        rejected=dict(rejected),distinct_donors=len(used),all_sources_disjoint_between_partitions=True,no_model_scores=True))
    print("V2 FROZEN",counts,flush=True)


def records(out,part,family):
    meta=read(out/"freeze.json");path=out/f"banks/{part}_{family}.jsonl"
    if sha(path)!=meta["files"][str(path)]:raise ValueError("Bank changed")
    return [json.loads(s) for s in path.read_text().splitlines()]


def load(row):
    source=row["sources"][0];split=source["coco_split"];_,anns,_,_=annotations_for(split);p=COCO/split/source["file"]
    if sha(p)!=source["sha256"]:raise ValueError("Source changed")
    with Image.open(p) as im:rgb=np.array(im.convert("RGB"))
    masks=[crop_array(decode(anns[s["ann_id"]],rgb.shape[0],rgb.shape[1]),row["crop_xyxy"]) for s in row["sources"]]
    rgb=crop_array(rgb,row["crop_xyxy"]);donor=None
    if "donor" in row:
        p=COCO/row["donor"].get("coco_split","val2017")/row["donor"]["file"]
        if sha(p)!=row["donor"]["sha256"]:raise ValueError("Donor changed")
        with Image.open(p) as im:donor=np.array(im.convert("RGB"))
    return rgb,masks,donor


def _validate(row):
    cv2.setNumThreads(1);rgb,masks,donor=load(row);result=[]
    for generator,fn in [("tint",tint),("native_hsv",hsv_recolor),("affine_hsv",hsv_recolor_visible)]:
        # All six colors checked per object independently before outcome scoring.
        for slot,mask in enumerate(masks):
            for color in ["red","blue","green","yellow","purple","orange"]:
                im=fn(rgb,mask,color);checks=edit_checks(rgb,im,mask,color,"tint" if generator=="tint" else "hsv")
                result.append(dict(anchor_id=row["anchor_id"],family=row["family"],generator=generator,
                    slot=slot,color=color,**checks,pixel_sha256=pixel_hash(im)))
    context_checks={}
    if row["family"]=="background":
        for kind in ["original","gray","blue","green","scene_swap","hue_cast"]:
            im=context(rgb,masks[0],kind,donor)
            context_checks[kind]=dict(object_pixels_identical=bool(np.array_equal(im[masks[0]],rgb[masks[0]])),pixel_sha256=pixel_hash(im))
    return result,dict(anchor_id=row["anchor_id"],context_checks=context_checks)


def _luminance_validate(row):
    cv2.setNumThreads(1);rgb,masks,donor=load(row);result=[]
    for slot,mask in enumerate(masks):
        for color in ["red","blue","green","yellow","purple","orange"]:
            im=luminance_recolor(rgb,mask,color)
            checks=edit_checks(rgb,im,mask,color,"luminance_affine")
            result.append(dict(anchor_id=row["anchor_id"],family=row["family"],generator="luminance_affine",
                slot=slot,color=color,**checks,pixel_sha256=pixel_hash(im)))
    return result


def validate_luminance(out,part,workers):
    if part!="development":raise ValueError("New candidate development only")
    dest=out/f"validation/{part}/luminance_candidate"
    dump(dest/"protocol.json",dict(generator="luminance_affine",formula="RGB'=target_RGB/max(target_RGB) * (.35+.65*source_grayscale) *255",
        rationale="HSV value is not luminance; a fixed color ray preserves source luminance order on multicolored objects",
        unchanged="No bank membership/threshold changes, no model scores",code_hashes=code_hashes()))
    rows=records(out,part,"background")+records(out,part,"routing")
    for s in {r["sources"][0]["coco_split"] for r in rows}:annotations_for(s)
    values=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rr in pool.map(_luminance_validate,rows,chunksize=2):values+=rr
    jsonl(dest/"per_edit.jsonl",values);groups=defaultdict(list)
    for r in values:groups[r["family"]].append(r["passes"])
    rates={k:dict(n=len(v),passed=sum(v),rate=float(np.mean(v))) for k,v in groups.items()}
    dump(dest/"summary.json",dict(rates=rates,no_scores=True,semantic_gate="pending"))
    print("LUMINANCE CHECK",rates,flush=True)


def validate(out,part,workers):
    if part=="reserve":raise ValueError("Reserve stays sealed until pilot is frozen")
    dest=out/f"validation/{part}"
    if dest.exists():raise FileExistsError(dest)
    rows=records(out,part,"background")+records(out,part,"routing");values=[];ctx=[]
    for s in {r["sources"][0]["coco_split"] for r in rows}:annotations_for(s)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for a,b in pool.map(_validate,rows,chunksize=2):values+=a;ctx.append(b)
    jsonl(dest/"per_edit.jsonl",values);jsonl(dest/"contexts.jsonl",ctx)
    groups=defaultdict(list)
    for r in values:groups[f"{r['family']}/{r['generator']}"].append(r["passes"])
    rates={k:dict(n=len(v),passed=sum(v),rate=float(np.mean(v))) for k,v in groups.items()}
    dump(dest/"summary.json",dict(rates=rates,n_anchors=len(rows),semantic_validity="pending blinded review",
        no_scores=True,raw_acceptance_before_filtering=True,code_hashes=code_hashes()))
    print("V2 VALIDATION",rates,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=["prepare","validate","validate-luminance"])
    p.add_argument("--source-split",choices=["train2017","val2017"],default="val2017")
    p.add_argument("--out",type=Path,default=OUT);p.add_argument("--partition",default="development",choices=list(LIMITS))
    p.add_argument("--workers",type=int,default=12);args=p.parse_args();log(args.out,"start")
    try:
        if args.command=="prepare":prepare(args.out,args.workers,args.source_split)
        elif args.command=="validate-luminance":validate_luminance(args.out,args.partition,args.workers)
        else:validate(args.out,args.partition,args.workers)
    except BaseException as e:log(args.out,"failed",error=repr(e));raise
    log(args.out,"complete")


if __name__=="__main__":main()
