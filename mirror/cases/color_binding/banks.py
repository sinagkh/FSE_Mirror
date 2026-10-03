"""Score-blind source selection and intervention validation; never loads a scorer."""
from collections import Counter; from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image; from PIL import ImageDraw
from pycocotools import mask as mask_utils
from mirror.core.io import ROOT; from mirror.core.io import CODE; from mirror.core.io import read; from mirror.core.io import sha; from mirror.core.io import dump; from mirror.core.io import jsonl; from mirror.core.io import code_hashes
from mirror.cases.color_binding.generators import hsv_recolor; from mirror.cases.color_binding.generators import tint; from mirror.cases.color_binding.generators import context; from mirror.cases.color_binding.generators import edit_checks; from mirror.cases.color_binding.generators import pixel_hash; from mirror.cases.color_binding.generators import render_background; from mirror.cases.color_binding.generators import render_routing; from mirror.cases.color_binding.generators import COLORS

COCO=ROOT/"clip/data/coco"
ANN=COCO/"annotations/instances_val2017.json"
BG_NOUNS=["car","boat","bus","truck","train","bicycle","cat","dog","horse","motorcycle","chair","airplane"]
TRAIN_PAIRS=[("car","boat"),("bus","airplane"),("person","bicycle"),("cat","couch"),("dog","sports ball")]
HELD_PAIRS=[("bicycle","bus"),("car","traffic light"),("cat","chair"),("person","dog"),("tv","remote")]
TRAIN_NOUNS=set(itertools.chain.from_iterable(TRAIN_PAIRS))
ROUTING_NOUNS=TRAIN_NOUNS | set(itertools.chain.from_iterable(HELD_PAIRS))
LIMITS={"development":dict(background=3,routing=24),"calibration":dict(background=10,routing=100),
        "pilot":dict(background=15,routing=200),"reserve":dict(background=15,routing=200)}


def order(*values):
    return hashlib.sha256(json.dumps(["interbind_v1_20260922",*values],sort_keys=True).encode()).hexdigest()


def partition(image_id):
    bucket=int(order("partition",int(image_id)),16)%100
    if bucket<10:return "development"
    if bucket<25:return "calibration"
    if bucket<50:return "pilot"
    if bucket<80:return "reserve"
    return "donor"


def decode(ann,h,w):
    seg=ann["segmentation"]
    if isinstance(seg,list): rle=mask_utils.merge(mask_utils.frPyObjects(seg,h,w))
    elif isinstance(seg.get("counts"),list):rle=mask_utils.frPyObjects(seg,h,w)
    else:rle=seg
    return mask_utils.decode(rle).astype(bool)


@lru_cache(maxsize=1)
def annotations():
    raw=read(ANN)
    images={r["id"]:r for r in raw["images"]}
    anns={r["id"]:r for r in raw["annotations"]}
    cats={r["id"]:r["name"] for r in raw["categories"]}
    byimage=defaultdict(list)
    for a in anns.values():byimage[a["image_id"]].append(a)
    return images,anns,cats,byimage


def eligibility(ann,im):
    h,w=im["height"],im["width"]
    if ann.get("iscrowd",0):return None,"crowd"
    fraction=ann["area"]/(h*w)
    if not .02<=fraction<=.60:return None,"area"
    x,y,bw,bh=ann["bbox"]
    if min(bw,bh)<24:return None,"small_box"
    if x<1 or y<1 or x+bw>w-1 or y+bh>h-1:return None,"border_contact"
    mask=decode(ann,h,w)
    if not mask.any():return None,"empty_mask"
    # Center-crop visibility corresponding to CLIP's square evaluation crop.
    side=min(h,w);x0=(w-side)//2;y0=(h-side)//2
    retained=float(mask[y0:y0+side,x0:x0+side].sum()/mask.sum())
    if retained<.90:return None,"center_crop_loss"
    return dict(area_fraction=fraction,center_crop_mask_retention=retained),None


def _source(im,a,noun,checks):
    path=COCO/"val2017"/im["file_name"]
    return dict(image_id=im["id"],ann_id=a["id"],noun=noun,file=im["file_name"],
                coco_split="val2017",width=im["width"],height=im["height"],
                sha256=sha(path),eligibility=checks)


def prepare(out,part,workers):
    if (out/"banks/freeze.json").exists():raise FileExistsError("Banks already frozen")
    images,anns,cats,byimage=annotations()
    rules=dict(namespace="interbind_v1_20260922",partition_percent=dict(development=10,calibration=15,pilot=25,reserve=30,donor=20),
        quotas=LIMITS,background_nouns=BG_NOUNS,routing_nouns=sorted(ROUTING_NOUNS),
        train_routing_nouns=sorted(TRAIN_NOUNS),train_pairs=TRAIN_PAIRS,heldout_pairs=HELD_PAIRS,
        eligibility=dict(area=[.02,.60],min_bbox_side=24,no_crowd=True,min_border_distance=1,min_center_crop_retention=.90,
            routing_mask_intersection=0,donor_policy="No annotated instance of either target class, not certified absence"),
        selection="SHA256 ordering; largest eligible instance per class; one anchor per image per family; no scores",
        context_independence="Same context at both object colors; blue means fixed blue, never color-conditioned opposite",
        gate=dict(chromatic_fraction=.7,hue_p90_degrees=15,lab_delta_e=65,luminance_correlation=.8,hsv_value_correlation=.995,
            generator_deterministic_pass_rate=.95,judge_fidelity_pass_rate=.90),
        source_history="COCO val2017 excluded from these synthetic adapter training streams but used in past public-benchmark evaluations. Not historically untouched.",
        model_scores_loaded=False,annotations_sha256=sha(ANN),implementation_hashes=code_hashes())
    # The enumeration rules are persisted before opening an image or scoring any model.
    dump(out/"banks/construction_rules.json",rules)
    candidates=defaultdict(lambda:defaultdict(list));rejected=Counter();eligible_counts=Counter()
    for iid in sorted(images,key=lambda i:order("image",i)):
        group=partition(iid)
        if group=="donor":continue
        im=images[iid];best={}
        for a in sorted(byimage[iid],key=lambda a:(-a["area"],a["id"])):
            noun=cats[a["category_id"]]
            if noun not in set(BG_NOUNS)|ROUTING_NOUNS or noun in best:continue
            checked,reason=eligibility(a,im)
            if reason:rejected[reason]+=1;continue
            best[noun]=(a,checked);eligible_counts[f"{group}/{noun}"]+=1
        bg=sorted(set(best)&set(BG_NOUNS),key=lambda n:order("noun",iid,n))
        if bg:
            noun=bg[0];a,checked=best[noun]
            candidates[group]["background"].append(dict(family="background",partition=group,objects=[noun],
                sources=[_source(im,a,noun,checked)],source_ids=[iid]))
        pairs=list(itertools.combinations(sorted(set(best)&ROUTING_NOUNS),2))
        for n1,n2 in sorted(pairs,key=lambda pair:order("pair",iid,pair)):
            a,c1=best[n1];b,c2=best[n2]
            if np.any(decode(a,im["height"],im["width"]) & decode(b,im["height"],im["width"])):
                rejected["routing_overlap"]+=1;continue
            nseen=int(n1 in TRAIN_NOUNS)+int(n2 in TRAIN_NOUNS)
            candidates[group]["routing"].append(dict(family="routing",partition=group,objects=[n1,n2],
                sources=[_source(im,a,n1,c1),_source(im,b,n2,c2)],source_ids=[iid],
                seen_class_stratum={0:"neither_seen",1:"one_seen",2:"both_seen"}[nseen],
                pair_seen=any(set((n1,n2))==set(pair) for pair in TRAIN_PAIRS)))
            break
    donor_ids=sorted([i for i in images if partition(i)=="donor"],key=lambda i:order("donor",i))
    donor_used=set();files={};counts={}
    for group in LIMITS:
        for family in ["background","routing"]:
            kept=[];pernoun=Counter()
            for row in candidates[group][family]:
                if family=="background":
                    noun=row["objects"][0]
                    if pernoun[noun]>=LIMITS[group][family]:continue
                    donor=next((i for i in donor_ids if i not in donor_used and
                       not(set(row["objects"]) & {cats[a["category_id"]] for a in byimage[i]})),None)
                    if donor is None:raise RuntimeError("Insufficient distinct donor images; do not silently reuse")
                    donor_used.add(donor);im=images[donor]
                    row["donor"]=dict(image_id=donor,file=im["file_name"],sha256=sha(COCO/"val2017"/im["file_name"]),
                                      absence_evidence="no annotated instance; unannotated presence unknown")
                    pernoun[noun]+=1
                elif len(kept)>=LIMITS[group][family]:break
                row["anchor_id"]=f"ib1_{group}_{family}_{row['source_ids'][0]}"
                kept.append(row)
            path=out/f"banks/{group}_{family}.jsonl";jsonl(path,kept);files[str(path)]=sha(path)
            counts[f"{group}/{family}"]=dict(n=len(kept),objects=dict(Counter(":".join(r["objects"]) for r in kept)),
                strata=dict(Counter(r.get("seen_class_stratum","not_applicable") for r in kept)))
    dump(out/"banks/freeze.json",dict(rules_sha256=sha(out/"banks/construction_rules.json"),files=files,counts=counts,
         eligibility_counts=dict(eligible_counts),rejection_counts=dict(rejected),distinct_donors=len(donor_used),
         all_partitions_source_disjoint=True,no_model_scores_loaded=True))
    print("BANKS FROZEN",{k:v["n"] for k,v in counts.items()},flush=True)


def records(out,part,family):
    frozen=read(out/"banks/freeze.json")
    path=out/f"banks/{part}_{family}.jsonl"
    if sha(path)!=frozen["files"][str(path)]:raise ValueError("Bank hash mismatch")
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_row(row):
    iminfo,anns,_,_=annotations()
    s=row["sources"][0];path=COCO/s["coco_split"]/s["file"]
    if sha(path)!=s["sha256"]:raise ValueError("Source image changed")
    with Image.open(path) as im:rgb=np.array(im.convert("RGB"))
    masks=[decode(anns[s["ann_id"]],rgb.shape[0],rgb.shape[1]) for s in row["sources"]]
    donor=None
    if "donor" in row:
        d=row["donor"];path=COCO/"val2017"/d["file"]
        if sha(path)!=d["sha256"]:raise ValueError("Donor changed")
        with Image.open(path) as im:donor=np.array(im.convert("RGB"))
    return rgb,masks,donor


def _validate_row(row):
    cv2.setNumThreads(1)
    rgb,masks,donor=load_row(row);result=[]
    if row["family"]=="background":
        for name,img,base,mask,color,method in render_background(rgb,masks[0],donor):
            checks=edit_checks(base,img,mask,color,method)
            checks["context_preserves_foreground"]=bool(np.array_equal(base[mask],rgb[mask]))
            checks["passes"] &= checks["context_preserves_foreground"]
            result.append(dict(anchor_id=row["anchor_id"],family=row["family"],state=name,pixel_sha256=pixel_hash(img),**checks))
    else:
        for name,img,colors in render_routing(rgb,*masks):
            slotchecks=[]
            for i in range(2):
                intermediate=hsv_recolor(rgb,masks[1-i],colors[1-i])
                slotchecks.append(edit_checks(intermediate,img,masks[i],colors[i]))
            result.append(dict(anchor_id=row["anchor_id"],family=row["family"],state=name,pixel_sha256=pixel_hash(img),
                               passes=all(c["passes"] for c in slotchecks),slots=slotchecks))
    return result


def validate(out,part,workers):
    dest=out/f"validation/{part}"
    if dest.exists():raise FileExistsError(dest)
    rr=records(out,part,"background")+records(out,part,"routing")
    annotations()  # shared read-only annotation structures for fork workers
    result=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,values in enumerate(pool.map(_validate_row,rr,chunksize=2)):
            result+=values
            if (i+1)%20==0:print("VALIDATING",part,i+1,"/",len(rr),flush=True)
    jsonl(dest/"per_corner.jsonl",result)
    groups=defaultdict(list)
    for r in result:groups[f"{r['family']}/{'/'.join(r['state'].split('/')[:2])}"].append(r["passes"])
    rates={k:dict(n=len(v),passed=sum(v),rate=float(np.mean(v))) for k,v in groups.items()}
    dump(dest/"summary.json",dict(partition=part,n_anchors=len(rr),n_corners=len(result),rates=rates,
        semantic_judge_gate="pending: two independent model-family judges not yet supplied",
        phase_A_complete=False,new_model_scores_computed=False,code_hashes=code_hashes(),
        manifest_freeze_sha256=sha(out/"banks/freeze.json")))
    print("VALIDATION",part,rates,flush=True)


def natural_inventory(out,part,workers):
    path=ROOT/"clip/aro_bows/aro_data/visual_genome_attribution.json"
    raw=read(path);candidates=[];counts=Counter()
    for i,r in enumerate(raw):
        attrs=[str(x).lower().strip() for x in r.get("attributes",[])]
        if len(attrs)!=2:counts["unresolved_attributes"]+=1;continue
        known=[x in COLORS for x in attrs]
        if not any(known):counts["no_six_color_label"]+=1;continue
        counts["one_or_more_six_color_labels"]+=1
        if all(known) and attrs[0]!=attrs[1]:
            exact=set(attrs)=={"red","blue"}
            counts["two_distinct_known_colors"]+=1;counts["exact_red_blue"]+=int(exact)
            candidates.append(dict(official_index=i,image_id=r["image_id"],objects=[r["obj1_name"],r["obj2_name"]],
                attributes=attrs,true_caption=r["true_caption"],false_caption=r["false_caption"],
                image_path=r["image_path"],bbox=[r[f"bbox_{k}"] for k in ["x","y","w","h"]],exact_red_blue=exact,
                role="Existing labeled crop-level pair, not a verified full-image retrieval gallery or paired object-color intervention"))
    jsonl(out/"natural/aro_color_candidates.jsonl",candidates)
    dump(out/"natural/inventory.json",dict(source=str(path),sha256=sha(path),counts=dict(counts),
        n_images=len({r["image_id"] for r in candidates}),no_scores_loaded=True,
        missing_attribute_policy="unknown, not negative",unexplained_failure_policy="unresolved, not under-binding",
        gallery_status="not constructed: ARO union crops do not supply complete object-level gallery relevance labels",
        natural_calibration_status="pending explicit canonical-caption/label verification; no new thresholds chosen"))
    print("NATURAL LABEL INVENTORY",dict(counts),flush=True)


def _panel(images,labels,cell=240):
    canvas=Image.new("RGB",(cell*len(images),cell+34),"white");draw=ImageDraw.Draw(canvas)
    for i,(arr,label) in enumerate(zip(images,labels)):
        im=Image.fromarray(arr);im.thumbnail((cell,cell))
        canvas.paste(im,(i*cell+(cell-im.width)//2,(cell-im.height)//2))
        draw.text((i*cell+4,cell+5),label,fill="black")
    return canvas


def judge_packet(out,part,workers):
    dest=out/f"judge_packets/{part}"
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True)
    samples=[];answers=[]
    for family in ["background","routing"]:
        rr=sorted(records(out,part,family),key=lambda r:order("judge",r["anchor_id"]))[:12]
        for index,row in enumerate(rr):
            rgb,masks,donor=load_row(row)
            if family=="background":
                state="hsv/scene_swap/red";im=hsv_recolor(context(rgb,masks[0],"scene_swap",donor),masks[0],"red")
                before=context(rgb,masks[0],"scene_swap",donor)
                declared=f"Only the {row['objects'][0]} is recolored red; its shape and the scene must remain unchanged."
                extra=hsv_recolor(rgb,masks[0],"blue")
            else:
                before=rgb
                im=hsv_recolor(hsv_recolor(rgb,masks[0],"red"),masks[1],"blue")
                state="hsv/in_situ/red_blue"
                declared=f"Only colors change: {row['objects'][0]} to red and {row['objects'][1]} to blue."
                extra=hsv_recolor(hsv_recolor(rgb,masks[0],"blue"),masks[1],"red")
            itemid=order("judge-item",row["anchor_id"])[:16]
            filename=f"{itemid}.png";_panel([before,im],["Before","After"]).save(dest/filename)
            samples.append(dict(item_id=itemid,image=filename,declared_change=declared,questions=["recognizable_objects","perceived_colors","undeclared_change","naturalness_1_to_4"],
                source_anchor=row["anchor_id"],state=state))
            # Label only the planted programmatic fact, never a subjective naturalness score.
            if index%3==0:
                plantid=order("judge-control",row["anchor_id"])[:16];fn=f"{plantid}.png"
                if family=="background":extra=hsv_recolor(before,masks[0],"blue")
                _panel([before,extra],["Before","After"]).save(dest/fn)
                samples.append(dict(item_id=plantid,image=fn,declared_change=declared,questions=["recognizable_objects","perceived_colors","undeclared_change","naturalness_1_to_4"],source_anchor=row["anchor_id"],state=state))
                answers.append(dict(item_id=plantid,planted_property="rendered_hue_disagrees_with_declared_color",expected_color=row["objects"],
                    known_answer="wrong_declared_color",scope="Calibration of this planted error only; no ground-truth naturalness"))
    samples.sort(key=lambda r:order("judge-display",r["item_id"]))
    jsonl(dest/"items.jsonl",samples);jsonl(dest/"private_calibration_answers.jsonl",answers)
    public=[{k:v for k,v in r.items() if k not in {"source_anchor","state"}} for r in samples]
    jsonl(dest/"judge_input.jsonl",public)
    dump(dest/"instructions.json",dict(blinding="No scorer identity, scores, recipe, variant, or success labels. Answer each item independently.",
        response_fields=dict(item_id="string",recognizable_objects="yes/no/uncertain",perceived_colors="object -> color/uncertain",undeclared_change="yes/no/uncertain",naturalness_1_to_4="1..4, descriptive only",reason="short visual explanation"),
        status="pending two independent model-family judge responses",calibration_status="wrong-color controls only in starter packet; other planted checks still required",
        metadata_only=True,new_model_scores_computed=False,files={p.name:sha(p) for p in sorted(dest.glob("*.png"))}))
    print("BLINDED JUDGE PACKET",len(samples),"items,",len(answers),"planted wrong-color controls; no validation claims",flush=True)
