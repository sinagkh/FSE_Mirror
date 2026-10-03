"""Deterministic audit and unseen-context/layout states for frozen source rows."""
import itertools
import numpy as np
from PIL import Image
from mirror.cases.color_binding.banks_v2 import load
from mirror.cases.color_binding.generators import luminance_recolor; from mirror.cases.color_binding.generators import context; from mirror.cases.color_binding.generators import pixel_hash

CONTEXTS=("gray","blue","green","hue_cast","scene_swap","original")
LAYOUTS=("canvas","swapped_canvas","in_situ")


def _rgba(rgb,mask,pad=4):
    ys,xs=np.where(mask)
    if not len(ys):raise ValueError("Empty object mask")
    x0=max(0,int(xs.min())-pad);x1=min(rgb.shape[1],int(xs.max())+1+pad)
    y0=max(0,int(ys.min())-pad);y1=min(rgb.shape[0],int(ys.max())+1+pad)
    arr=np.zeros((y1-y0,x1-x0,4),np.uint8)
    arr[...,:3]=rgb[y0:y1,x0:x1];arr[...,3]=mask[y0:y1,x0:x1]*255
    return Image.fromarray(arr,"RGBA")


def canvas_pair(rgb,masks,swapped=False):
    """Same 256px/112px/16px geometry as the legacy canvas renderer.

    Returned masks always belong to object slots 1,2, even after position swap.
    """
    canvas=Image.new("RGB",(256,256),(235,235,235));result=[None,None]
    objects=[_rgba(rgb,m) for m in masks]
    for position,slot in enumerate([1,0] if swapped else [0,1]):
        obj=objects[slot];scale=112/max(obj.size)
        obj=obj.resize((max(1,int(obj.width*scale)),max(1,int(obj.height*scale))),Image.Resampling.LANCZOS)
        x=16 if position==0 else 256-16-obj.width;y=(256-obj.height)//2
        canvas.paste(obj,(x,y),obj)
        m=np.zeros((256,256),bool);m[y:y+obj.height,x:x+obj.width]=np.asarray(obj.getchannel("A"))>0
        result[slot]=m
    if np.any(result[0]&result[1]):raise ValueError("Canvas slots overlap")
    return np.array(canvas),result


def render_anchor(row,colors=("red","blue")):
    if len(colors)!=2 or colors[0]==colors[1]:raise ValueError("Two distinct color states required")
    rgb,masks,donor=load(row);images=[];names=[]
    if row["family"]=="background":
        for ctx in CONTEXTS:
            base=context(rgb,masks[0],ctx,donor)
            if not np.array_equal(base[masks[0]],rgb[masks[0]]):raise AssertionError("Context changed foreground")
            for color in colors:
                im=luminance_recolor(base,masks[0],color)
                if not np.array_equal(im[~masks[0]],base[~masks[0]]):raise AssertionError("Color changed background")
                images.append(im);names.append(f"{ctx}/{color}")
    else:
        for layout in LAYOUTS:
            base,mm=(rgb,masks) if layout=="in_situ" else canvas_pair(rgb,masks,layout=="swapped_canvas")
            union=mm[0]|mm[1]
            for c1,c2 in itertools.product(colors,repeat=2):
                im=luminance_recolor(luminance_recolor(base,mm[0],c1),mm[1],c2)
                if not np.array_equal(im[~union],base[~union]):raise AssertionError("Routing changed undeclared pixels")
                images.append(im);names.append(f"{layout}/{c1}_{c2}")
    return images,names,[pixel_hash(im) for im in images]


def captions(family,objects,colors=("red","blue")):
    if family=="background":
        noun=objects[0]
        return [[f"a {c} {noun}",f"a photo of a {c} {noun}",f"the {noun} is {c}"] for c in colors]
    a,b=objects
    return [[f"a {c1} {a} and a {c2} {b}",f"a {c1} {a} next to a {c2} {b}",f"the {a} is {c1} and the {b} is {c2}"]
        for c1,c2 in itertools.product(colors,repeat=2)]
