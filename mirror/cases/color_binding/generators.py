"""Independent deterministic rendering. No model or score dependencies."""
import hashlib
import cv2
import numpy as np

COLORS = dict(red=(220,30,30), blue=(30,90,220), green=(35,170,80),
              yellow=(235,205,35), purple=(135,70,190), orange=(235,130,35))


def pixel_hash(rgb):
    x=np.asarray(rgb,dtype=np.uint8)
    return hashlib.sha256(str(x.shape).encode()+x.tobytes()).hexdigest()


def hsv_recolor(rgb, mask, color):
    """Target hue + saturation floor, retaining native HSV value and outside pixels."""
    x=np.asarray(rgb,dtype=np.uint8)
    mask=np.asarray(mask,bool)
    hsv=cv2.cvtColor(x.astype(np.float32)/255.,cv2.COLOR_RGB2HSV)
    target=cv2.cvtColor(np.array([[COLORS[color]]],np.float32)/255.,cv2.COLOR_RGB2HSV)[0,0]
    hsv[mask,0]=target[0]
    hsv[mask,1]=np.maximum(hsv[mask,1],.70)
    new=np.clip(np.rint(cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)*255),0,255).astype(np.uint8)
    out=x.copy();out[mask]=new[mask]
    return out


def hsv_recolor_visible(rgb, mask, color, value_floor=.35):
    """Development candidate: preserve relative shading, not absolute HSV value.

    V' = value_floor + (1-value_floor)*V makes black surfaces visibly colored.
    This is a separate renderer, not a silent change to the native-value G1.
    The affine lift alters lightness as part of recoloring; report it explicitly.
    """
    if not 0 <= value_floor < 1:
        raise ValueError("value_floor must be in [0,1)")
    x=np.asarray(rgb,dtype=np.uint8);mask=np.asarray(mask,bool)
    hsv=cv2.cvtColor(x.astype(np.float32)/255.,cv2.COLOR_RGB2HSV)
    target=cv2.cvtColor(np.array([[COLORS[color]]],np.float32)/255.,cv2.COLOR_RGB2HSV)[0,0]
    hsv[mask,0]=target[0]
    hsv[mask,1]=np.maximum(hsv[mask,1],.70)
    hsv[mask,2]=value_floor+(1-value_floor)*hsv[mask,2]
    new=np.clip(np.rint(cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)*255),0,255).astype(np.uint8)
    out=x.copy();out[mask]=new[mask]
    return out


def luminance_recolor(rgb, mask, color, value_floor=.35):
    """Hue-exact luminance-affine recoloring, distinct from native-value HSV.

    HSV value is not luminance. On multicolored objects, preserving HSV value can
    change the luminance ordering. A fixed color ray scaled by source grayscale
    intensity preserves that ordering (up to 8-bit rounding). Absolute lightness
    and saturation change, as explicitly declared for this generator.
    """
    if not 0 <= value_floor < 1:raise ValueError("Invalid luminance floor")
    x=np.asarray(rgb,np.uint8);mask=np.asarray(mask,bool)
    luminance=cv2.cvtColor(x.astype(np.float32)/255.,cv2.COLOR_RGB2GRAY)
    ray=np.asarray(COLORS[color],np.float32);ray/=ray.max()
    value=value_floor+(1-value_floor)*luminance
    out=x.copy();out[mask]=np.clip(np.rint(value[mask,None]*ray*255),0,255).astype(np.uint8)
    return out


def tint(rgb,mask,color):
    out=np.asarray(rgb,np.uint8).copy()
    out[mask]=np.clip(.45*out[mask].astype(float)+.55*np.array(COLORS[color]),0,255).astype(np.uint8)
    return out


def context(rgb,mask,kind,donor=None):
    """Every context is independent of the object's red/blue state."""
    x=np.asarray(rgb,np.uint8)
    if kind=="original":return x.copy()
    if kind in ["gray","blue","green"]:
        c=(128,128,128) if kind=="gray" else COLORS[kind]
        out=np.full_like(x,c)
    elif kind=="scene_swap":
        if donor is None:raise ValueError("scene_swap requires the frozen donor")
        out=cv2.resize(np.asarray(donor,np.uint8),(x.shape[1],x.shape[0]),interpolation=cv2.INTER_LINEAR)
    elif kind=="hue_cast":
        hsv=cv2.cvtColor(x.astype(np.float32)/255.,cv2.COLOR_RGB2HSV)
        hsv[...,0]=(hsv[...,0]+60.)%360
        out=np.clip(np.rint(cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)*255),0,255).astype(np.uint8)
    else:raise ValueError(kind)
    out[mask]=x[mask]
    return out


def corr(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    sa,sb=np.std(a),np.std(b)
    if sa<1e-8 or sb<1e-8:return 1. if sa<1e-8 and sb<1e-8 else 0.
    return float(np.corrcoef(a,b)[0,1])


def edit_checks(original,rendered,mask,color,method="hsv"):
    x=np.asarray(original,np.uint8);y=np.asarray(rendered,np.uint8);mask=np.asarray(mask,bool)
    if not mask.any():raise ValueError("Empty edit region")
    hsv=cv2.cvtColor(y.astype(np.float32)/255.,cv2.COLOR_RGB2HSV)
    original_hsv=cv2.cvtColor(x.astype(np.float32)/255.,cv2.COLOR_RGB2HSV)
    target=cv2.cvtColor(np.array([[COLORS[color]]],np.float32)/255.,cv2.COLOR_RGB2HSV)[0,0]
    chromatic=mask & (hsv[...,1]>=.4) & (hsv[...,2]>=.08)
    fraction=float(chromatic.sum()/mask.sum())
    hue=np.abs((hsv[...,0][chromatic]-target[0]+180)%360-180)
    h90=float(np.quantile(hue,.9)) if len(hue) else None
    lab=cv2.cvtColor(y.astype(np.float32)/255.,cv2.COLOR_RGB2LAB)
    proto=cv2.cvtColor(np.array([[COLORS[color]]],np.float32)/255.,cv2.COLOR_RGB2LAB)[0,0]
    de=float(np.linalg.norm(lab[mask].mean(0)-proto))
    lx=cv2.cvtColor(x.astype(np.float32)/255.,cv2.COLOR_RGB2GRAY)
    ly=cv2.cvtColor(y.astype(np.float32)/255.,cv2.COLOR_RGB2GRAY)
    luminance=corr(lx[mask],ly[mask])
    value_corr=corr(original_hsv[...,2][mask],hsv[...,2][mask])
    outside=bool(np.array_equal(x[~mask],y[~mask]))
    checks=dict(outside_unchanged=outside,chromatic_fraction=fraction,hue_error_p90_degrees=h90,
                mean_lab_delta_e=de,luminance_correlation=luminance,hsv_value_correlation=value_corr)
    # Historical tint is measured, not silently declared equivalent to clean recoloring.
    checks["passes"]=bool(outside and fraction>=.7 and h90 is not None and h90<=15 and de<=65
                           and luminance>=.8 and (method!="hsv" or value_corr>=.995))
    return checks


def render_background(rgb,mask,donor=None):
    for method,fn in [("tint",tint),("hsv",hsv_recolor)]:
        for ctx in ["original","gray","blue","green","scene_swap","hue_cast"]:
            base=context(rgb,mask,ctx,donor)
            for color in ["red","blue"]:
                yield f"{method}/{ctx}/{color}",fn(base,mask,color),base,mask,color,method


def render_routing(rgb,mask1,mask2):
    if np.any(mask1 & mask2):raise ValueError("Routing masks overlap")
    union=mask1|mask2
    for c1 in ["red","blue"]:
        first=hsv_recolor(rgb,mask1,c1)
        for c2 in ["red","blue"]:
            out=hsv_recolor(first,mask2,c2)
            if not np.array_equal(out[~union],rgb[~union]):raise AssertionError("Out-of-mask edit")
            yield f"hsv/in_situ/{c1}_{c2}",out,(c1,c2)
