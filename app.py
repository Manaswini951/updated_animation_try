import io
import json
import math
import os
import tempfile
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

st.set_page_config(page_title="Hand-Drawn Animal Walk Animator", page_icon="🦒", layout="wide")
st.title("🦒 Hand-Drawn Animal Walk Animator — Articulated v2")
st.caption("Gemini supplies anatomy. Python performs a joint-aware, two-segment deformation of the ORIGINAL pixels; no AI redraw is used.")

# ----------------------------- settings -----------------------------
st.sidebar.header("⚙️ Settings")
GEMINI_API_KEY = st.sidebar.text_input("Gemini API Key", type="password")

ANIMATION_MODE = st.sidebar.selectbox("Animation mode", [
    "Walk in → walk in place → merge",
    "Walk in → walk in place",
    "Walk in place only",
])
TOTAL_FRAMES = st.sidebar.slider("Total animation frames", 24, 120, 64, 2)
FPS = st.sidebar.slider("FPS", 4, 20, 7)
WALK_CYCLES = st.sidebar.slider("Walking cycles", 1, 5, 2)

st.sidebar.markdown("### 🦵 Motion")
STEP_ANGLE = st.sidebar.slider("Leg swing", 3.0, 22.0, 10.0, 0.5)
KNEE_BEND = st.sidebar.slider("Knee bend", 0.0, 18.0, 7.0, 0.5)
FOOT_LIFT = st.sidebar.slider("Foot lift", 0.0, 0.10, 0.035, 0.005)
BODY_BOB = st.sidebar.slider("Body bob", 0.0, 0.035, 0.006, 0.001)
GROUND_LOCK = st.sidebar.slider("Ground contact", 0.0, 1.0, 0.85, 0.05)

st.sidebar.markdown("### 🎞️ Timing")
WALK_IN_FRACTION = st.sidebar.slider("Walk-in portion", 0.05, 0.40, 0.18, 0.02)
MERGE_FRACTION = st.sidebar.slider("Final merge portion", 0.05, 0.30, 0.14, 0.02)

st.sidebar.markdown("### 🧹 Extraction")
INK_DILATION = st.sidebar.slider("Ink capture", 1, 7, 3, 1)
SHADOW_SUPPRESSION = st.sidebar.checkbox("Suppress floor shadows", True)

# ----------------------------- Gemini -----------------------------
def clean_json_text(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"): lines = lines[1:]
        if lines and lines[-1].strip() == "```": lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text

def model_name(obj: Any) -> str:
    return str(getattr(obj, "name", "") or "").strip()

def model_actions(obj: Any) -> List[str]:
    a = getattr(obj, "supported_actions", None)
    if a is None: a = getattr(obj, "supportedActions", None)
    try: return [str(x) for x in (a or [])]
    except Exception: return []

def discover_models(client: genai.Client) -> List[str]:
    out = []
    try:
        for m in client.models.list():
            n = model_name(m)
            if not n: continue
            acts = model_actions(m)
            if acts and not any("generatecontent" == x.lower() for x in acts): continue
            out.append(n)
    except Exception:
        return []
    seen = set(); unique = []
    for n in out:
        if n.lower() not in seen:
            seen.add(n.lower()); unique.append(n)
    return unique

def model_score(n: str) -> Tuple[int, str]:
    x = n.lower().replace("models/", "")
    patterns = [
        ("gemini-3.7-flash", 0), ("gemini-3.6-flash", 1), ("gemini-3.5-flash", 2),
        ("gemini-3.1-flash-lite", 4), ("gemini-3.1-flash", 3),
        ("gemini-3", 5), ("gemini-2.5-flash-lite", 7), ("gemini-2.5-flash", 6),
        ("gemini-2.5", 8), ("gemini-2", 10), ("gemini", 20),
    ]
    for p, s in patterns:
        if p in x: return s, x
    return 100, x

def safe_text(resp: Any) -> str:
    t = getattr(resp, "text", None)
    if t: return str(t)
    parts = []
    try:
        for c in getattr(resp, "candidates", []) or []:
            for p in getattr(getattr(c, "content", None), "parts", []) or []:
                if getattr(p, "text", None): parts.append(str(p.text))
    except Exception: pass
    return "\n".join(parts).strip()

def analyze_scene(image_bytes: bytes, api_key: str) -> Optional[Dict[str, Any]]:
    if not api_key:
        st.error("Please enter your Gemini API key."); return None
    try: client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Could not initialize Gemini: {e}"); return None
    models = discover_models(client)
    models = sorted(models, key=model_score)
    if not models:
        st.error("No generateContent-capable Gemini model was returned for this API key."); return None

    prompt = r'''Analyze this SINGLE hand-drawn animal scene for a 2D cut-out animation system.
DO NOT redraw it. Return geometry only. Coordinates are normalized 0..100 as [y,x].
The Python program will move ORIGINAL pixels, so geometry must be conservative and follow visible ink.

CRITICAL:
- Identify the main animal only; do not include floor, cast shadows, table edges, glare, or background.
- The animal polygon must tightly surround the visible animal.
- For EVERY visible leg, give a tight polygon around the actual leg drawing only. Do NOT include the floor shadow below the hoof.
- Give three joints per leg: proximal, middle, distal.
- Proximal is where the leg attaches to the body.
- Middle is the actual knee/elbow bend location, not halfway by default.
- Distal is the ankle/wrist/hoof connection near the end of the drawn leg, BEFORE any cast shadow.
- If a leg is mostly straight, still place middle near the natural anatomical bend.
- Give a side/name: front_left, front_right, back_left, back_right when possible.
- Do not invent hidden legs.
- Also give head, neck, tail, and body polygons only when clearly visible, but legs are the priority.
- Estimate the visible drawing, not imagined anatomy.

Return ONLY valid JSON:
{
  "identified_character":"giraffe",
  "animal_bbox":[ymin,xmin,ymax,xmax],
  "animal_polygon":[[y,x],...],
  "parts":[
    {"name":"body","type":"body","polygon":[[y,x],...]},
    {"name":"front_left_leg","type":"leg","side":"front_left","polygon":[[y,x],...],"joints":{"proximal":[y,x],"middle":[y,x],"distal":[y,x]} }
  ],
  "notes":"..."
}
If no animal is visible, return identified_character=none and empty parts.'''

    errors=[]
    for current in models:
        try:
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/png"), prompt]
            try:
                resp=client.models.generate_content(model=current, contents=contents,
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.05))
            except Exception:
                resp=client.models.generate_content(model=current, contents=contents,
                    config=types.GenerateContentConfig(temperature=0.05))
            data=json.loads(clean_json_text(safe_text(resp)))
            if not isinstance(data, dict): raise ValueError("Model response was not a JSON object")
            st.success(f"✅ Anatomy analysis completed with `{current}`")
            return data
        except Exception as e:
            errors.append(f"{current}: {str(e).replace(chr(10),' ')}")
    st.error("Gemini analysis failed after trying all available models.")
    with st.expander("Model attempts"): [st.code(e) for e in errors]
    return None

# ----------------------------- geometry -----------------------------
def pt_px(p, w, h):
    y=float(np.clip(p[0],0,100)); x=float(np.clip(p[1],0,100))
    return np.array([x*w/100.0, y*h/100.0], dtype=np.float32)

def poly_px(poly,w,h):
    if not isinstance(poly,list): return np.empty((0,2),np.int32)
    pts=[]
    for p in poly:
        if isinstance(p,(list,tuple)) and len(p)>=2: pts.append(pt_px(p,w,h))
    return np.asarray(pts,np.int32) if len(pts)>=3 else np.empty((0,2),np.int32)

def poly_mask(shape, poly, dil=0):
    h,w=shape[:2]; m=np.zeros((h,w),np.uint8); p=poly_px(poly,w,h)
    if len(p)>=3: cv2.fillPoly(m,[p],255)
    if dil:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dil+1,2*dil+1)); m=cv2.dilate(m,k)
    return m

def bbox_poly(poly,w,h,margin=10):
    p=poly_px(poly,w,h)
    if len(p)<3: return 0,0,w,h
    x,y,bw,bh=cv2.boundingRect(p)
    return max(0,x-margin),max(0,y-margin),min(w,x+bw+margin),min(h,y+bh+margin)

def bbox_norm(bb,w,h,margin=10):
    if not isinstance(bb,list) or len(bb)!=4: return 0,0,w,h
    ymin,xmin,ymax,xmax=[float(v) for v in bb]
    x1=int(xmin*w/100); y1=int(ymin*h/100); x2=int(xmax*w/100); y2=int(ymax*h/100)
    return max(0,x1-margin),max(0,y1-margin),min(w,x2+margin),min(h,y2+margin)

def rotation_matrix(angle_deg, center):
    return cv2.getRotationMatrix2D((float(center[0]),float(center[1])), angle_deg, 1.0)

def affine_rotate_point(p, center, angle):
    a=math.radians(angle); c=math.cos(a); s=math.sin(a); q=np.asarray(p,dtype=np.float32)-center
    return center+np.array([c*q[0]-s*q[1], s*q[0]+c*q[1]],np.float32)

# ----------------------------- foreground extraction -----------------------------
def ink_mask(image_bgr, region_mask, shadow_suppress=True, dilation=3):
    """Keep colored drawing plus black/brown outlines attached to colored ink; reject soft gray floor shadows."""
    hsv=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2HSV)
    H,S,V=cv2.split(hsv)
    colored=(S>42).astype(np.uint8)*255
    # Yellow/brown colored pixels are the strongest foreground cue.
    strong_color=((S>60)&(V>55)).astype(np.uint8)*255
    color_support=cv2.dilate(strong_color,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilation+3,2*dilation+3)))
    dark=(V<125).astype(np.uint8)*255
    # Very dark ink is only admitted when it touches colored drawing.
    dark_attached=cv2.bitwise_and(dark,color_support)
    if shadow_suppress:
        fg=cv2.bitwise_or(colored,dark_attached)
    else:
        fg=cv2.bitwise_or(colored,dark)
    fg=cv2.bitwise_and(fg,region_mask)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fg=cv2.morphologyEx(fg,cv2.MORPH_CLOSE,k,iterations=1)
    if dilation>0:
        fg=cv2.dilate(fg,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilation+1,2*dilation+1)))
    # Keep only pixels inside the region; no floor outside polygon.
    return fg

def rgba_from_mask(image_bgr, mask, bbox):
    x1,y1,x2,y2=bbox; crop=image_bgr[y1:y2,x1:x2].copy(); m=mask[y1:y2,x1:x2]
    if crop.size==0: return np.zeros((1,1,4),np.uint8)
    rgba=cv2.cvtColor(crop,cv2.COLOR_BGR2BGRA); rgba[:,:,3]=cv2.GaussianBlur(m,(3,3),0); return rgba

# ----------------------------- leg preparation -----------------------------
def prepare_leg(image_bgr, part):
    poly=part.get("polygon") or []; joints=part.get("joints") or {}
    if len(poly)<3 or not all(k in joints for k in ("proximal","middle","distal")): return None
    h,w=image_bgr.shape[:2]
    region=poly_mask((h,w),poly,dil=max(1,INK_DILATION))
    mask=ink_mask(image_bgr,region,SHADOW_SUPPRESSION,INK_DILATION)
    bbox=bbox_poly(poly,w,h,margin=max(20,min(h,w)//60))
    x1,y1,x2,y2=bbox
    sprite=cv2.cvtColor(image_bgr[y1:y2,x1:x2],cv2.COLOR_BGR2BGRA)
    sm=mask[y1:y2,x1:x2]
    sprite[:,:,3]=cv2.GaussianBlur(sm,(3,3),0)
    g={k:pt_px(joints[k],w,h) for k in ("proximal","middle","distal")}
    local={k:g[k]-np.array([x1,y1],np.float32) for k in g}
    return {"name":part.get("name","leg"),"side":part.get("side",part.get("name","leg")),"sprite":sprite,
            "bbox":bbox,"joints":local,"global":g,"mask":sm}

def leg_parts(scene):
    out=[]
    for p in scene.get("parts",[]) or []:
        if isinstance(p,dict) and (str(p.get("type","")).lower()=="leg" or "leg" in str(p.get("name","")).lower()): out.append(p)
    return out

# ----------------------------- articulated deformation -----------------------------
def warp_rgba_with_affine(sprite, alpha_mask, M):
    h,w=sprite.shape[:2]
    # Transform all pixels; transparent background remains transparent.
    warped=cv2.warpAffine(sprite,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
    wm=cv2.warpAffine(alpha_mask,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    warped[:,:,3]=wm
    return warped

def articulated_leg(leg, swing, knee, lift, scale=1.0):
    """Two-joint cutout: upper segment follows hip/shoulder; lower segment follows the moved knee + extra knee bend."""
    sprite=leg["sprite"]; h,w=sprite.shape[:2]
    P=leg["joints"]["proximal"]; M0=leg["joints"]["middle"]; D0=leg["joints"]["distal"]
    v=D0-P; L=max(1.0,float(np.linalg.norm(v))); u=v/L
    projM=float(np.dot(M0-P,u)/L)
    yy,xx=np.mgrid[0:h,0:w].astype(np.float32)
    q=np.stack([xx,yy],axis=-1)
    proj=np.sum((q-P)*u,axis=-1)/L
    band=0.075
    upper_w=np.clip((proj-(projM+band))/(-2*band),0,1)
    lower_w=1.0-upper_w
    base_alpha=sprite[:,:,3].astype(np.float32)/255.0
    # Remove pixels not belonging to either region.
    upper=(base_alpha*upper_w*255).astype(np.uint8)
    lower=(base_alpha*lower_w*255).astype(np.uint8)

    # Upper leg rotates around attachment.
    R1=rotation_matrix(swing,P)
    upper_rgba=warp_rgba_with_affine(sprite,upper,R1)
    M1=affine_rotate_point(M0,P,swing)
    D1=affine_rotate_point(D0,P,swing)

    # Lower leg first follows the upper rotation, then bends at the knee.
    # A modest opposite sign produces a natural forward/back knee fold.
    R2a=rotation_matrix(swing,M0)
    lower1=warp_rgba_with_affine(sprite,lower,R2a)
    R2b=rotation_matrix(knee,M1)
    lower2=warp_rgba_with_affine(lower1,lower1[:,:,3],R2b)

    # Translate the lower layer because R2a/R2b are expressed in original local coordinates.
    # The first rotation keeps M0 fixed; second keeps M1 fixed only approximately in same canvas.
    # Correct its anchor by aligning transformed M0 to M1.
    anchor_after=affine_rotate_point(M0,M0,knee)  # equals M0; used only for clarity
    # R2b around M1 on the canvas is correct; lower1 was already around M0, so M0 stays M0.
    # Shift the result by M1-M0 to place the knee at the moved upper knee.
    shift=M1-M0
    shifted=np.zeros_like(lower2)
    dx,dy=float(shift[0]),float(shift[1])
    T=np.array([[1,0,dx],[0,1,dy]],np.float32)
    shifted=cv2.warpAffine(lower2,T,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))

    # Foot lift is vertical in image coordinates (up = negative y), reduced by ground lock.
    lift_px=float(lift*h)
    if abs(lift_px)>0.01:
        T3=np.array([[1,0,0],[0,1,-lift_px]],np.float32)
        shifted=cv2.warpAffine(shifted,T3,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))

    return upper_rgba, shifted

def alpha_over(dst, src, x, y):
    if src is None or src.size==0:return dst
    sh,sw=src.shape[:2]; H,W=dst.shape[:2]
    x1=max(0,x); y1=max(0,y); x2=min(W,x+sw); y2=min(H,y+sh)
    if x1>=x2 or y1>=y2:return dst
    sx1=x1-x; sy1=y1-y; sx2=sx1+(x2-x1); sy2=sy1+(y2-y1)
    s=src[sy1:sy2,sx1:sx2].astype(np.float32); a=(s[:,:,3:4]/255.0); d=dst[y1:y2,x1:x2].astype(np.float32)
    dst[y1:y2,x1:x2]=np.clip(s[:,:,:3]*a+d*(1-a),0,255).astype(np.uint8)
    return dst

def erase_masked_polygon(image, polygon, shadow=True):
    h,w=image.shape[:2]; region=poly_mask((h,w),polygon,dil=max(3,INK_DILATION+2)); mask=ink_mask(image,region,shadow,INK_DILATION+1)
    # Also remove a small outline halo so the rest pose doesn't leave a duplicate leg.
    mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    try:return cv2.inpaint(image,mask,5,cv2.INPAINT_TELEA)
    except:return image

# ----------------------------- animal plate -----------------------------
def extract_animal(image,scene):
    h,w=image.shape[:2]; poly=scene.get("animal_polygon") or []
    if len(poly)>=3:
        bbox=bbox_poly(poly,w,h,margin=max(8,min(h,w)//150)); region=poly_mask((h,w),poly,dil=INK_DILATION)
    else:
        bbox=bbox_norm(scene.get("animal_bbox"),w,h,margin=max(8,min(h,w)//150)); region=np.zeros((h,w),np.uint8); x1,y1,x2,y2=bbox; region[y1:y2,x1:x2]=255
    mask=ink_mask(image,region,SHADOW_SUPPRESSION,INK_DILATION)
    x1,y1,x2,y2=bbox; rgba=cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2BGRA); rgba[:,:,3]=mask[y1:y2,x1:x2]
    return rgba,bbox,mask

def clean_animation_plate(image,scene):
    plate=image.copy()
    for p in scene.get("parts",[]) or []:
        if isinstance(p,dict) and (str(p.get("type","")).lower()=="leg" or "leg" in str(p.get("name","")).lower()):
            plate=erase_masked_polygon(plate,p.get("polygon") or [],SHADOW_SUPPRESSION)
    return plate

# ----------------------------- keyframes -----------------------------
def phase_for(side,index):
    s=str(side).lower()
    if "front_left" in s:return 0.0
    if "back_right" in s:return 0.0
    if "front_right" in s:return math.pi
    if "back_left" in s:return math.pi
    return 0 if index%2==0 else math.pi

def keyframes(image,prepared,scene):
    plate=clean_animation_plate(image,scene); h,w=image.shape[:2]; out=[]
    # Four-contact locomotion pattern: diagonal pairs alternate, with knees bending during lift.
    poses=[0.00,0.25,0.50,0.75]
    for t in poses:
        f=plate.copy()
        bob=math.sin(2*math.pi*t)*h*BODY_BOB
        for idx,leg in enumerate(prepared):
            s=math.sin(2*math.pi*t+phase_for(leg["side"],idx))
            smooth=s*(0.65+0.35*abs(s))
            side=str(leg["side"]).lower()
            swing=smooth*STEP_ANGLE
            if "back" in side:swing*=0.88
            # Knee bends more when the foot is moving forward/up.
            lift=max(0.0,smooth)*FOOT_LIFT*(1.0-GROUND_LOCK*0.25)
            knee=-math.copysign(KNEE_BEND,max(abs(smooth),0.001)) if s>0 else math.copysign(KNEE_BEND*0.45,s)
            up,low=articulated_leg(leg,swing,knee,lift)
            bx,by=leg["bbox"][0],leg["bbox"][1]
            # Both sprites use the same local crop, so the original proximal point stays fixed after the rotation.
            f=alpha_over(f,up,bx,by+int(bob))
            f=alpha_over(f,low,bx,by+int(bob))
        out.append(f)
    return out

def ease(t):return t*t*(3-2*t)

def interpolate(a,b,t):return cv2.addWeighted(a,1-t,b,t,0)

def cycle_frames(keys,n):
    if not keys:return []
    out=[]
    n=max(4,n)
    # Interpolate each quarter with image-space dissolve. This is acceptable for preview, but the final sequence below
    # primarily repeats pose drawings; motion itself comes from joint deformation.
    for i in range(n):
        pos=(i/n)*4.0; k=int(pos)%4; frac=ease(pos-int(pos)); out.append(interpolate(keys[k],keys[(k+1)%4],frac))
    return out

# ----------------------------- walk-in / merge -----------------------------
def walk_in(background,animal,bbox,p):
    x1,y1,_,_=bbox; ah,aw=animal.shape[:2]
    x=int((-aw-30)+(x1+aw+30)*ease(p)); y=y1+int(math.sin(p*math.pi*4)*background.shape[0]*0.004)
    return alpha_over(background.copy(),animal,x,y)

def remove_animal_plate(image,animal,bbox):
    x1,y1,x2,y2=bbox; mask=np.zeros(image.shape[:2],np.uint8); m=animal[:,:,3]
    mask[y1:y2,x1:x2]=m; mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    try:return cv2.inpaint(image,mask,7,cv2.INPAINT_TELEA)
    except:return image

def merge(a,b,p):return cv2.addWeighted(a,1-ease(p),b,ease(p),0)

def build_animation(image,animal,bbox,keys,total,cycles,mode,win_frac,merge_frac):
    original=image.copy(); background=remove_animal_plate(image,animal,bbox)
    walk_n=max(1,int(total*win_frac)) if mode!="Walk in place only" else 0
    merge_n=max(1,int(total*merge_frac)) if mode=="Walk in → walk in place → merge" else 0
    cycle_n=max(1,total-walk_n-merge_n)
    # Generate enough frames for the requested number of cycles and resample to the available slot.
    source_n=max(16,cycles*16)
    cyc=cycle_frames(keys,source_n)
    seq=[]
    if walk_n:
        for i in range(walk_n):seq.append(walk_in(background,animal,bbox,i/max(1,walk_n-1)))
    for i in range(cycle_n):seq.append(cyc[i%len(cyc)].copy())
    if merge_n:
        last=seq[-1] if seq else original
        for i in range(merge_n):seq.append(merge(last,original,i/max(1,merge_n-1)))
    return seq[:total]+([original.copy()]*(total-len(seq)))

# ----------------------------- exports -----------------------------
def gif_bytes(frames,fps):
    if not frames:return b""
    ims=[Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)) for f in frames]; b=io.BytesIO()
    ims[0].save(b,format="GIF",save_all=True,append_images=ims[1:],duration=max(30,int(1000/fps)),loop=0,optimize=False); return b.getvalue()

def mp4_bytes(frames,fps):
    if not frames:return None
    fd,path=tempfile.mkstemp(suffix=".mp4"); os.close(fd); h,w=frames[0].shape[:2]
    writer=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*"mp4v"),float(fps),(w,h))
    if not writer.isOpened():return None
    for f in frames:writer.write(f)
    writer.release()
    try:
        with open(path,"rb") as fh:return fh.read()
    finally:
        try:os.remove(path)
        except OSError:pass

def zip_frames(frames,prefix="frame"):
    b=io.BytesIO()
    with zipfile.ZipFile(b,"w",zipfile.ZIP_DEFLATED) as z:
        for i,f in enumerate(frames,1):
            ok,e=cv2.imencode(".png",f)
            if ok:z.writestr(f"{prefix}_{i:03d}.png",e.tobytes())
    return b.getvalue()

# ----------------------------- detection visualization -----------------------------
def detection_overlay(image,scene):
    o=image.copy(); h,w=o.shape[:2]
    p=poly_px(scene.get("animal_polygon") or [],w,h)
    if len(p)>=3:cv2.polylines(o,[p],True,(0,180,0),max(2,min(h,w)//300))
    for part in scene.get("parts",[]) or []:
        if not isinstance(part,dict):continue
        p=poly_px(part.get("polygon") or [],w,h)
        if len(p)>=3:cv2.polylines(o,[p],True,(255,120,0),max(1,min(h,w)//450))
        for name,j in (part.get("joints") or {}).items():
            if isinstance(j,(list,tuple)) and len(j)>=2:
                x,y=pt_px(j,w,h).astype(int); cv2.circle(o,(x,y),5,(0,0,255),-1); cv2.putText(o,str(name),(x+5,y-5),0,.45,(0,0,255),1,cv2.LINE_AA)
    return o

# ----------------------------- UI -----------------------------
up=st.file_uploader("Upload your hand-drawn scene",type=["png","jpg","jpeg"])
if not up:
    st.info("Upload the drawing containing the animal and background."); st.stop()
raw=up.getvalue()
try:
    pil=Image.open(io.BytesIO(raw)).convert("RGB"); pb=io.BytesIO(); pil.save(pb,format="PNG"); gemini_bytes=pb.getvalue()
except Exception as e:
    st.error(f"Could not prepare image: {e}"); st.stop()
image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
if image is None:st.error("Could not read image.");st.stop()

c1,c2=st.columns(2)
with c1:st.image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),caption="Original drawing",use_container_width=True)
with c2:
    st.markdown("### New animation pipeline\n1. Gemini finds anatomy\n2. Python builds an **ink-only** foreground mask\n3. Each leg is split at the knee\n4. Upper + lower segments rotate around **different joints**\n5. Feet are gently lifted and ground-contact is preserved\n6. Four poses are interpolated and rendered slowly\n7. Walk-in and final merge use a separate clean background plate")

if st.button("🔍 Analyze drawing with Gemini",type="primary",use_container_width=True):
    with st.spinner("Analyzing animal anatomy..."):
        d=analyze_scene(gemini_bytes,GEMINI_API_KEY)
    if d:
        st.session_state.scene=d; st.session_state.pop("keys",None); st.session_state.pop("frames",None)

if "scene" not in st.session_state:st.stop()
scene=st.session_state.scene
st.success(f"Detected: **{scene.get('identified_character','character')}**")
with st.expander("Gemini anatomy JSON"):st.json(scene)
st.image(cv2.cvtColor(detection_overlay(image,scene),cv2.COLOR_BGR2RGB),caption="Detection: green animal, orange leg polygons, red joints",use_container_width=True)

parts=leg_parts(scene)
prepared=[]
for p in parts:
    q=prepare_leg(image,p)
    if q:prepared.append(q)
if not prepared:
    st.error("No usable leg polygons/joints were returned. Try a clearer photo or analyze again.");st.stop()
st.success(f"Prepared {len(prepared)} original leg cut-outs.")

st.markdown("---")
st.header("🦵 Step 2 — Joint-aware walking poses")
st.write("Unlike the old version, the whole leg is no longer rotated as one rigid stick. The upper leg follows the attachment joint and the lower leg follows the moved knee, which creates an actual bend.")

if st.button("🦒 Generate 4 improved walking poses",type="primary",use_container_width=True):
    with st.spinner("Building articulated poses..."):
        keys=keyframes(image,prepared,scene)
        st.session_state.keys=keys; st.session_state.pop("frames",None)

if "keys" in st.session_state:
    cols=st.columns(4)
    for i,f in enumerate(st.session_state.keys):
        with cols[i]:st.image(cv2.cvtColor(f,cv2.COLOR_BGR2RGB),caption=f"Pose {i+1}",use_container_width=True)
    st.download_button("⬇️ Download 4 key poses",zip_frames(st.session_state.keys,"walk_pose"),"animal_walk_keyposes.zip","application/zip",use_container_width=True)

st.markdown("---")
st.header("🎞️ Step 3 — Render animation")
if st.button("🚀 Render animation",type="primary",use_container_width=True):
    with st.spinner("Rendering..."):
        animal,bbox,_=extract_animal(image,scene)
        frames=build_animation(image,animal,bbox,st.session_state.keys,TOTAL_FRAMES,WALK_CYCLES,ANIMATION_MODE,WALK_IN_FRACTION,MERGE_FRACTION)
        st.session_state.frames=frames
    st.success(f"Rendered {len(frames)} frames at {FPS} FPS.")

if "frames" in st.session_state:
    frames=st.session_state.frames; gb=gif_bytes(frames,FPS)
    st.markdown("### 🎉 Result")
    st.image(gb,caption="Articulated hand-drawn walking animation",use_container_width=True)
    a,b,c=st.columns(3)
    with a:st.download_button("⬇️ Download GIF",gb,"hand_drawn_animal_walk_v2.gif","image/gif",use_container_width=True)
    with b:
        mb=mp4_bytes(frames,FPS)
        if mb:st.download_button("⬇️ Download MP4",mb,"hand_drawn_animal_walk_v2.mp4","video/mp4",use_container_width=True)
        else:st.info("MP4 unavailable in this environment.")
    with c:st.download_button("⬇️ Download all PNG frames",zip_frames(frames),"hand_drawn_animal_walk_v2_frames.zip","application/zip",use_container_width=True)
    st.markdown("### Preview frames")
    idx=np.linspace(0,len(frames)-1,min(12,len(frames)),dtype=int)
    cols=st.columns(4)
    for n,i in enumerate(idx):
        with cols[n%4]:st.image(cv2.cvtColor(frames[i],cv2.COLOR_BGR2RGB),caption=f"Frame {i+1}",use_container_width=True)

st.markdown("---")
st.caption("v2: original-pixel articulated cutout animation. Gemini is used for anatomy/geometry only; it does not generate replacement artwork.")

