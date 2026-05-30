"""Genera mtc_obb_pipeline.ipynb: notebook autocontenido para Kaggle (GPU T4) que
hace TODO el pipeline OBB sin depender de la maquina local.

Pasos del notebook:
  1. Instala gdown + ultralytics.
  2. Descarga los datos desde la carpeta de Google Drive del concurso (gdown).
  3. Descomprime train.zip y test.zip.
  4. Parsea train.csv, construye la ROI por video (rejilla de ocupacion) y convierte
     a dataset YOLO-OBB (recorte + ennegrecido a la ROI, convencion de angulo ccw_math).
  5. Entrena YOLO-OBB (yolo11s-obb) con augmentation aerea.
  6. Valida y guarda metricas por clase + best.pt en /kaggle/working.

No usa nbformat: arma el JSON del notebook a mano.
"""
import json
from pathlib import Path

DRIVE_FOLDER = "1gnCApZW1oUsh1Bb1LtSQlEeoBGmC9AjA"

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}

cells = []

cells.append(md(
"""# SMART CHALLENGE 2026 — Pipeline YOLO-OBB (autocontenido)

Detección y clasificación vehicular con **oriented bounding boxes** sobre las 9 clases
oficiales. Este notebook hace todo en Kaggle (GPU T4): descarga los datos desde Google
Drive, construye la ROI por video, convierte a YOLO-OBB y entrena.

**Antes de Run All:** Settings → Accelerator → **GPU T4** y → Internet → **On**.

Clases (category_id): 1 auto · 2 combi · 3 microbus · 4 minibus · 5 omnibus ·
6 articulado · 7 camion · 8 mototaxi · 9 motocicleta.
"""))

cells.append(md("## 1. Dependencias"))
cells.append(code(
"""
import os, sys, subprocess
def sh(c): print("+", c); assert subprocess.call(c, shell=True) == 0, c
sh(f"{sys.executable} -m pip install -q 'ultralytics>=8.3.0' pyyaml")
import numpy as np
"""))

cells.append(md(
"""## 2. Localizar datos (Kaggle Dataset adjunto)

Los datos del concurso se subieron una vez como **Kaggle Dataset** privado
(`yeltsinvalero/mtc-smart-challenge-2026-data`) porque la descarga directa de Google
Drive falla por *cuota* en los zips grandes (train 43 GB / test 18 GB).

**Add Input →** ese dataset antes de Run All. Aparecerá bajo `/kaggle/input/...`.
"""))
cells.append(code(
'''
from pathlib import Path
WORK = Path("/kaggle/working"); WORK.mkdir(exist_ok=True)
INPUT = Path("/kaggle/input")

# localizar la carpeta del dataset adjunto (la que contenga train.csv)
DATA = None
for d in sorted(INPUT.glob("*")):
    if any(d.rglob("train.csv")):
        DATA = next(p.parent for p in d.rglob("train.csv")); break
assert DATA is not None, "Adjunta el dataset 'mtc-smart-challenge-2026-data' (Add Input)."
print("DATA:", DATA)
for p in sorted(DATA.iterdir()):
    if p.is_file():
        print("  %s  %.1f MB" % (p.name, p.stat().st_size/1e6))
'''))

cells.append(md("## 3. Localizar y descomprimir"))
cells.append(code(
"""
import zipfile, glob
def find(name_substr, exts):
    hits = [p for p in DATA.rglob("*") if p.is_file() and p.suffix.lower() in exts
            and name_substr in p.name.lower()]
    return hits[0] if hits else None

train_csv = next((p for p in DATA.rglob("*.csv") if "train" in p.name.lower()), None)
train_zip = find("train", {".zip"})
test_zip  = find("test",  {".zip"})
print("train_csv:", train_csv); print("train_zip:", train_zip); print("test_zip:", test_zip)

IMG_ROOT = WORK / "images"; IMG_ROOT.mkdir(exist_ok=True)
def unzip(z, dest):
    if z and not (dest).exists():
        print("unzip", z, "->", dest)
        with zipfile.ZipFile(z) as zf: zf.extractall(dest)
unzip(train_zip, IMG_ROOT / "train")
unzip(test_zip,  IMG_ROOT / "test")
train_imgs = list((IMG_ROOT/"train").rglob("*.jpg"))
print("train imgs:", len(train_imgs))
"""))

cells.append(md(
"""## 4. ROI por video + conversión a YOLO-OBB

- **ROI por rejilla de ocupación**: marca las celdas (64 px) con anotaciones, acumulado
  por video, dilatadas 1 celda. Sigue la forma de cruz de la intersección sin rellenar
  esquinas (manzanas con vehículos no etiquetados).
- **Recorte + ennegrecido** a la ROI → negativos limpios (no enseña "vehículo = fondo").
- Ángulo **ccw_math** (validado): `θ = deg2rad(angle_deg)`.
- **Split por video** (no por frame) para evitar fuga.
"""))
cells.append(code(
r"""
import csv, math, io
from PIL import Image
import numpy as np, yaml

W, H = 1920, 1080
CELL = 64; DILATE = 1; VAL_FRAC = 0.15
ID2NAME = {1:"auto",2:"combi",3:"microbus",4:"minibus",5:"omnibus",
           6:"articulado",7:"camion",8:"mototaxi",9:"motocicleta"}
NAMES = [ID2NAME[i] for i in range(1,10)]

def corners(cx,cy,w,h,ang):
    t=math.radians(ang); c,s=math.cos(t),math.sin(t); dx,dy=w/2,h/2
    P=[(-dx,-dy),(dx,-dy),(dx,dy),(-dx,dy)]
    return [(cx+px*c-py*s, cy+px*s+py*c) for px,py in P]

# parse
frames={}
with open(train_csv, newline="", encoding="utf-8") as f:
    r=csv.reader(f); next(r)
    for line in r:
        if len(line)<2: continue
        fid,t=line[0].strip(),line[1].strip(); bx=[]
        if t and t.lower()!="none":
            for d in t.split(";"):
                p=d.split()
                if len(p)<6: continue
                cid=int(float(p[0])); cx,cy,w,h,a=map(float,p[1:6])
                if w>0 and h>0: bx.append((cid,cx,cy,w,h,a))
        frames[fid]=bx
print("frames:", len(frames))

# ROI grid por video
gw,gh = math.ceil(W/CELL), math.ceil(H/CELL)
def dilate(m,k):
    o=m.copy()
    for _ in range(k):
        x=o.copy()
        x[:-1]|=o[1:]; x[1:]|=o[:-1]; x[:,:-1]|=o[:,1:]; x[:,1:]|=o[:,:-1]; o=x
    return o
roi={}
by_vid={}
for fid in frames: by_vid.setdefault(fid.rsplit("_",1)[0],[]).append(fid)
for vid,fids in by_vid.items():
    g=np.zeros((gh,gw),bool)
    for fid in fids:
        for (cid,cx,cy,w,h,a) in frames[fid]:
            cs=corners(cx,cy,w,h,a); xs=[p[0] for p in cs]; ys=[p[1] for p in cs]
            c0=max(0,int(min(xs)//CELL)); c1=min(gw-1,int(max(xs)//CELL))
            r0=max(0,int(min(ys)//CELL)); r1=min(gh-1,int(max(ys)//CELL))
            g[r0:r1+1,c0:c1+1]=True
    roi[vid]=dilate(g,DILATE) if g.any() else g

# convertir
OUT = WORK/"dataset_obb"
for sp in ("train","val"):
    (OUT/"images"/sp).mkdir(parents=True,exist_ok=True)
    (OUT/"labels"/sp).mkdir(parents=True,exist_ok=True)
val_vids={v for v in by_vid if (hash(v)%1000)/1000.0 < VAL_FRAC}
rep={"frames":0,"boxes":0,"by_class":{n:0 for n in NAMES}}
src_train = IMG_ROOT/"train"
img_index={p.stem:p for p in src_train.rglob("*.jpg")}
for vid,fids in by_vid.items():
    m=roi.get(vid)
    if m is None or not m.any(): continue
    ys,xs=np.where(m)
    x0,x1=int(xs.min()*CELL),int(min(W,(xs.max()+1)*CELL))
    y0,y1=int(ys.min()*CELL),int(min(H,(ys.max()+1)*CELL))
    sp="val" if vid in val_vids else "train"
    full=np.repeat(np.repeat(m,CELL,0),CELL,1)[:H,:W]
    for fid in fids:
        if fid not in img_index: continue
        arr=np.asarray(Image.open(img_index[fid]).convert("RGB"))
        arr=np.where(full[:,:,None],arr,0)
        crop=arr[y0:y1,x0:x1]; ch,cw=crop.shape[:2]
        lines=[]
        for (cid,cx,cy,w,h,a) in frames[fid]:
            gc=min(gw-1,max(0,int(cx//CELL))); gr=min(gh-1,max(0,int(cy//CELL)))
            if not m[gr,gc]: continue
            cs=corners(cx,cy,w,h,a); norm=[]
            for (px,py) in cs: norm+=[ (px-x0)/cw, (py-y0)/ch ]
            xs_n=norm[0::2]; ys_n=norm[1::2]
            if max(xs_n)<0 or min(xs_n)>1 or max(ys_n)<0 or min(ys_n)>1: continue
            norm=[min(1,max(0,v)) for v in norm]
            lines.append(f"{cid-1} "+" ".join(f"{v:.6f}" for v in norm))
            rep["boxes"]+=1; rep["by_class"][NAMES[cid-1]]+=1
        Image.fromarray(crop).save(OUT/"images"/sp/f"{fid}.jpg", quality=88)
        (OUT/"labels"/sp/f"{fid}.txt").write_text("\n".join(lines)+("\n" if lines else ""))
        rep["frames"]+=1
yaml.safe_dump({"path":str(OUT),"train":"images/train","val":"images/val",
               "names":{i:NAMES[i] for i in range(9)}},
              open(OUT/"data.yaml","w"), allow_unicode=True, sort_keys=False)
print("convertido:", rep["frames"], "frames |", rep["boxes"], "cajas")
print(rep["by_class"])
"""))

cells.append(md("## 5. Entrenar YOLO-OBB"))
cells.append(code(
"""
import torch, json, shutil
from ultralytics import YOLO
dev = 0 if torch.cuda.is_available() else "cpu"
print("cuda:", torch.cuda.is_available())
model = YOLO("yolo11s-obb.pt")
model.train(
    data=str(OUT/"data.yaml"), task="obb",
    imgsz=1280, epochs=60, batch=8, patience=15, seed=17, device=dev,
    optimizer="AdamW", lr0=0.001,
    hsv_h=0.012, hsv_s=0.5, hsv_v=0.4,
    degrees=180.0, translate=0.1, scale=0.5, fliplr=0.5, flipud=0.5,
    mosaic=0.8, close_mosaic=10, cls=0.7,
    project=str(WORK/"runs"), name="obb", exist_ok=True, verbose=True,
)
"""))

cells.append(md("## 6. Validar + guardar artefactos"))
cells.append(code(
"""
ART = WORK/"artifacts"; ART.mkdir(exist_ok=True)
m = model.val(data=str(OUT/"data.yaml"), task="obb", imgsz=1280, split="val", device=dev)
out={}
box=getattr(m,"box",None)
RARE=["combi","microbus","minibus","omnibus","articulado","mototaxi"]
if box is not None:
    out["map50_95"]=float(box.map); out["map50"]=float(box.map50)
    names=list(model.names.values())
    maps=getattr(box,"maps",None)
    if maps is not None:
        per={names[i]:float(v) for i,v in enumerate(list(maps)) if i<len(names)}
        out["per_class_map50_95"]=per
        rr=[per[c] for c in RARE if c in per]
        if rr: out["rare_map50_95"]=sum(rr)/len(rr)
print(json.dumps(out, indent=2))
(ART/"val_metrics.json").write_text(json.dumps(out,indent=2))
run=WORK/"runs"/"obb"
for n in ["weights/best.pt","weights/last.pt","results.csv","results.png","confusion_matrix.png"]:
    s=run/n
    if s.exists(): shutil.copy2(s, ART/Path(n).name)
print("artefactos:", [p.name for p in ART.iterdir()])
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
                   "language_info": {"name":"python"},
                   "accelerator":"GPU"},
      "nbformat": 4, "nbformat_minor": 5}

out_path = Path(__file__).resolve().parent / "mtc_obb_pipeline.ipynb"
out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Notebook generado:", out_path, "-", len(cells), "celdas")
