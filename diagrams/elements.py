import json

E = []
n = [0]
def uid(p):
    n[0] += 1
    return f"{p}{n[0]}"

BASE = dict(roughness=0, seed=1, version=1, versionNonce=1, isDeleted=False,
            groupIds=[], frameId=None, boundElements=None, updated=1, link=None,
            locked=False, angle=0, opacity=100, strokeStyle="solid", fillStyle="solid",
            strokeWidth=1.5, roundness=None)

def rect(x, y, w, h, stroke="#334155", bg="#ffffff", sw=1.5, dashed=False, radius=True):
    e = dict(BASE, type="rectangle", id=uid("r"), x=x, y=y, width=w, height=h,
             strokeColor=stroke, backgroundColor=bg, strokeWidth=sw,
             strokeStyle="dashed" if dashed else "solid",
             roundness={"type": 3} if radius else None)
    E.append(e); return e

def ellipse(x, y, w, h, stroke="#475569", bg="#e2e8f0"):
    E.append(dict(BASE, type="ellipse", id=uid("e"), x=x, y=y, width=w, height=h,
                  strokeColor=stroke, backgroundColor=bg))

def text(x, y, w, s, size=14, color="#1e293b", align="center", family=2):
    E.append(dict(BASE, type="text", id=uid("t"), x=x, y=y, width=w, height=size + 4,
                  strokeColor=color, backgroundColor="transparent", text=s,
                  fontSize=size, fontFamily=family, textAlign=align,
                  verticalAlign="top", containerId=None, originalText=s,
                  lineHeight=1.25, autoResize=False))

def arrow(points, stroke="#2563eb", sw=2, dashed=False, both=False):
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    ox, oy = points[0]
    E.append(dict(BASE, type="arrow", id=uid("a"), x=ox, y=oy,
                  width=max(xs) - min(xs), height=max(ys) - min(ys),
                  strokeColor=stroke, backgroundColor="transparent", strokeWidth=sw,
                  strokeStyle="dashed" if dashed else "solid",
                  points=[[p[0] - ox, p[1] - oy] for p in points],
                  startBinding=None, endBinding=None, elbowed=False,
                  startArrowhead="arrow" if both else None, endArrowhead="arrow"))

def line(points, stroke="#e2e8f0", sw=1.5, dashed=False):
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    ox, oy = points[0]
    E.append(dict(BASE, type="line", id=uid("l"), x=ox, y=oy,
                  width=max(xs) - min(xs), height=max(ys) - min(ys),
                  strokeColor=stroke, backgroundColor="transparent", strokeWidth=sw,
                  strokeStyle="dashed" if dashed else "solid",
                  points=[[p[0] - ox, p[1] - oy] for p in points],
                  startArrowhead=None, endArrowhead=None))

def box(x, y, w, h, lines, stroke="#334155", bg="#ffffff", sw=1.5, title_color="#1e293b"):
    rect(x, y, w, h, stroke, bg, sw)
    sizes = [16 if i == 0 else 12 for i in range(len(lines))]
    total = sum(s + 8 for s in sizes) - 8
    cy = y + (h - total) / 2
    for i, (s, sz) in enumerate(zip(lines, sizes)):
        text(x, cy, w, s, sz, title_color if i == 0 else "#64748b")
        cy += sz + 8

SLATE, ACCENT, GRAY = "#64748b", "#2563eb", "#94a3b8"
GREEN, GREEN_BG, GREEN_TX = "#059669", "#ecfdf5", "#065f46"
RED, RED_BG, RED_TX = "#dc2626", "#fef2f2", "#991b1b"


def zone(x, y, w, h, title):
    rect(x, y, w, h, "#e2e8f0", "#f8fafc", 1.5)
    text(x + 20, y + 14, w - 40, title, 13, GRAY, "left")


def save(path):
    json.dump({"elements": E}, open(path, "w"))
    return len(E)
