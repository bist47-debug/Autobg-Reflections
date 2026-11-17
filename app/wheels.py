"""
wheels.py - tyre reflections from the REAL wheels of the cut-out.

  1. find every visible wheel with a wheel detector (OWLv2, "car wheel" / "car tire")
  2. cut each wheel out exactly with SAM (box prompt) -> exact shape and width, nothing guessed
  3. reflection = the wheel's own pixels mirrored below its lowest point (contact), same exact shape,
     fading linearly from the contact point; never drawn over the car itself

First use downloads OWLv2 (~0.6 GB) and SAM ViT-H (~2.5 GB) from Hugging Face, once.
"""
import numpy as np, cv2, torch
from PIL import Image

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OWL = "google/owlv2-base-patch16-ensemble"
SAM = "facebook/sam-vit-huge"
QUERIES = ["a photo of a car wheel", "a photo of a car tire"]
_M = {}


def _models():
    if not _M:
        from transformers import Owlv2Processor, Owlv2ForObjectDetection, SamModel, SamProcessor
        _M["op"] = Owlv2Processor.from_pretrained(OWL)
        _M["om"] = Owlv2ForObjectDetection.from_pretrained(OWL).to(DEV).eval()
        _M["sp"] = SamProcessor.from_pretrained(SAM)
        _M["sm"] = SamModel.from_pretrained(SAM).to(DEV).eval()
    return _M


def _yolo():
    """your own trained tyre finder (train_tyres.py -> models/tyres.pt); None until it exists"""
    if "yolo" not in _M:
        import os
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "tyres.pt")
        if os.path.exists(p):
            from ultralytics import YOLO
            _M["yolo"] = YOLO(p)
        else:
            _M["yolo"] = None
    return _M["yolo"]


def _nms(b, s, iou=0.3):
    order, keep = np.argsort(-s), []
    while len(order):
        i = order[0]; keep.append(i)
        xx0 = np.maximum(b[i, 0], b[order[1:], 0]); yy0 = np.maximum(b[i, 1], b[order[1:], 1])
        xx1 = np.minimum(b[i, 2], b[order[1:], 2]); yy1 = np.minimum(b[i, 3], b[order[1:], 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        ar = lambda k: (b[k, 2] - b[k, 0]) * (b[k, 3] - b[k, 1])
        small = np.minimum(ar(i), ar(order[1:]))
        # drop boxes that overlap a lot OR sit mostly inside the kept one
        order = order[1:][(inter / (ar(i) + ar(order[1:]) - inter + 1e-6) < iou) & (inter / (small + 1e-6) < 0.7)]
    return keep


@torch.no_grad()
def detect(car, thr=0.08, max_side=1024, raw=False):
    """car: BGRA uint8 (any size). Returns a list of wheels:
    dict(box=(X0,Y0,X1,Y1), m=bool mask of the box, top, yb=contact row, near=bool, score)."""
    a = car[..., 3]
    ys, xs = np.where(a > 128)
    if len(xs) < 50:
        return []
    H_, W_ = a.shape
    bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bw, bh = bx1 - bx0, by1 - by0
    yp = _yolo()
    if yp is not None:                                  # trained tyre model: use it, nothing else
        f_ = car.astype(np.float32); al_ = f_[..., 3:] / 255
        bgr = np.clip(f_[..., :3] * al_ + 128 * (1 - al_), 0, 255).astype(np.uint8)
        # crop to the car first (like the training images): on the full backdrop canvas the car is small after
        # the model's 1024 px resize and small / half-hidden tyres were missed
        pc = int(0.05 * max(bw, bh))
        ox, oy = max(0, bx0 - pc), max(0, by0 - pc)
        bgr = np.ascontiguousarray(bgr[oy:min(H_, by1 + pc), ox:min(W_, bx1 + pc)])
        out = []
        # 1024 px first; if fewer than 2 tyres were found (tyres hidden behind a bumper, small corners only),
        # look again at 1280 px and add what is new
        polys = []
        for sz_ in (1024, 1280):
            if sz_ == 1280 and len(polys) >= 2:
                break
            r = yp.predict(bgr, imgsz=sz_, conf=0.35, retina_masks=True, verbose=False)[0]
            if r.masks is None:
                continue
            for poly, s_, bb in zip(r.masks.xy, r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
                if any(min(bb[2], q[2][2]) - max(bb[0], q[2][0]) > 0.3 * (bb[2] - bb[0]) for q in polys):
                    continue                                 # same tyre as one already found
                polys.append((poly, s_, bb))
        if polys:
            for poly, s_, _ in polys:
                # smooth outline: the model's polygon drawn anti-aliased at full resolution (the pixel mask is
                # blocky), times the cut-out's own anti-aliased alpha
                if len(poly) < 3:
                    continue
                cv_ = np.zeros((H_, W_), np.uint8)
                cv2.fillPoly(cv_, [np.round((poly + [ox, oy]) * 16).astype(np.int32)], 255, lineType=cv2.LINE_AA, shift=4)
                ma = (cv_.astype(np.float32) / 255) * (a.astype(np.float32) / 255)
                mk = ma > 0.5
                if mk.sum() < 30:
                    continue
                n_, lab_, st_, _ = cv2.connectedComponentsWithStats(mk.astype(np.uint8), 8)
                mk = lab_ == 1 + np.argmax(st_[1:, cv2.CC_STAT_AREA])
                ma = ma * cv2.dilate(mk.astype(np.uint8), np.ones((3, 3), np.uint8))
                rr, cc = np.where(mk.any(1))[0], np.where(mk.any(0))[0]
                X0, Y0, X1, Y1 = cc.min(), rr.min(), cc.max() + 1, rr.max() + 1
                out.append(dict(box=(X0, Y0, X1, Y1), m=mk[Y0:Y1, X0:X1], ma=ma[Y0:Y1, X0:X1], obox=np.array([X0, Y0, X1, Y1], float),
                                top=Y0, yb=Y1 - 1, score=float(s_)))
        if out:
            # drop flat strips under the bumper (tow eye, diffuser, sensor): a real far tyre is never much wider
            # than tall while being tiny next to the main tyre
            hm = max(w["box"][3] - w["box"][1] for w in out)
            out = [w for w in out if not ((w["box"][3] - w["box"][1]) < 0.2 * hm
                                          and (w["box"][2] - w["box"][0]) > 3.0 * (w["box"][3] - w["box"][1]))]
        return out if raw else _complete(out, car, (bx0, by0, bx1, by1))
    p = int(0.03 * max(bw, bh))
    cx0, cy0, cx1, cy1 = max(0, bx0 - p), max(0, by0 - p), min(W_, bx1 + p), min(H_, by1 + p)
    crop = car[cy0:cy1, cx0:cx1].astype(np.float32)
    al = crop[..., 3:] / 255
    rgb = crop[..., 2::-1] * al + 128 * (1 - al)                       # BGR -> RGB on mid grey
    sc = min(1.0, max_side / max(rgb.shape[:2]))
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    if sc < 1:
        img = img.resize((round(img.width * sc), round(img.height * sc)), Image.LANCZOS)
    M = _models()
    # ---- detector
    inp = M["op"](text=[QUERIES], images=img, return_tensors="pt").to(DEV)
    o = M["om"](**inp)
    S = max(img.size)                                                   # OWLv2 pads to a square
    score = o.logits[0].sigmoid().max(-1).values.float().cpu().numpy()
    cb = o.pred_boxes[0].float().cpu().numpy() * S
    box = np.stack([cb[:, 0] - cb[:, 2] / 2, cb[:, 1] - cb[:, 3] / 2, cb[:, 0] + cb[:, 2] / 2, cb[:, 1] + cb[:, 3] / 2], 1)
    t = max(thr, 0.2 * score.max())
    k = score > t
    box, score = box[k], score[k]
    # plausible wheels: inside the car, in its lower part, sensible size
    iw, ih = img.size
    cw, ch = bw * sc, bh * sc
    ok = ((box[:, 2] - box[:, 0]) > 0.03 * cw) & ((box[:, 2] - box[:, 0]) < 0.5 * cw) \
        & ((box[:, 3] + box[:, 1]) / 2 > ih - p * sc - 0.75 * ch) & (box[:, 0] > -5) & (box[:, 2] < iw + 5)
    box, score = box[ok], score[ok]
    if not len(box):
        return []
    keep = _nms(box, score)
    box, score = box[keep], score[keep]
    box0 = box.copy()                                   # the detector's own tight wheel box
    gx, gy = 0.06 * (box[:, 2] - box[:, 0]), 0.06 * (box[:, 3] - box[:, 1])
    box = np.clip(box + np.stack([-gx, -gy, gx, gy], 1), 0, [iw, ih, iw, ih])
    # ---- exact wheel masks
    si = M["sp"](img, input_boxes=[[list(map(float, b)) for b in box]], return_tensors="pt").to(DEV)
    so = M["sm"](**si, multimask_output=False)
    ms = M["sp"].image_processor.post_process_masks(so.pred_masks.cpu(), si["original_sizes"].cpu(),
                                                     si["reshaped_input_sizes"].cpu())[0][:, 0].numpy()
    aa = a[cy0:cy1, cx0:cx1] > 128
    out = []
    for b, b0, s, m in zip(box, box0, score, ms):
        m = cv2.resize(m.astype(np.uint8), (cx1 - cx0, cy1 - cy0), interpolation=cv2.INTER_NEAREST) > 0
        # stay inside the detector box (+4 %), and on the car
        X0, Y0, X1, Y1 = (b / sc).astype(int)
        g = int(0.03 * (X1 - X0)) + 2
        X0, Y0, X1, Y1 = max(0, X0 - g), max(0, Y0 - g), min(m.shape[1], X1 + g), min(m.shape[0], Y1 + g)
        mm = (m & aa)[Y0:Y1, X0:X1].astype(np.uint8)
        if mm.sum() < 30:
            continue
        n, lab, st, _ = cv2.connectedComponentsWithStats(mm, 8)
        mm = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        # fill holes (spokes / rim gaps)
        ff = np.pad(1 - mm, 1, constant_values=1).astype(np.uint8)
        cv2.floodFill(ff, None, (0, 0), 0)
        mm = (mm | ff[1:-1, 1:-1]).astype(bool)
        rows = np.where(mm.any(1))[0]
        cc = np.where(mm.any(0))[0]
        if cc.max() - cc.min() > 2.5 * (rows.max() - rows.min() + 1):    # long flat strip = bumper trim, not a tyre
            continue
        r1 = rows.max()
        cols = np.where(mm[max(0, r1 - 2):r1 + 1].any(0))[0]
        yp = min(H_ - 1, cy0 + Y0 + r1 + int(0.02 * bh) + 2)
        if (a[yp, cx0 + X0 + cols] < 128).mean() < 0.5:
            continue
        ob = b0 / sc + [cx0, cy0, cx0, cy0]
        bx_ = b0 / sc
        if mm.sum() < 0.35 * max(1.0, (bx_[2] - bx_[0]) * (bx_[3] - bx_[1])):   # thin strip (exhaust, trim)
            continue
        out.append(dict(box=(cx0 + X0, cy0 + Y0, cx0 + X1, cy0 + Y1), m=mm, obox=ob,
                        top=cy0 + Y0 + rows.min(), yb=cy0 + Y0 + rows.max(), score=float(s)))
    if raw:                                             # plain detector + SAM masks (for labelling)
        return out
    return _complete(out, car, (bx0, by0, bx1, by1))


def _complete(out, car, cb):
    """Reflection shape = exactly the VISIBLE tyre (its mask), nothing invented: a tyre half hidden by the
    bumper reflects only its visible part, as in reality. Fade length is the same for every tyre of the
    car (based on the biggest wheel), so all reflections fade evenly. Contact line = straight line along the
    tyre's bottom edge (slanted in 3/4 views), so the mirror follows the tyre's perspective."""
    bx0, by0, bx1, by1 = cb
    bh = by1 - by0
    a = car[..., 3] > 128
    if not out:
        return []
    D = max(w["yb"] - w["top"] + 1 for w in out)          # biggest wheel height in this photo
    low = max(w["yb"] for w in out)
    res = []
    for w in out:
        X0, Y0, X1, Y1 = [int(v) for v in w["box"]]
        M = w["m"].astype(bool)
        w = dict(w)
        # side-on tyre: replace the model's slightly lumpy outline by a smooth ellipse fitted to the tyre's
        # lower part (always visible), drawn anti-aliased -> clean, round reflection edge
        if "ma" in w and 0.75 < (X1 - X0) / max(1, Y1 - Y0) < 1.35:
            cs, _ = cv2.findContours(M.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cs:
                c = max(cs, key=cv2.contourArea)[:, 0, :]
                lower = c[c[:, 1] > 0.35 * M.shape[0]].astype(np.float32)
                if len(lower) >= 20:
                    (ex, ey), (ea, eb), ang = cv2.fitEllipse(lower)
                    big = max(X1 - X0, Y1 - Y0)
                    if 0.75 * big < max(ea, eb) < 1.35 * big and min(ea, eb) > 0.5 * big:
                        pad = 6
                        Hb, Wb = (Y1 - Y0) + 2 * pad, (X1 - X0) + 2 * pad
                        cv_ = np.zeros((Hb, Wb), np.uint8)
                        cv2.ellipse(cv_, (int(round((ex + pad) * 16)), int(round((ey + pad) * 16))),
                                    (int(round(ea / 2 * 16)), int(round(eb / 2 * 16))), ang, 0, 360, 255, -1,
                                    cv2.LINE_AA, 4)
                        ell_ = cv_.astype(np.float32) / 255
                        # keep it on the tyre: only where the car is (no ellipse sticking out under the floor line)
                        H_, W_ = car.shape[:2]
                        y0_, x0_ = Y0 - pad, X0 - pad
                        sub = np.zeros((Hb, Wb), np.float32)
                        ys0, xs0 = max(0, y0_), max(0, x0_)
                        ys1, xs1 = min(H_, y0_ + Hb), min(W_, x0_ + Wb)
                        sub[ys0 - y0_:ys1 - y0_, xs0 - x0_:xs1 - x0_] = car[ys0:ys1, xs0:xs1, 3].astype(np.float32) / 255
                        ma = ell_ * sub
                        X0, Y0, X1, Y1 = x0_, y0_, x0_ + Wb, y0_ + Hb
                        M = ma > 0.5
                        w.update(box=(X0, Y0, X1, Y1), ma=ma)
        yb = int(w["yb"])
        h = yb - int(w["top"]) + 1
        bp = np.full(M.shape[1], -1)
        has = M.any(0)
        bp[has] = M.shape[0] - 1 - np.argmax(M[::-1, has], axis=0)
        xs_ = np.where(has & (bp >= (yb - Y0) - max(2, 0.12 * h)))[0]
        slope = 0.0
        round_ = 0.8 < (X1 - X0) / max(1, Y1 - Y0) < 1.25         # side-on tyre: contact line is level
        if not round_ and len(xs_) >= 4 and xs_.max() - xs_.min() > 0.1 * M.shape[1]:
            slope = float(np.clip(np.polyfit(xs_ + X0, bp[xs_] + Y0, 1)[0], -0.35, 0.35))
        cx = (X0 + X1) / 2.0
        cols_ = np.where(has)[0]
        yref = float(max([bp[c] + Y0 - slope * (c + X0 - cx) for c in cols_], default=yb))
        rub = car[Y0:Y1, X0:X1, :3][M & (car[Y0:Y1, X0:X1, :3].mean(-1) < 70)]
        col = np.median(rub, 0) if len(rub) > 20 else np.array([25.0, 25.0, 25.0])
        res.append(dict(w, m=M, hm=np.zeros_like(M), full=M, top=int(yb - D + 1), yb=yb,
                        near=yb >= low - 0.05 * bh, col=col,
                        ell=(cx, (Y0 + Y1) / 2.0, (X1 - X0) / 2.0, (Y1 - Y0) / 2.0), line=(slope, cx, yref)))
    return sorted(res, key=lambda w: w["yb"])


def _ghosts(out, a, cb, D):
    """3/4 views: the far-side wheel that is (almost) hidden behind the body still shows faintly in the
    reflection. Its place is estimated from the visible wheels (parallelogram of the 4 contact points)."""
    bx0, by0, bx1, by1 = cb
    bw, bh = bx1 - bx0, by1 - by0
    if len(out) < 2 or len(out) > 3:
        return []
    c = lambda w: ((w["box"][0] + w["box"][2]) / 2.0, float(w["yb"]))
    ws = sorted(out, key=lambda w: -w["yb"])            # lowest (closest) first
    A, B = ws[0], ws[1]
    (ax, ay), (bxx, byy) = c(A), c(B)
    if abs(ay - byy) < 0.03 * bh:                        # side view: far wheels sit exactly behind
        return []
    if len(out) == 3:                                    # 3 visible -> 4th by parallelogram
        C = ws[2]
        cx_, cy_ = c(C)
        near = min((A, B), key=lambda w: abs(c(w)[0] - cx_))
        other = B if near is A else A
        gx, gy = cx_ + (c(other)[0] - c(near)[0]), cy_ + (c(other)[1] - c(near)[1])
    else:                                                # 2 visible: far wheel of the closest axle
        side = -1 if bxx > ax else 1                     # the car end beyond A, away from B
        edge = bx0 if side < 0 else bx1
        gx = ax + 0.6 * (edge - ax)
        gy = ay - 0.07 * bh
    gx, gy = int(np.clip(gx, bx0, bx1 - 1)), int(np.clip(gy, by0, by1 - 1))
    hw = int(0.3 * D * 0.8)
    if any(abs(c(w)[0] - gx) < hw for w in out):         # a visible wheel is already there
        return []
    return [dict(box=(gx - hw, gy, gx + hw + 1, gy + 1), m=np.zeros((1, 2 * hw + 1), bool), top=gy, yb=gy,
                 score=0.0, ghost=True)]


def scale(wheels, f):
    """wheels found at one resolution -> another (factor f)."""
    out = []
    for w in wheels:
        X0, Y0, X1, Y1 = [int(round(v * f)) for v in w["box"]]
        sz = (max(1, X1 - X0), max(1, Y1 - Y0))
        rs = lambda k: cv2.resize(w[k].astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST) > 0
        out.append(dict(w, box=(X0, Y0, X1, Y1), m=rs("m"), hm=rs("hm"), full=rs("full"), top=int(round(w["top"] * f)),
                        yb=int(round(w["yb"] * f)), ell=tuple(v * f for v in w["ell"]),
                        line=(w["line"][0], w["line"][1] * f, w["line"][2] * f)))
        if "ma" in w:
            out[-1]["ma"] = cv2.resize(w["ma"].astype(np.float32), sz, interpolation=cv2.INTER_AREA)
    return out


def flip(base, car, wheels, strength=0.6, length=0.55, fade=1.0, blur=0.0, far=1.0, hidden=1.0, detail=0.1, shift=0.0, contact=0.4, gloss=0.0, darken=False, hug=False, edge=0.0):
    """base: BGR float image the car sits in (same size as car). Each wheel's reflection = its real
    silhouette (+ the hidden part of its ellipse), mirrored about the tyre's own contact line (slanted in
    perspective views), dark tyre tone with a little real detail near the tyre. Linear fade over
    length x wheel height. Never drawn over the car itself."""
    H_, W_ = base.shape[:2]
    out = base.astype(np.float32).copy()
    rgb = car[..., :3].astype(np.float32)
    if blur > 0:
        rgb = cv2.GaussianBlur(rgb, (0, 0), blur)
    a = car[..., 3].astype(np.float32) / 255
    for w in wheels:
        wt = strength * (1.0 if w["near"] else far)
        if wt <= 0:
            continue
        X0, Y0, X1, Y1 = [int(v) for v in w["box"]]
        th = max(2, int(w["yb"]) - int(w["top"]) + 1)
        slope, lx, ly = w.get("line", (0.0, (X0 + X1) / 2.0, float(w["yb"])))
        # snap the mirror line to the tyre's REAL bottom edge in THIS image (the tyre model works on a
        # smaller copy, and the cut-out edge may have been tightened -> a few px gap otherwise)
        mb = w["m"]
        ly0 = ly
        if mb.any():
            rr = np.where(mb.any(1))[0].max()
            cols = np.where(mb[max(0, rr - max(2, int(0.06 * th))):rr + 1].any(0))[0] + X0
            tol = max(3, int(0.08 * th))
            snaps = []
            for x in cols:
                col = np.where(a[max(0, int(w["yb"]) - tol):min(H_, int(w["yb"]) + tol + 1), x] > 0.5)[0]
                if len(col):
                    snaps.append(max(0, int(w["yb"]) - tol) + col.max() - slope * (x - lx))
            if snaps:
                ly = float(max(snaps))
        moff = ly - ly0                                      # the tyre's real bottom is lower than its mask's:
        ly_true = ly
        ly -= float(shift) + float(w.get("lift", 0.0))       # manual fine-tune: + = reflection up
        MX0 = X0                                             # mask origin (the tyre's box)
        sxs = max(1.0, float(w.get("soft_sides") or 0) * (X1 - X0))
        if w.get("soft_sides"):
            # room beside the tyre so the softened sides are not clipped by its box (hard vertical edge)
            pd_ = int(3 * sxs) + 2
            X0, X1 = max(0, X0 - pd_), min(W_, X1 + pd_)
        xs = np.arange(X0, X1, dtype=np.float32)
        yl = ly + slope * (xs - lx)                          # contact line per column
        ov = max(3, int(0.02 * th))                          # start a few px UNDER the tyre's soft edge
        Lf = float(w["flen"]) if w.get("flen") else length * th     # fixed fade length in px (front/back) or x wheel height
        # car placed low in the frame: the fade must end BEFORE the picture's bottom edge (it ran out of the
        # frame at full strength -> a hard cut at the bottom)
        Lf = max(2.0, min(Lf, 0.9 * (H_ - float(yl.max()))))
        R0, R1 = int(np.floor(yl.min())) + 1 - ov, int(min(H_, np.ceil(yl.max()) + max(th, Lf) + 1))
        if R1 <= R0 or X1 <= X0:
            continue
        yy = np.arange(R0, R1, dtype=np.float32)[:, None]
        dep = yy - yl[None, :]                               # depth below the contact line
        ym = yl
        if hug:
            # front/back: mirror each column about the tyre's OWN bottom in that column, so the reflection
            # hugs the rounded tyre corners (full width at once, no square block beside the corner, no gap)
            ym = yl.copy()
            lo = max(0, int(yl.min()) - int(0.35 * th))
            sub = a[lo:R1, X0:X1] > (0.05 if w.get("hug_lo") else 0.5)
            for j in range(sub.shape[1]):
                r_ = np.where(sub[:, j])[0]
                if len(r_):
                    yb_ = lo + r_.max() + 1.0
                    if yb_ < yl[j]:
                        ym[j] = yb_
            R0 = min(R0, int(np.floor(ym.min())) + 1 - ov)
            yy = np.arange(R0, R1, dtype=np.float32)[:, None]
            dep = yy - yl[None, :]
        depm = yy - ym[None, :]
        sy = (ym[None, :] - np.maximum(depm, 0.5) / float(w.get("vstretch", 1.0))).astype(np.float32)   # vstretch: stretched vertically only   # mirrored source row
        sx = np.broadcast_to(xs[None, :], sy.shape).astype(np.float32)
        my = (sy - Y0 - moff).astype(np.float32)             # shape sampled with the mask moved onto it
        msrc = w["ma"].astype(np.float32) if "ma" in w else w["m"].astype(np.float32)
        vis = cv2.remap(msrc, (sx - MX0).astype(np.float32), my, cv2.INTER_LINEAR, borderValue=0)
        full = vis if "ma" in w else cv2.remap(w["full"].astype(np.float32), (sx - MX0).astype(np.float32), my, cv2.INTER_LINEAR, borderValue=0)
        below = (depm > -ov).astype(np.float32)
        shape = (vis + hidden * np.clip(full - vis, 0, 1)) * below
        shape = cv2.GaussianBlur(shape, (0, 0), max(0.6, edge * th))   # outline is already anti-aliased; edge: softer outline (x wheel height)
        if w.get("soft_sides"):
            # front/back: sides get softer with depth, like a real floor reflection (no straight hard side edges)
            sb = cv2.GaussianBlur(shape, (0, 0), sigmaX=sxs, sigmaY=0.5)
            tt = np.clip((0.0 if w.get("clip_contact") else 0.7) + (dep - (ly_true - ly) if w.get("clip_contact") else dep) / max(2.0, Lf), 0, 1)
            shp0 = shape
            shape = shape * (1 - tt) + sb * tt
            if w.get("clip_contact"):
                # above the real floor line (the corners beside the round tyre bottom): crisp, no soft spread
                # sideways, so the corners fill solid and nothing shows beside the tyre
                above_ = (dep - (ly_true - ly)) < 0
                # ... and only in columns where the tyre itself is right above (not beside a mudflap / the body)
                yt0_ = ly_true + slope * (xs - lx)
                lf_ = int(max(2, ly_true - ly))
                colok = np.zeros(len(xs), np.float32)
                for j_ in range(len(xs)):
                    y1_ = int(yt0_[j_]); y0_ = max(0, y1_ - lf_)
                    if y1_ > y0_ and (a[y0_:y1_, int(xs[j_])] > 0.5).any():
                        colok[j_] = 1.0
                colok = cv2.GaussianBlur(colok[None], (0, 0), 1.5)[0]
                shape = np.where(above_, shp0 * colok[None, :], shape)
                # below the floor line, columns outside the tyre's bottom (its round corners) fade in over the
                # lift height: the reflection's top corners mirror the tyre's round corners (no square step)
                dfp_ = np.maximum(dep - (ly_true - ly), 0)
                shape = np.where(above_, shape, shape * np.clip(colok[None, :] + dfp_ / max(2.0, lf_), 0, 1))
        col = np.broadcast_to(w["col"].astype(np.float32), shape.shape + (3,)).copy()
        if detail > 0:
            patch = cv2.remap(rgb, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            ramp = np.clip(1 - depm / (0.5 * (Lf if w.get("flen") else th)), 0, 1)       # detail only near the tyre, gone halfway
            d = (detail * ramp * cv2.GaussianBlur(vis, (0, 0), 1.0))[..., None]
            col = col * (1 - d) + patch * d
        if gloss > 0:
            # glossy floor: the real tyre + rim mirrored over the whole reflection (fades out with it)
            patch = cv2.remap(rgb, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            g = (gloss * cv2.GaussianBlur(vis, (0, 0), 1.0))[..., None]
            col = col * (1 - g) + patch * g
        dfd = dep - (ly_true - ly) if w.get("clip_contact") else dep   # lifted: fade counts from the REAL contact
        lin = np.clip(1 - dfd / max(2.0, Lf), 0, 1) ** fade
        solid = np.clip((a[R0:R1, X0:X1] - 0.9) / 0.1, 0, 1)      # only the SOLID car blocks it: under the
        # right at the contact the floor mirrors the tyre at full darkness (no light seam between tyre and
        # reflection); it eases to the chosen strength over the first few % of the wheel height
        # only a thin line: under the cut-out's soft edge and ~2 px below it (was 6 % of the wheel -> black patch)
        near_c = np.clip(1 - (depm + ov) / (ov + 2.0), 0, 1) * contact
        wt_ = wt + (1 - wt) * near_c
        if w.get("start_full"):
            # continuous with the tyre: the reflection starts as dark as the tyre at the floor line and eases to the
            # set strength over the first third of the fade (no lighter band / shade step under the tyre)
            dfd_ = dep - (ly_true - ly) if w.get("clip_contact") else dep
            sf_ = np.clip(1 - np.maximum(dfd_, 0) / max(2.0, 0.33 * Lf), 0, 1) * float(w["start_full"])
            wt_ = np.maximum(wt_, wt + (1 - wt) * sf_)
        if w.get("clip_contact"):
            # lifted reflection: nothing above the real floor contact (no block beside the rounded tyre corners)
            yt_ = ly_true + slope * (xs - lx)
            G_ = max(1.0, float(w.get("clip_soft", 1.0)))
            shape = shape * np.clip((yy - yt_[None, :] + G_) / G_, 0, 1)
        k = (wt_ * shape * lin * (1 - solid))[..., None]           # soft cut-out edge the reflection continues (no light rim)
        new_ = out[R0:R1, X0:X1] * (1 - k) + col * k
        if darken:
            # never lighter than what is already there (e.g. the AutoBG shadow next to the tyre): a lighter patch
            # inside the dark shadow read as a cut-off reflection
            new_ = np.minimum(out[R0:R1, X0:X1], new_)
        out[R0:R1, X0:X1] = new_
    return out


def tyre_column(base, car, wheels, strength=0.25, length=1.1, soft=0.02, spread=0.05, up=8, equal=False, ellipse=False, curve=False, mirror=True, stretch=1.0):
    """front / back views (11-16 reference look): each tyre's reflection is a plain, matte, slightly darker
    column of floor EXACTLY the tyre's width - no tread / rim detail. Its top corners are the tyre's own round
    bottom corners mirrored (read from the cut-out in this photo), the sides are soft and get a little softer
    with depth, and it fades linearly over `length` x tyre width - always ending inside the picture."""
    H_, W_ = base.shape[:2]
    out = base.astype(np.float32).copy()
    a = car[..., 3].astype(np.float32) / 255
    # pass 1: measure each tyre at the floor (width + its round corner profile)
    meas = []
    for w in wheels:
        X0, Y0, X1, Y1 = [int(v) for v in w["box"]]
        X0, X1 = max(0, X0), min(W_, X1)
        yb = int(w["yb"])
        th = max(2, yb - int(w["top"]) + 1)
        tol = max(3, int(0.1 * th))
        lo, hi = max(0, Y0), min(H_, yb + tol + 1)
        if X1 - X0 < 3 or hi <= lo:
            continue
        sub = a[lo:hi, X0:X1] > 0.5
        has = sub.any(0)
        if not has.any():
            continue
        bot = np.where(has, lo + sub.shape[0] - 1 - np.argmax(sub[::-1], 0), -1)   # lowest solid row per column
        cyr = int(bot.max())                                  # the tyre's REAL floor contact
        g = (cyr - bot).astype(np.float32)                    # corner depth per column
        lim = max(3.0, 0.35 * th)                             # deeper than the tyre's round corner = not the tyre
        inside = has & (g <= lim)
        cols = np.where(inside)[0]
        if len(cols) < 3:
            continue
        c0, c1 = cols.min(), cols.max()
        prof_g = np.where(inside[c0:c1 + 1], g[c0:c1 + 1], lim)      # gaps inside the width filled
        mc_ = np.where(np.asarray(w["m"]).any(0))[0]               # the tyre's FULL width (its mask), for ellipse mode
        fw, fcen = (float(mc_.max() - mc_.min() + 1), w["box"][0] + (mc_.min() + mc_.max()) / 2.0) if len(mc_) else (float(c1 - c0 + 1), X0 + (c0 + c1) / 2.0)
        # the tyre's silhouette from its floor contact upwards: half width (x half the full tyre width) per height
        # (x tyre height), never narrowing upwards - mirrored, it is the exact outline of the reflection (11-16 reference)
        hp = np.zeros(101, np.float32)
        mm_ = np.asarray(w["m"])
        if mm_.any() and len(mc_):
            rws_ = np.where(mm_.any(1))[0]
            rb_ = rws_.max()
            hs_, ws_ = [], []
            for r_ in range(rb_, rws_.min() - 1, -1):
                cc_ = np.where(mm_[r_])[0]
                if len(cc_):
                    hs_.append((rb_ - r_) / float(th)); ws_.append((cc_.max() - cc_.min() + 1) / max(1.0, fw))
            if hs_:
                hp = np.interp(np.linspace(0, 1, 101), hs_, np.maximum.accumulate(ws_)).astype(np.float32)
        meas.append(dict(w=w, cy=cyr - int(up), cen=X0 + (c0 + c1) / 2.0, tw=float(c1 - c0 + 1), g=prof_g, fw=fw, fcen=fcen, th=th, hp=hp))
    if not meas:
        return out
    # 11-16: every tyre's reflection gets the SAME width (the widest tyre's), centred under its own tyre; the
    # round corner profile is stretched to that width
    # (equal=True: every tyre gets the widest tyre's width - tried 11-16, back to each tyre's own width)
    tw_eq = max(m_["tw"] for m_ in meas)
    for m_ in meas:
        w, cy = m_["w"], m_["cy"]
        tw = tw_eq if equal else m_["tw"]
        if ellipse or (mirror and m_["hp"].max() > 0):
            tw = m_["fw"]
        # side shots: both tyres get the SAME fade length and tone (the bigger / nearer tyre's own size made its
        # reflection longer and more visible than the other one)
        twL = float(np.mean([q["fw"] if ellipse else q["tw"] for q in meas]))   # same fade length for every tyre
        L = max(4.0, min(length * twL, 0.9 * (H_ - cy)))
        sx = soft * tw + 1.0
        pad = int(3 * (sx + spread * tw)) + 2
        cen_ = m_["fcen"] if (ellipse or (mirror and m_["hp"].max() > 0)) else m_["cen"]
        xl, xr = cen_ - (tw - 1) / 2.0, cen_ + (tw - 1) / 2.0
        xa, xb = max(0, int(np.floor(xl)) - pad), min(W_, int(np.ceil(xr)) + 1 + pad)
        ya, yb2 = max(0, cy - 1), min(H_, int(cy + L) + 2)
        if yb2 <= ya or xb <= xa:
            continue
        yy = np.arange(ya, yb2, dtype=np.float32)[:, None]
        dep = yy - cy                                         # depth below the (lifted) contact row
        xs_ = np.arange(xa, xb, dtype=np.float32)
        ok_ = (xs_ >= xl - 0.5) & (xs_ <= xr + 0.5)
        # one SHARED corner shape for all tyres (each tyre's own, partly hidden by bumper / mudflap, made the
        # reflections look different from each other): mean of all tyres' profiles, made symmetric
        u101 = np.linspace(0, 1, 101)
        gsh = np.mean([np.interp(u101, np.linspace(0, 1, len(q["g"])), q["g"]) for q in meas], 0)
        gsh = 0.5 * (gsh + gsh[::-1])
        src = np.interp(np.clip((xs_ - xl) / max(1.0, tw - 1), 0, 1), u101, gsh)
        gj = np.where(ok_, src, np.inf).astype(np.float32)    # where each column's reflection starts (mirrored corner)
        M = np.clip(dep - gj[None, :] + 1.0, 0, 1)             # anti-aliased top edge = the tyre's corners mirrored
        if mirror and not ellipse and all(q["hp"].max() > 0 for q in meas):
            # front / back (11-16 reference): the reflection is the tyre's own silhouette mirrored - its width at each
            # depth = the tyre's width at the same height above the floor (one shared, symmetric profile for all tyres,
            # each scaled to its own tyre), so it widens slightly from the contact patch like the real tyre does
            hpm = np.mean([q["hp"] for q in meas], 0)
            fw_, fc_ = m_["fw"], m_["fcen"]
            dr_ = np.maximum(dep, 0)                            # depth below the LIFTED contact (reflection pushed up `up` px)
            hw_ = 0.5 * fw_ * np.interp(dr_ / (2.0 * float(m_["th"])), np.linspace(0, 1, 101), hpm)   # mirrored x2 stretched (perspective, as in the reference)
            M = np.clip(hw_ - np.abs(xs_[None, :] - fc_) + 0.5, 0, 1) * np.clip(dep + 1.0, 0, 1)
        if ellipse:
            # side shots: the round tyre mirrored and STRETCHED down into a long smooth oval (a straight column under a
            # round tyre gave square shoulders - edges sticking out). The oval is longer than the fade, so it fades out
            # before its own bottom; its top is the tyre's round bottom mirrored (stretched)
            rx = tw / 2.0
            ry = stretch * max(float(np.mean([q["th"] for q in meas])) / 2.0, 0.6 * L)   # stretch < 1: flatter oval
            u = (xs_[None, :] - cen_) / rx
            v = (dep - ry) / ry
            e = np.sqrt(u ** 2 + v ** 2)
            M = np.clip((1.0 - e) * min(rx, ry) + 0.5, 0, 1)    # anti-aliased oval edge
        elif curve:
            # front / back (11-16 evening, then switched off - back to the straight reference look): the column narrows as it goes down, a gently curved U instead of straight
            # sides: half width x sqrt(1 - (depth / (1.4 x fade))^2) -> ~70 % of the tyre width where the fade ends
            hw0 = tw / 2.0
            hwd = hw0 * np.sqrt(np.clip(1 - (np.maximum(dep, 0) / (1.4 * L)) ** 2, 0, 1))
            M = M * np.clip(hwd - np.abs(xs_[None, :] - cen_) + 0.5, 0, 1)
        Ms = cv2.GaussianBlur(M, (0, 0), sigmaX=sx, sigmaY=0.5, borderType=cv2.BORDER_CONSTANT)
        Mw = cv2.GaussianBlur(M, (0, 0), sigmaX=sx + spread * tw, sigmaY=0.5, borderType=cv2.BORDER_CONSTANT)
        t = np.clip(dep / L, 0, 1)
        M = Ms * (1 - t) + Mw * t                             # sides soften with depth
        prof = strength * np.clip(1 - dep / L, 0, 1)          # linear fade
        prof = np.maximum(prof, 0.6 * np.clip(1 - (dep - int(up) + 1) / 3.0, 0, 1))   # no light seam right under the tyre (at the REAL contact)
        k = (prof * M * (1 - np.clip((a[ya:yb2, xa:xb] - 0.9) / 0.1, 0, 1)))[..., None]
        col = np.asarray(w.get("col", (22, 22, 26)), np.float32).reshape(1, 1, 3)
        if True:                                       # same tone for every tyre (all views)
            col = np.mean([np.asarray(q["w"].get("col", (22, 22, 26)), np.float32) for q in meas], 0).reshape(1, 1, 3)
        out[ya:yb2, xa:xb] = out[ya:yb2, xa:xb] * (1 - k) + col * k
    return out


def debug(img, wheels):
    """green = near wheel, yellow = far wheel (visible part), blue = full wheel incl. hidden part, red = contact."""
    d = img.copy()
    for w in wheels:
        X0, Y0, X1, Y1 = w["box"]
        full = np.zeros(d.shape[:2], np.uint8)
        full[Y0:Y1, X0:X1] = w["m"]
        cs, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        col = (0, 255, 0) if w["near"] else (0, 220, 255)
        cv2.drawContours(d, cs, -1, col, max(2, d.shape[1] // 600))
        cx, cy, rx, ry = w["ell"]
        cv2.ellipse(d, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, (255, 80, 0), max(1, d.shape[1] // 900))
        sl, lx, ly = w["line"]
        cv2.line(d, (X0, int(ly + sl * (X0 - lx))), (X1, int(ly + sl * (X1 - lx))), (0, 0, 255), max(1, d.shape[1] // 900))
    return d


def body_strip(base, car, wheels, strength=0.08, length=0.6, edge=0.15):
    """side views: a faint mirror of the car's underside / sills BETWEEN the two main tyres only (no bumpers,
    no front / rear of the car). Mirrored about the line through the two tyres' floor contacts, fading out over
    `length` x wheel height, eased in / out near the tyres (`edge` x the gap)."""
    out = base.astype(np.float32).copy()
    if len(wheels) < 2:
        return out
    w2 = sorted(sorted(wheels, key=lambda q: -(q["box"][3] - q["box"][1]))[:2], key=lambda q: q["box"][0])
    (a0, _, a1, _), (b0, _, b1, _) = [[int(v) for v in q["box"]] for q in w2]
    xa, xb = a1, b0                                     # inner edges of the two tyres
    if xb - xa < 10:
        return out
    pa = ((a0 + a1) / 2.0, float(w2[0]["yb"]))
    pb = ((b0 + b1) / 2.0, float(w2[1]["yb"]))
    sl = (pb[1] - pa[1]) / max(1.0, pb[0] - pa[0])
    H_, W_ = out.shape[:2]
    th = max(q["box"][3] - q["box"][1] for q in w2)
    L = max(4.0, length * th)
    xs = np.arange(xa, xb, dtype=np.float32)
    yl = pa[1] + sl * (xs - pa[0])                      # floor contact line between the tyres
    R0, R1 = int(np.floor(yl.min())), int(min(H_, np.ceil(yl.max()) + L + 1))
    if R1 <= R0:
        return out
    yy = np.arange(R0, R1, dtype=np.float32)[:, None]
    dep = yy - yl[None, :]
    sy = (yl[None, :] - np.maximum(dep, 0.5)).astype(np.float32)
    sx = np.broadcast_to(xs[None, :], sy.shape).astype(np.float32)
    rgb = cv2.remap(car[..., :3].astype(np.float32), sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    al = cv2.remap(car[..., 3].astype(np.float32) / 255, sx, sy, cv2.INTER_LINEAR, borderValue=0)
    # how far below the contact line the mirrored body starts is real geometry; fade by depth
    fadev = np.clip(1 - dep / L, 0, 1) ** 1.5 * (dep > 0)
    n = xb - xa
    e = max(2.0, edge * n)
    t = np.arange(n, dtype=np.float32)
    fadeh = np.clip(np.minimum(t, n - 1 - t) / e, 0, 1)[None, :]
    solid = np.clip((car[R0:R1, xa:xb, 3].astype(np.float32) / 255 - 0.9) / 0.1, 0, 1)
    k = (strength * al * fadev * fadeh * (1 - solid))[..., None]
    out[R0:R1, xa:xb] = out[R0:R1, xa:xb] * (1 - k) + rgb * k
    return out


def mirror_wheel(w, cx):
    """the same tyre on the other side of a straight front / rear shot (mirrored about the car's centre
    column cx) - for when only one of the two tyres could be found"""
    X0, Y0, X1, Y1 = [int(v) for v in w["box"]]
    nx0, nx1 = int(round(2 * cx - X1)), int(round(2 * cx - X0))
    o = dict(w)
    o["box"] = (nx0, Y0, nx1, Y1)
    o["obox"] = np.array([nx0, Y0, nx1, Y1], float)
    for k in ("m", "ma", "hm", "full"):
        if k in w and hasattr(w[k], "shape"):
            o[k] = np.ascontiguousarray(w[k][:, ::-1])
    if "line" in w:
        sl, lx, ly = w["line"]
        o["line"] = (-sl, 2 * cx - lx, ly)
    if "ell" in w:
        ex, ey, ea, eb = w["ell"]
        o["ell"] = (2 * cx - ex, ey, ea, eb)
    o["mirrored"] = True
    return o
