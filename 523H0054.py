import cv2
import numpy as np
import math
import os


# =============================================================================
# 1. COLOR SEGMENTATION MASKS (HSV)
# =============================================================================

def create_red_mask(hsv):
    """Strict red mask for standalone red signs (No Entry, Caution, No Left Turn)."""
    lower1 = np.array([0, 80, 55])
    upper1 = np.array([11, 255, 255])
    lower2 = np.array([158, 80, 55])
    upper2 = np.array([179, 255, 255])
    return cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)


def create_red_rim_mask(hsv):
    """Looser red/magenta mask for thin red rims/slashes or sun-bleached red signs."""
    lower1 = np.array([0, 48, 50])
    upper1 = np.array([14, 255, 255])
    lower2 = np.array([142, 45, 50])
    upper2 = np.array([179, 255, 255])
    return cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)


def create_blue_mask(hsv):
    """Strict traffic-sign blue (suppresses sky, road, glass buildings, and tree shadows)."""
    return cv2.inRange(hsv, np.array([98, 125, 75]), np.array([126, 255, 255]))


def create_blue_bright_mask(hsv):
    """Bright illuminated traffic-sign blue (separates Keep Right from dark post shadows)."""
    return cv2.inRange(hsv, np.array([98, 120, 108]), np.array([125, 255, 255]))


def create_blue_np_mask(hsv):
    """Medium-saturation blue used for No Parking signs that also have a red outer rim."""
    return cv2.inRange(hsv, np.array([98, 90, 85]), np.array([125, 255, 255]))


def create_yellow_mask(hsv):
    """Golden-yellow interior of triangular warning signs."""
    return cv2.inRange(hsv, np.array([13, 105, 95]), np.array([32, 255, 255]))


# =============================================================================
# 2. GEOMETRIC & SPATIAL VALIDATION HELPERS
# =============================================================================

def _is_valid_street_pos(cx, cy, bw, bh, w, h):
    """
    Validate that a sign candidate lies in the centre median or right side
    of the street and above the road surface / horizon line.
    """
    if cy < h * 0.015:
        return False
    if cx < w * 0.62:
        if cy > h * 0.432:
            return False
    elif cx < w * 0.85:
        if cy > h * 0.442:
            return False
    elif cx < w * 0.93:
        if cy > h * 0.465:
            return False
    else:
        if cy > h * 0.395:
            return False

    if cy >= h * 0.28:
        min_cx = w * 0.415
    else:
        min_cx = max(w * 0.14, w * 0.415 - (h * 0.28 - cy) * 1.65)
    if bw > 65:
        min_cx = min(min_cx, max(w * 0.14, w * 0.415 - (bw - 65) * 2.8))
    if cx < min_cx or cx > w * 0.992:
        return False
    return True


def _disc_fill(mask, cx, cy, r, shrink=0.08):
    """Compute foreground fill ratio of a binary mask inside a circular disc."""
    x1, y1 = max(0, int(cx - r)), max(0, int(cy - r))
    x2, y2 = min(mask.shape[1], int(cx + r)), min(mask.shape[0], int(cy + r))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    disc = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.circle(disc, (int(cx) - x1, int(cy) - y1), max(2, int(r * (1.0 - shrink))), 255, -1)
    sub = mask[y1:y2, x1:x2]
    inside = cv2.countNonZero(cv2.bitwise_and(sub, disc))
    area = max(1, cv2.countNonZero(disc))
    return inside / float(area)


def hull_metrics(cnt):
    """Compute convex-hull geometric descriptors for a contour."""
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    hull_peri = cv2.arcLength(hull, True)
    if hull_area <= 0 or hull_peri <= 0:
        return None
    hull_circ = 4.0 * math.pi * hull_area / (hull_peri * hull_peri)
    (cx, cy), r = cv2.minEnclosingCircle(hull)
    bx, by, bw, bh = cv2.boundingRect(hull)
    if bw <= 0 or bh <= 0 or r <= 0:
        return None
    hull_extent = hull_area / float(bw * bh)
    hpts = hull.reshape(-1, 2).astype(np.float32)
    if hpts.shape[0] < 5:
        return None
    dists = np.linalg.norm(hpts - np.array([cx, cy]), axis=1)
    mean_d = dists.mean()
    if mean_d <= 0:
        return None
    rad_var = dists.std() / mean_d
    return hull_area, hull_circ, hull_extent, rad_var, cx, cy, r, bx, by, bw, bh


def is_circular_shape(hull_circ, hull_extent, rad_var, bw, bh):
    """Check whether convex-hull metrics match a circular traffic sign."""
    if bw <= 0 or bh <= 0:
        return False
    aspect = max(bw, bh) / float(min(bw, bh))
    if aspect > 1.28:
        return False
    if hull_circ < 0.82:
        return False
    if hull_extent < 0.65 or hull_extent > 0.845:
        return False
    if rad_var > 0.13:
        return False
    return True


# =============================================================================
# 3. CONTENT INFERENCE & MULTI-SHAPE SIGN CLASSIFICATION
# =============================================================================

def _classify_no_parking_vs_stopping(hsv, cx, cy, r):
    """
    Distinguish 'No Parking' (single main-diagonal red slash \\) from
    'No Stopping & Parking' (two crossed diagonal red slashes X).
    """
    h, w = hsv.shape[:2]
    x1, y1 = max(0, int(cx - r)), max(0, int(cy - r))
    x2, y2 = min(w, int(cx + r)), min(h, int(cy + r))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return "No Parking"
    roi = cv2.resize(hsv[y1:y2, x1:x2], (40, 40))
    cH, cS, cV = cv2.split(roi)
    yy, xx = np.mgrid[0:40, 0:40]
    nx, ny = (xx - 19.5) / 19.5, (yy - 19.5) / 19.5
    rad = np.sqrt(nx**2 + ny**2)
    inner = (rad < 0.54) & (rad > 0.10)
    anti = inner & (nx * ny < -0.03) & (np.abs(nx + ny) < 0.24)
    main = inner & (nx * ny > 0.03) & (np.abs(nx - ny) < 0.24)
    is_red = ((cH <= 18) | (cH >= 125)) & (cS >= 42)
    is_blue = (cH >= 96) & (cH <= 123) & (cS >= 75)
    ra = is_red[anti].mean() if anti.any() else 0.0
    rm = is_red[main].mean() if main.any() else 0.0
    ba = is_blue[anti].mean() if anti.any() else 1.0
    bm = is_blue[main].mean() if main.any() else 1.0
    if ra >= 0.15 and ba <= 0.68 and (ba - bm) <= 0.16 and ra >= 0.42 * rm:
        return "No Stopping & Parking"
    return "No Parking"


def verify_and_classify_circle(hsv, cx, cy, r, frame_w, allow_loose_np=True):
    """
    Verify radial color structure of a circular sign candidate and infer its
    specific traffic sign content ('No Left Turn', 'No Parking',
    'No Stopping & Parking', 'No Entry', or 'Keep Right').
    """
    h, w = hsv.shape[:2]
    x1, y1 = max(0, int(cx - r)), max(0, int(cy - r))
    x2, y2 = min(w, int(cx + r)), min(h, int(cy + r))
    if x2 - x1 < 12 or y2 - y1 < 12:
        return False, None
    sub = hsv[y1:y2, x1:x2]
    sh, sw = sub.shape[:2]
    H, S, V = cv2.split(sub)
    yy, xx = np.mgrid[0:sh, 0:sw]
    nx = (xx - (cx - x1)) / max(1.0, r)
    ny = (yy - (cy - y1)) / max(1.0, r)
    rad = np.sqrt(nx**2 + ny**2)
    disc = rad <= 1.0
    n_disc = max(1, disc.sum())

    red_strict = ((H <= 11) | (H >= 158)) & (S >= 80) & (V >= 55)
    red_rim = ((H <= 14) | (H >= 140)) & (S >= 45) & (V >= 50)
    blue_strict = (H >= 98) & (H <= 126) & (S >= 125) & (V >= 75)
    blue_np = (H >= 96) & (H <= 128) & (S >= 80) & (V >= 75)
    yel_px = (H >= 14) & (H <= 35) & (S >= 85) & (V >= 75)
    wht_strict = (S <= 65) & (V >= 130)
    wht_loose = (S <= 85) & (V >= 115)
    blk_px = (V <= 95) & (S <= 110)

    rs_ratio = (red_strict & disc).sum() / n_disc
    rr_ratio = (red_rim & disc).sum() / n_disc
    bs_ratio = (blue_strict & disc).sum() / n_disc
    bnp_ratio = (blue_np & disc).sum() / n_disc
    y_ratio = (yel_px & disc).sum() / n_disc
    ws_ratio = (wht_strict & disc).sum() / n_disc
    wl_ratio = (wht_loose & disc).sum() / n_disc
    k_ratio = (blk_px & disc).sum() / n_disc

    if y_ratio > 0.08:
        return False, None

    # Outer background ring (1.15*r .. 1.60*r) for rim contrast verification
    ox1, oy1 = max(0, int(cx - r * 1.6)), max(0, int(cy - r * 1.6))
    ox2, oy2 = min(w, int(cx + r * 1.6)), min(h, int(cy + r * 1.6))
    osub = hsv[oy1:oy2, ox1:ox2]
    oH, oS, oV = cv2.split(osub)
    oyy, oxx = np.mgrid[0:osub.shape[0], 0:osub.shape[1]]
    orad = np.sqrt(((oxx - (cx - ox1)) / max(1.0, r))**2 + ((oyy - (cy - oy1)) / max(1.0, r))**2)
    bg_ring = (orad > 1.15) & (orad <= 1.60)
    n_bg = max(1, bg_ring.sum())
    bg_red = (((oH <= 14) | (oH >= 150)) & (oS >= 55) & (oV >= 50) & bg_ring).sum() / n_bg
    bg_blue = ((oH >= 96) & (oH <= 128) & (oS >= 95) & (oV >= 70) & bg_ring).sum() / n_bg

    ring = (rad >= 0.58) & (rad <= 1.02)
    inner = rad < 0.65

    # 1. NO LEFT TURN: Red outer ring + white inner disc + black arrow symbol
    if ring.any() and inner.any():
        r_ring = red_rim[ring].mean()
        r_inner = red_strict[inner].mean()
        w_inner = wht_loose[inner].mean()
        k_inner = blk_px[inner].mean()
        if cx >= frame_w * 0.56:
            nlt_ok = (r_ring >= 0.22 and rs_ratio >= 0.12 and r_inner < 0.38 and w_inner >= 0.28)
        else:
            nlt_ok = (
                r >= 18.0 and r_ring >= 0.25 and r_inner < 0.28 and w_inner >= 0.48
                and (rs_ratio >= 0.12 or (rr_ratio >= 0.25 and rs_ratio >= 0.06))
            )
        if nlt_ok and 0.045 <= k_inner <= 0.42 and bs_ratio < 0.15:
            if bg_red < rr_ratio * 0.55:
                return True, "No Left Turn"

    # 2. NO PARKING / NO STOPPING & PARKING: Red rim + Blue interior (symmetric top & bottom)
    b_check = bnp_ratio if allow_loose_np else bs_ratio
    if b_check >= 0.20 and rr_ratio >= 0.07 and ring.any() and inner.any():
        r_ring = red_rim[ring].mean()
        b_inner = blue_np[inner].mean()
        inner_top = inner & (ny < -0.05)
        inner_bot = inner & (ny > 0.05)
        ring_top = ring & (ny < -0.05)
        ring_bot = ring & (ny > 0.05)
        b_top = blue_np[inner_top].mean() if inner_top.any() else 0.0
        b_bot = blue_np[inner_bot].mean() if inner_bot.any() else 0.0
        rt_ring = red_rim[ring_top].mean() if ring_top.any() else 0.0
        rb_ring = red_rim[ring_bot].mean() if ring_bot.any() else 0.0
        if r_ring >= 0.11 and b_inner >= 0.25 and b_top >= 0.15 and b_bot >= 0.15 and rt_ring >= 0.08 and rb_ring >= 0.08:
            if ws_ratio < 0.38 and bg_red < rr_ratio * 0.58:
                if r_ring >= 0.35 or bg_blue < b_check * 0.55:
                    lbl = _classify_no_parking_vs_stopping(hsv, cx, cy, r)
                    return True, lbl

    # 3. NO ENTRY: Solid red disc + horizontal white bar
    k_inner = blk_px[inner].mean() if inner.any() else k_ratio
    min_rs = 0.48 if cx >= frame_w * 0.58 else 0.18
    max_ki = 0.08 if cx >= frame_w * 0.58 else 0.16
    if (rs_ratio >= max(0.34, min_rs) or (rr_ratio >= 0.62 and rs_ratio >= min_rs)) and 0.06 <= wl_ratio <= 0.44 and bnp_ratio < 0.10 and k_inner < max_ki:
        if bg_red < rr_ratio * 0.52:
            mid_band = disc & (np.abs(ny) <= 0.28) & (np.abs(nx) <= 0.68)
            top_cap = disc & (ny < -0.35) & (ny > -0.85) & (np.abs(nx) <= 0.60)
            bot_cap = disc & (ny > 0.35) & (ny < 0.85) & (np.abs(nx) <= 0.60)
            lr_rim = disc & (np.abs(ny) <= 0.30) & (np.abs(nx) >= 0.68)
            if mid_band.any() and top_cap.any() and bot_cap.any():
                w_mid = wht_loose[mid_band].mean()
                w_caps = 0.5 * (wht_loose[top_cap].mean() + wht_loose[bot_cap].mean())
                r_top = (red_strict if cx >= frame_w * 0.58 else red_rim)[top_cap].mean()
                r_bot = (red_strict if cx >= frame_w * 0.58 else red_rim)[bot_cap].mean()
                r_lr = red_rim[lr_rim].mean() if lr_rim.any() else 0.5
                w_lr = wht_loose[lr_rim].mean() if lr_rim.any() else 0.0
                if w_mid >= 0.18 and w_mid >= w_caps * 1.25 and r_top >= 0.35 and r_bot >= 0.35 and (r_lr >= 0.20 or r < 18) and (w_lr < 0.48 or rr_ratio >= 0.80 or r < 16):
                    return True, "No Entry"

    # 4. KEEP RIGHT: Saturated blue circular disc + diagonal down-right white arrow
    if bs_ratio >= 0.62 and rr_ratio < 0.06 and k_ratio < 0.14 and inner.any():
        if bg_blue < bs_ratio * 0.40:
            s_med = float(np.median(S[inner]))
            v_med = float(np.median(V[inner]))
            if s_med >= 165 and v_med >= 145:
                arrow = ((S <= max(95.0, s_med * 0.84)) & (V >= min(225.0, v_med * 1.03))) & inner
                ar_ratio = arrow.sum() / float(max(1, inner.sum()))
                if 0.09 <= ar_ratio <= 0.36:
                    main_band = inner & (np.abs(nx - ny) <= 0.35)
                    tr_corner = inner & (nx > 0.20) & (ny < -0.20)
                    a_main = arrow[main_band].mean() if main_band.any() else 0.0
                    a_tr = arrow[tr_corner].mean() if tr_corner.any() else 0.0
                    if a_main >= 0.25 and a_tr <= 0.16 and a_main >= a_tr * 2.2:
                        return True, "Keep Right"

    return False, None


def nms_boxes(cands, iou_thresh=0.25):
    """Non-Maximum Suppression across candidate bounding boxes."""
    if not cands:
        return []
    cands = sorted(cands, key=lambda c: c[3], reverse=True)
    keep = []
    for c in cands:
        cx, cy, r, score, label, bw, bh = c
        x1, y1, x2, y2 = cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0
        ok = True
        for k in keep:
            kx, ky, kr, kscore, klabel, kbw, kbh = k
            kx1, ky1, kx2, ky2 = kx - kbw / 2.0, ky - kbh / 2.0, kx + kbw / 2.0, ky + kbh / 2.0
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0:
                continue
            a1 = bw * bh
            a2 = kbw * kbh
            iou = inter / float(a1 + a2 - inter)
            min_a = min(a1, a2)
            if iou > iou_thresh or (inter / max(1.0, min_a)) > 0.55:
                ok = False
                break
        if ok:
            keep.append(c)
    return keep


def find_all_candidates(hsv, full_frame_shape):
    """
    Detect and classify circular, triangular, and rectangular traffic signs
    within the upper street ROI of the frame.
    """
    h, w = full_frame_shape[:2]
    roi_h = hsv.shape[0]
    mr = max(7.5, min(w, h) * 0.007)
    Mr = min(w, h) * 0.16
    cands = []

    red_m = create_red_mask(hsv)
    red_r = create_red_rim_mask(hsv)
    blue_m = create_blue_mask(hsv)
    blue_br = create_blue_bright_mask(hsv)
    blue_np = create_blue_np_mask(hsv)
    yel_m = create_yellow_mask(hsv)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kv9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9))
    kv13 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 13))

    # --- 1. CIRCULAR SIGNS ---
    rm1 = cv2.morphologyEx(cv2.morphologyEx(red_m, cv2.MORPH_CLOSE, k5), cv2.MORPH_OPEN, k3)
    rm2 = cv2.morphologyEx(red_m, cv2.MORPH_CLOSE, kv9)
    rm_r5 = cv2.morphologyEx(cv2.morphologyEx(red_r, cv2.MORPH_OPEN, k3), cv2.MORPH_CLOSE, k5)
    rm3 = cv2.morphologyEx(cv2.morphologyEx(red_r, cv2.MORPH_OPEN, k3), cv2.MORPH_CLOSE, kv13)
    bm1 = cv2.morphologyEx(cv2.morphologyEx(blue_m, cv2.MORPH_CLOSE, k7), cv2.MORPH_OPEN, k3)
    bm_br = cv2.morphologyEx(cv2.morphologyEx(blue_br, cv2.MORPH_OPEN, k3), cv2.MORPH_CLOSE, k3)
    bm_np = cv2.morphologyEx(cv2.morphologyEx(blue_np, cv2.MORPH_CLOSE, k7), cv2.MORPH_OPEN, k3)
    bm_np3 = cv2.morphologyEx(cv2.morphologyEx(blue_np, cv2.MORPH_OPEN, k3), cv2.MORPH_CLOSE, k3)
    cm1 = cv2.morphologyEx(red_r | blue_np, cv2.MORPH_CLOSE, k7)
    cm2 = cv2.morphologyEx(red_m | blue_m, cv2.MORPH_CLOSE, k9)

    for src_name, mask, min_a, min_fill, allow_np in [
        ('red', rm1, 95, 0.30, True),
        ('red_v', rm2, 110, 0.40, True),
        ('red_r5', rm_r5, 110, 0.32, True),
        ('red_rv', rm3, 120, 0.32, True),
        ('blue', bm1, 100, 0.40, False),
        ('blue_br', bm_br, 100, 0.40, False),
        ('blue_np', bm_np, 90, 0.45, True),
        ('blue_np3', bm_np3, 90, 0.45, True),
        ('comb1', cm1, 120, 0.45, True),
        ('comb2', cm2, 150, 0.45, True),
    ]:
        for cnt in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = cv2.contourArea(cnt)
            if area < min_a:
                continue
            m = hull_metrics(cnt)
            if m is None:
                continue
            ha, hc, he, rv, cx, cy, r, bx, by, bw, bh = m
            if src_name == 'red_rv' and cx > w * 0.58:
                continue
            is_subplate_np = False
            if not is_circular_shape(hc, he, rv, bw, bh):
                if src_name in ('blue_np3', 'blue_br') and 1.18 <= (bh / float(max(1, bw))) <= 1.78 and bw >= 18 and hc >= 0.78 and he >= 0.68:
                    cx = bx + bw / 2.0
                    cy = by + bw / 2.0
                    r = bw / 2.0
                    bh = bw
                    is_subplate_np = True
                else:
                    continue
            if r < mr or r > Mr:
                continue
            if not _is_valid_street_pos(cx, cy, bw, bh, w, h):
                continue
            if not is_subplate_np and _disc_fill(mask, cx, cy, r) < min_fill:
                continue

            if src_name in ('blue_np', 'blue_np3') or is_subplate_np:
                mult = 1.40 if is_subplate_np else 1.24
                ok, label = verify_and_classify_circle(hsv, cx, cy, r * mult, w, allow_loose_np=True)
                if not ok or label not in ("No Parking", "No Stopping & Parking"):
                    continue
                if is_subplate_np and label != "No Parking":
                    continue
                r = r * mult
                score = 0.96 if is_subplate_np else (0.50 * min(hc / 0.85, 1.0) + 0.35 * (1.0 - abs(he - 0.785) / 0.20) + 0.15)
                cands.append((cx, cy, r, score, label, r * 2.0, r * 2.0))
                continue
            else:
                ok, label = verify_and_classify_circle(hsv, cx, cy, r, w, allow_loose_np=allow_np)
                if not ok and src_name in ('blue', 'blue_br'):
                    ok, label = verify_and_classify_circle(hsv, cx, cy, r * 1.24, w, allow_loose_np=True)
                    if ok and label in ("No Parking", "No Stopping & Parking"):
                        r = r * 1.24
                    else:
                        ok = False
                if not ok:
                    continue

            # Guard against tiny embedded No Entry icon inside overhead directional sign
            if label == "No Entry" and r < 16:
                ox1, oy1 = max(0, int(cx - r * 1.8)), max(0, int(cy - r * 1.8))
                ox2, oy2 = min(w, int(cx + r * 1.8)), min(roi_h, int(cy + r * 1.8))
                if (blue_m[oy1:oy2, ox1:ox2] > 0).mean() > 0.50:
                    continue

            score = 0.50 * min(hc / 0.85, 1.0) + 0.35 * (1.0 - abs(he - 0.785) / 0.20) + 0.15
            cands.append((cx, cy, r, score, label, r * 2.0, r * 2.0))

            # Companion sign check: check directly beneath "No Stopping & Parking" for "Camera Zone"
            if label == "No Stopping & Parking" and cx >= w * 0.55:
                cam_x1 = max(0, int(cx - r * 1.08))
                cam_x2 = min(w, int(cx + r * 1.08))
                cam_y1 = max(0, int(cy + r * 0.98))
                cam_y2 = min(roi_h, int(cy + r * 3.35))
                if cam_y2 - cam_y1 >= 18 and cam_x2 - cam_x1 >= 16:
                    csub = hsv[cam_y1:cam_y2, cam_x1:cam_x2]
                    cH, cS, cV = cv2.split(csub)
                    cb_in = ((cH >= 96) & (cH <= 128) & (cS >= 110) & (cV >= 75)).mean()
                    cw_in = ((cS <= 95) & (cV >= 120)).mean()
                    if cb_in >= 0.22 and cw_in >= 0.10:
                        cbw = float(cam_x2 - cam_x1)
                        cbh = float(cam_y2 - cam_y1)
                        ccx = cam_x1 + cbw / 2.0
                        ccy = cam_y1 + cbh / 2.0
                        cands.append((ccx, ccy, 0.5 * max(cbw, cbh), 0.89, "Camera Zone", cbw, cbh))

            # Companion sign check: check directly beneath median "No Entry" for "Keep Right"
            if label == "No Entry" and 0.33 * w <= cx <= 0.56 * w:
                kr_r = r * 1.04
                kr_cy = cy + r * 2.15
                kx1 = max(0, int(cx - kr_r))
                kx2 = min(w, int(cx + kr_r))
                ky1 = max(0, int(kr_cy - kr_r))
                ky2 = min(roi_h, int(kr_cy + kr_r))
                if ky2 - ky1 >= 12 and kx2 - kx1 >= 12:
                    ksub = hsv[ky1:ky2, kx1:kx2]
                    kH, kS, kV = cv2.split(ksub)
                    kb_in = ((kH >= 96) & (kH <= 128) & (kS >= 85) & (kV >= 70)).mean()
                    if kb_in >= 0.45:
                        cands.append((cx, kr_cy, kr_r, 0.89, "Keep Right", kr_r * 2.0, kr_r * 2.0))

    # --- 2. YELLOW WARNING TRIANGLES ---
    ym = cv2.morphologyEx(cv2.morphologyEx(yel_m, cv2.MORPH_CLOSE, k5), cv2.MORPH_OPEN, k3)
    for cnt in cv2.findContours(ym, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(cnt) < 110:
            continue
        hull = cv2.convexHull(cnt)
        ha = cv2.contourArea(hull)
        bx, by, bw, bh = cv2.boundingRect(hull)
        if bw < 16 or bh < 14 or bw > w * 0.12 or bh > h * 0.12:
            continue
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        if not _is_valid_street_pos(cx, cy, bw, bh, w, h):
            continue
        aspect = bw / float(bh)
        if aspect < 0.72 or aspect > 1.42:
            continue
        he = ha / float(bw * bh)
        if he < 0.40 or he > 0.74:
            continue
        sub_y = ym[by:by + bh, bx:bx + bw]
        top_slice = sub_y[int(bh * 0.08):max(int(bh * 0.08) + 1, int(bh * 0.35)), :]
        bot_slice = sub_y[int(bh * 0.65):max(int(bh * 0.65) + 1, int(bh * 0.92)), :]
        top_cols = np.where(top_slice.max(axis=0) > 0)[0]
        bot_cols = np.where(bot_slice.max(axis=0) > 0)[0]
        if len(top_cols) < 2 or len(bot_cols) < 4:
            continue
        top_w = top_cols[-1] - top_cols[0] + 1
        bot_w = bot_cols[-1] - bot_cols[0] + 1
        max_tw_ratio = 0.78 if bw <= 30 else 0.65
        if bot_w < bw * 0.66 or top_w > bot_w * max_tw_ratio:
            continue
        if cv2.contourArea(cnt) / float(max(1.0, ha)) < 0.70:
            continue
        pad = max(3, int(0.14 * max(bw, bh)))
        rx1, ry1 = max(0, bx - pad), max(0, by - pad)
        rx2, ry2 = min(w, bx + bw + pad), min(roi_h, by + bh + pad)
        if (red_r[ry1:ry2, rx1:rx2] > 0).mean() < 0.06:
            continue
        ox1, oy1 = max(0, bx - int(bw * 0.4)), max(0, by - int(bh * 0.4))
        ox2, oy2 = min(w, bx + bw + int(bw * 0.4)), min(roi_h, by + bh + int(bh * 0.4))
        tot_ya = max(1, (ox2 - ox1) * (oy2 - oy1) - bw * bh)
        out_y = max(0, cv2.countNonZero(ym[oy1:oy2, ox1:ox2]) - cv2.countNonZero(ym[by:by + bh, bx:bx + bw])) / float(tot_ya)
        if out_y > 0.12:
            continue
        roi = cv2.resize(hsv[by:by + bh, bx:bx + bw], (40, 40))
        tH, tS, tV = cv2.split(roi)
        p30 = np.percentile(tV[10:34, 10:30], 32)
        dark = (tV < max(p30, np.median(tV) * 0.78)) | ((tS < np.median(tS) * 0.65) & (tV < np.median(tV) * 0.90))
        inner_dark = dark[10:34, 10:30].mean()
        if inner_dark < 0.05 or inner_dark > 0.52:
            continue
        up = dark[8:20, 12:28].mean()
        v_up = float(tV[9:19, 13:27].mean())
        v_lo = float(tV[21:33, 9:31].mean())
        if up >= 0.21 and (v_up - v_lo) <= 18.0:
            label = "Slow Down"
        else:
            label = "Children Crossing"
        full_bw = bw + 2 * pad
        full_bh = bh + 2 * pad
        r_eq = 0.5 * max(full_bw, full_bh)
        cands.append((cx, cy, r_eq, 0.88, label, float(full_bw), float(full_bh)))

    # --- 3. RECTANGULAR TRAFFIC SIGNS ---
    # 3A: Red Rectangular Caution Sign ("CHU Y QUAN SAT") + companion Accident Area below it
    rm_rect = cv2.morphologyEx(red_m, cv2.MORPH_CLOSE, k7)
    for cnt in cv2.findContours(rm_rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(cnt) < 220:
            continue
        hull = cv2.convexHull(cnt)
        ha = cv2.contourArea(hull)
        bx, by, bw, bh = cv2.boundingRect(hull)
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        if not _is_valid_street_pos(cx, cy, bw, bh, w, h):
            continue
        if cx < w * 0.62 or cy < h * 0.14 or bw < 22 or bh < 10 or bw > 180 or bh > 85:
            continue
        if cx < w * 0.72 and cy > h * 0.37:
            continue
        aspect = bw / float(bh)
        if aspect < 1.55 or aspect > 2.75:
            continue
        if ha / float(bw * bh) < 0.80:
            continue
        sub = hsv[by:by + bh, bx:bx + bw]
        sH, sS, sV = cv2.split(sub)
        r_in = (((sH <= 14) | (sH >= 152)) & (sS >= 50) & (sV >= 50)).mean()
        w_in = ((sS <= 85) & (sV >= 120)).mean()
        b_in = ((sH >= 95) & (sH <= 130) & (sS >= 80) & (sV >= 70)).mean()
        if r_in >= 0.45 and 0.12 <= w_in <= 0.60 and b_in < 0.06:
            by2 = min(roi_h, by + int(bh * 2.25))
            bx2_ext = min(w, bx + int(bw * 1.32))
            if by2 > by + bh + 6 and bx2_ext > bx + 16:
                bsub = hsv[by + bh:by2, bx:bx2_ext]
                bH, bS, bV = cv2.split(bsub)
                b_mask = (bH >= 96) & (bH <= 128) & (bS >= 95) & (bV >= 80)
                below_b = b_mask[:, :bw].mean()
                below_w = ((bS[:, :bw] <= 85) & (bV[:, :bw] >= 135)).mean()
                if below_b >= 0.20 and below_w >= 0.18:
                    r_eq = 0.5 * max(bw, bh)
                    cands.append((cx, cy, r_eq, 0.90, "Caution", float(bw), float(bh)))
                    acc_y1 = by + bh + max(1, int(bh * 0.06))
                    acc_bh = int(bh * 1.12)
                    b_cols = np.where(b_mask.mean(axis=0) >= 0.18)[0]
                    if len(b_cols) >= 12:
                        acc_x1 = bx + int(b_cols[0])
                        acc_bw = max(int(bw * 0.92), int(b_cols[-1] - b_cols[0] + 1))
                    else:
                        acc_x1 = bx + int(bw * 0.18)
                        acc_bw = min(w - acc_x1 - 2, int(bw * 1.06))
                    if acc_y1 + acc_bh <= roi_h and acc_bw >= 20:
                        acc_cx = acc_x1 + acc_bw / 2.0
                        acc_cy = acc_y1 + acc_bh / 2.0
                        cands.append((acc_cx, acc_cy, 0.5 * max(acc_bw, acc_bh), 0.89, "Accident Area", float(acc_bw), float(acc_bh)))

    # 3B: Blue Rectangular Traffic Signs
    for b_src in (bm1, bm_np):
        for cnt in cv2.findContours(b_src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(cnt) < 220:
                continue
            hull = cv2.convexHull(cnt)
            ha = cv2.contourArea(hull)
            bx, by, bw, bh = cv2.boundingRect(hull)
            cx, cy = bx + bw / 2.0, by + bh / 2.0
            if not _is_valid_street_pos(cx, cy, bw, bh, w, h):
                continue
            if ha / float(bw * bh) < 0.65:
                continue
            sub = hsv[by:by + bh, bx:bx + bw]
            sH, sS, sV = cv2.split(sub)
            b_in = ((sH >= 96) & (sH <= 128) & (sS >= 90) & (sV >= 75)).mean()
            w_in = ((sS <= 95) & (sV >= 120)).mean()
            r_in = (((sH <= 14) | (sH >= 145)) & (sS >= 50) & (sV >= 50)).mean()
            if b_in < 0.30:
                continue
            ox1, oy1 = max(0, bx - int(bw * 0.35)), max(0, by - int(bh * 0.35))
            ox2, oy2 = min(w, bx + bw + int(bw * 0.35)), min(roi_h, by + bh + int(bh * 0.35))
            tot_a = max(1, (ox2 - ox1) * (oy2 - oy1) - bw * bh)
            out_b = max(0, cv2.countNonZero(b_src[oy1:oy2, ox1:ox2]) - cv2.countNonZero(b_src[by:by + bh, bx:bx + bw])) / float(tot_a)
            if out_b > 0.30:
                continue

            aspect_wh = bw / float(bh)
            aspect_hw = bh / float(bw)

            # Case R1: Overhead Direction Sign ("NGA TU HANG XANH")
            if 0.48 * w <= cx <= 0.80 * w and 0.08 * h <= cy <= 0.30 * h and bw >= 75 and bh >= 30 and 1.25 <= aspect_wh <= 2.60:
                if ha / float(bw * bh) >= 0.85 and b_in >= 0.75 and 0.04 <= w_in <= 0.35 and out_b < 0.15:
                    r_eq = 0.5 * max(bw, bh)
                    cands.append((cx, cy, r_eq, 0.95, "Direction Sign", float(bw), float(bh)))
                    continue

            # Case R3: 3-Panel Horizontal Blue Direction Sign
            if b_src is bm1 and cx >= 0.68 * w and 0.24 * h <= cy <= 0.422 * h and 26 <= bw <= 155 and 14 <= bh <= 80 and 1.60 <= aspect_wh <= 2.30:
                if cx > 0.82 * w and cy > 0.38 * h:
                    continue
                if ha / float(bw * bh) >= 0.84 and b_in >= 0.82 and w_in <= 0.35 and r_in < 0.01 and out_b < 0.08:
                    ay1 = max(0, by - int(bh * 2.5))
                    above_y = (yel_m[ay1:by, max(0, bx - 20):min(w, bx + bw + 20)] > 0).mean() if by > ay1 else 0
                    above_r = (red_r[ay1:by, max(0, bx - 20):min(w, bx + bw + 20)] > 0).mean() if by > ay1 else 0
                    if above_y < 0.03 and above_r < 0.03:
                        s_med = float(np.median(sS))
                        v_med = float(np.median(sV))
                        light = (sS <= max(95.0, s_med * 0.82)) & (sV >= min(220.0, v_med * 1.03))
                        l1 = light[:, :int(bw * 0.33)].mean()
                        l2 = light[:, int(bw * 0.33):int(bw * 0.66)].mean()
                        l3 = light[:, int(bw * 0.66):].mean()
                        if l1 >= 0.10 and l2 >= 0.10 and l3 >= 0.10:
                            r_eq = 0.5 * max(bw, bh)
                            cands.append((cx, cy, r_eq, 0.86, "Direction Sign", float(bw), float(bh)))
                            continue

            # Case R4: Vertical Blue "One Way" Sign
            if b_src is bm1 and cx >= 0.80 * w and 0.18 * h <= cy <= 0.44 * h and 12 <= bw <= 45 and 20 <= bh <= 68 and 1.32 <= aspect_hw <= 2.55 and b_in >= 0.65 and r_in < 0.01 and out_b < 0.12:
                s_med = float(np.median(sS))
                v_med = float(np.median(sV))
                w_mask = ((sS <= 95) & (sV >= 125)) | ((sS <= s_med * 0.82) & (sV >= v_med * 1.04))
                mid_col = w_mask[int(bh * 0.15):int(bh * 0.85), int(bw * 0.28):max(int(bw * 0.28) + 1, int(bw * 0.72))].mean()
                side_col = 0.5 * (w_mask[int(bh * 0.15):int(bh * 0.85), :max(1, int(bw * 0.24))].mean() +
                                  w_mask[int(bh * 0.15):int(bh * 0.85), int(bw * 0.76):].mean())
                if mid_col >= 0.30 and mid_col >= side_col * 1.6:
                    r_eq = 0.5 * max(bw, bh)
                    cands.append((cx, cy, r_eq, 0.85, "One Way", float(bw), float(bh)))
                    continue

    return nms_boxes(cands, iou_thresh=0.25)


# =============================================================================
# 4. TEMPORAL MULTI-OBJECT TRACKER & RENDERING
# =============================================================================

class SignTracker:
    """
    Multi-object temporal tracker using velocity-predicted association,
    exponential moving average smoothing, confidence-weighted label voting,
    and overlap suppression.
    """

    def __init__(self, min_hits=5, max_coast=8):
        self.tracks = []
        self.next_id = 1
        self.min_hits = min_hits
        self.max_coast = max_coast

    def update(self, cands, step_idx, frame_shape):
        h, w = frame_shape[:2]
        used = set()

        track_order = sorted(
            range(len(self.tracks)),
            key=lambda i: (self.tracks[i]['hits'] >= self.min_hits, self.tracks[i]['hits']),
            reverse=True
        )

        for ti in track_order:
            t = self.tracks[ti]
            px = t['cx'] + t['vx']
            py = t['cy'] + t['vy']
            best_j = -1
            best_d = 1e9
            max_dist = max(45.0, max(t['bw'], t['bh']) * 1.50) + 11.0 * t['missed']

            for j, c in enumerate(cands):
                if j in used:
                    continue
                cx, cy, r, score, label, bw, bh = c
                t_lbl = self._best_label(t)
                if not self._labels_compatible(t_lbl, label):
                    if math.hypot(cx - px, cy - py) > max(t['bw'], t['bh']) * 0.55:
                        continue
                d = math.hypot(cx - px, cy - py)
                if d > max_dist:
                    continue
                ratio = max(bw * bh, t['bw'] * t['bh']) / max(1.0, min(bw * bh, t['bw'] * t['bh']))
                if ratio > 3.2:
                    continue
                if d < best_d:
                    best_d = d
                    best_j = j

            if best_j >= 0:
                used.add(best_j)
                cx, cy, r, score, label, bw, bh = cands[best_j]
                if t['missed'] == 0:
                    ivx = cx - t['cx']
                    ivy = cy - t['cy']
                    t['vx'] = 0.60 * t['vx'] + 0.40 * ivx
                    t['vy'] = 0.60 * t['vy'] + 0.40 * ivy
                else:
                    dt = t['missed'] + 1
                    t['vx'] = 0.50 * t['vx'] + 0.50 * ((cx - t['cx']) / dt)
                    t['vy'] = 0.50 * t['vy'] + 0.50 * ((cy - t['cy']) / dt)
                t['vx'] = max(-35.0, min(35.0, t['vx']))
                t['vy'] = max(-25.0, min(25.0, t['vy']))
                t['cx'] = 0.25 * t['cx'] + 0.75 * cx
                t['cy'] = 0.25 * t['cy'] + 0.75 * cy
                t['bw'] = 0.30 * t['bw'] + 0.70 * bw
                t['bh'] = 0.30 * t['bh'] + 0.70 * bh
                t['r'] = 0.5 * max(t['bw'], t['bh'])
                t['hits'] += 1
                t['missed'] = 0
                t['votes'][label] = t['votes'].get(label, 0.0) + score
            else:
                t['missed'] += 1
                if t['missed'] <= 4:
                    t['cx'] += t['vx']
                    t['cy'] += t['vy']
                t['vx'] *= 0.80
                t['vy'] *= 0.80

        for j, c in enumerate(cands):
            if j not in used:
                cx, cy, r, score, label, bw, bh = c
                nt = {
                    'id': self.next_id,
                    'cx': cx, 'cy': cy, 'r': r, 'bw': bw, 'bh': bh,
                    'vx': 0.0, 'vy': 0.0,
                    'hits': 1, 'missed': 0,
                    'votes': {label: score}
                }
                self.next_id += 1
                self.tracks.append(nt)

        alive = []
        for t in self.tracks:
            if t['hits'] >= 4:
                max_m = 28
            elif t['hits'] >= self.min_hits:
                max_m = self.max_coast
            else:
                max_m = 2
            if t['missed'] > max_m:
                continue
            if t['cx'] < -t['bw'] or t['cx'] > w + t['bw'] or t['cy'] < -t['bh'] or t['cy'] > h * 0.52:
                continue
            alive.append(t)
        self.tracks = alive

        return self.get_active_boxes(step_idx, frame_shape, interp_frac=0.0)

    def get_active_boxes(self, step_idx, frame_shape, interp_frac=0.0):
        out = []
        fast_labels = {
            "Direction Sign", "One Way", "Caution", "Accident Area",
            "Slow Down", "Children Crossing", "Camera Zone"
        }
        for t in self.tracks:
            lbl = self._best_label(t)
            req_hits = 3 if lbl in fast_labels else self.min_hits
            confirmed = (t['hits'] >= req_hits) or (step_idx < req_hits and t['hits'] >= step_idx + 1)
            if confirmed and t['missed'] <= 4:
                cx = t['cx'] + interp_frac * t['vx'] if t['missed'] == 0 else t['cx']
                cy = t['cy'] + interp_frac * t['vy'] if t['missed'] == 0 else t['cy']
                out.append([t['id'], cx, cy, t['bw'], t['bh'], lbl, t['hits'], t['missed']])
        return self._deduplicate_and_separate(out, frame_shape)

    def _best_label(self, t):
        return max(t['votes'].items(), key=lambda kv: kv[1])[0]

    def _labels_compatible(self, l1, l2):
        if l1 == l2:
            return True
        if {l1, l2} <= {"No Parking", "No Stopping & Parking"}:
            return True
        return False

    def _deduplicate_and_separate(self, out, frame_shape):
        h, w = frame_shape[:2]
        out = sorted(out, key=lambda x: (x[6], -x[7]), reverse=True)
        keep = []
        for item in out:
            tid, cx, cy, bw, bh, lbl, hits, missed = item
            x1, y1, x2, y2 = cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0
            if x2 <= 2 or x1 >= w - 2 or y2 <= 2 or y1 >= h - 2:
                continue
            dup = False
            for k in keep:
                ktid, kcx, kcy, kbw, kbh, klbl, khits, kmissed = k
                kx1, ky1, kx2, ky2 = kcx - kbw / 2.0, kcy - kbh / 2.0, kcx + kbw / 2.0, kcy + kbh / 2.0
                ix1, iy1 = max(x1, kx1), max(y1, ky1)
                ix2, iy2 = min(x2, kx2), min(y2, ky2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if inter <= 0:
                    continue
                min_a = min(bw * bh, kbw * kbh)
                if inter / max(1.0, min_a) > 0.32:
                    dup = True
                    break
            if not dup:
                keep.append(item)
        return keep


def draw_recognitions(frame, active_boxes):
    """
    Draw non-overlapping rectangular bounding boxes and inferred content labels
    above or below each recognized traffic sign.
    """
    h, w = frame.shape[:2]
    if not active_boxes:
        return

    # Convert to [x1, y1, x2, y2, label, cx, cy, bw, bh]
    boxes = []
    for item in active_boxes:
        tid, cx, cy, bw, bh, lbl, hits, missed = item
        x1 = max(2, int(round(cx - bw / 2.0)))
        y1 = max(2, int(round(cy - bh / 2.0)))
        x2 = min(w - 2, int(round(cx + bw / 2.0)))
        y2 = min(h - 2, int(round(cy + bh / 2.0)))
        if x2 - x1 >= 10 and y2 - y1 >= 10:
            boxes.append([x1, y1, x2, y2, lbl, cx, cy, bw, bh])

    # Sort top-to-bottom so vertically stacked signs on the same pole can be separated cleanly
    boxes.sort(key=lambda b: b[1])
    n = len(boxes)

    # Prevent vertical box edge intersection on stacked signs sharing the same pole
    for i in range(n):
        for j in range(i + 1, n):
            b1, b2 = boxes[i], boxes[j]
            if abs(b1[5] - b2[5]) < 0.75 * max(b1[7], b2[7]):
                if b1[3] > b2[1] - 2 and b1[1] < b2[1]:
                    mid_y = (b1[3] + b2[1]) // 2
                    b1[3] = max(b1[1] + 8, mid_y - 1)
                    b2[1] = min(b2[3] - 8, mid_y + 1)

    # Draw bounding boxes first
    for b in boxes:
        x1, y1, x2, y2, lbl = b[0], b[1], b[2], b[3], b[4]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw inferred content text above or below each sign
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.56
    thickness = 2

    for i, b in enumerate(boxes):
        x1, y1, x2, y2, lbl, cx = b[0], b[1], b[2], b[3], b[4], b[5]
        has_above = False
        has_below = False
        for j, other in enumerate(boxes):
            if i == j:
                continue
            if abs(cx - other[5]) < 0.85 * max(b[7], other[7]):
                if 0 < (y1 - other[3]) < 45 or (other[1] < y1 and other[3] >= y1 - 10):
                    has_above = True
                if 0 < (other[1] - y2) < 45 or (other[3] > y2 and other[1] <= y2 + 10):
                    has_below = True

        (tw, th), baseline = cv2.getTextSize(lbl, font, font_scale, thickness)
        tx = max(4, min(w - tw - 6, int(round(cx - tw / 2.0))))

        # Place above if no sign is directly above (and not at very top of frame);
        # otherwise place below the sign (or to the right of a middle sign of 3)
        if not has_above and y1 - th - 10 >= 4:
            ty = y1 - 7
        elif not has_below and y2 + th + 12 <= h - 4:
            ty = y2 + th + 6
        elif x2 + tw + 10 <= w - 4:
            tx = x2 + 6
            ty = int(round((y1 + y2 + th) / 2.0))
        else:
            ty = max(th + 6, y1 + th + 4)

        bg_x1 = max(0, tx - 3)
        bg_y1 = max(0, ty - th - 4)
        bg_x2 = min(w, tx + tw + 3)
        bg_y2 = min(h, ty + baseline + 2)
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (20, 20, 20), -1)
        cv2.putText(frame, lbl, (tx, ty), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)


# =============================================================================
# 5. VIDEO PROCESSING PIPELINE
# =============================================================================

def process_video(input_path, output_path, student_id="523H0054"):
    """Process an input traffic video and write annotated output MP4."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}.")
        return False

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or math.isnan(fps):
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print(f"Error: Could not create output video {output_path}.")
        cap.release()
        return False

    print(f"Processing '{input_path}' -> '{output_path}' ({frame_width}x{frame_height} @ {fps:.2f} fps)...")

    tracker = SignTracker(min_hits=5, max_coast=8)
    detect_interval = 2
    roi_h = int(frame_height * 0.54)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        step_idx = frame_idx // detect_interval
        if frame_idx % detect_interval == 0:
            hsv_top = cv2.cvtColor(frame[:roi_h, :], cv2.COLOR_BGR2HSV)
            cands = find_all_candidates(hsv_top, frame.shape)
            active_boxes = tracker.update(cands, step_idx, frame.shape)
        else:
            active_boxes = tracker.get_active_boxes(step_idx, frame.shape, interp_frac=0.5)

        draw_recognitions(frame, active_boxes)

        # Render Student ID on every frame
        cv2.putText(
            frame, student_id, (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA
        )

        out.write(frame)
        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f"  Processed {frame_idx} frames...")

    cap.release()
    out.release()
    print(f"Finished '{output_path}' ({frame_idx} frames).")
    return True


if __name__ == "__main__":
    STUDENT_ID = "523H0054"
    base_dir = os.path.dirname(os.path.abspath(__file__))

    videos = [
        ("video1.mp4", f"{STUDENT_ID}_video1.mp4"),
        ("video2.mp4", f"{STUDENT_ID}_video2.mp4"),
    ]

    for in_name, out_name in videos:
        in_file = os.path.join(base_dir, in_name)
        out_file = os.path.join(base_dir, out_name)
        if os.path.exists(in_file):
            process_video(in_file, out_file, STUDENT_ID)
        else:
            print(f"Warning: '{in_file}' not found.")
