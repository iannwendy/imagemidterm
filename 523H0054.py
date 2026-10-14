#!/usr/bin/env python3
"""
Traffic Sign Detection - Circular Signs Only
MSSV: 523H0054
Subject: Introduction to Digital Image Processing (505060)

Based on DIP Guideline:
- HSV color filtering with specified thresholds
- Contour detection and circularity verification
- Multiple detection strategies combined for robustness
"""

import cv2
import numpy as np
import math
import os


# ============================================================================
# STUDENT ID CONFIGURATION
# ============================================================================
STUDENT_ID = "523H0054"


# ============================================================================
# STEP 1: COLOR FILTERING (HSV-based) - From Guideline
# ============================================================================
def create_red_mask(hsv):
    """Red wraps around the HSV hue seam. Two ranges joined with bitwise OR."""
    return cv2.inRange(hsv, np.array([0, 55, 45]), np.array([14, 255, 255])) | \
           cv2.inRange(hsv, np.array([158, 55, 45]), np.array([179, 255, 255]))


def create_blue_mask(hsv):
    """Blue traffic signs use highly saturated blue.  Loose thresholds merge
    the sign with the sky and the road, so we keep the saturation high
    (>= 130) to suppress those large low-saturation regions."""
    return cv2.inRange(hsv, np.array([95, 130, 60]), np.array([130, 255, 255]))


def create_yellow_mask(hsv):
    """Yellow traffic signs - lower saturation threshold because yellow
    in real scenes is often desaturated by sunlight and dirt."""
    return cv2.inRange(hsv, np.array([18, 80, 70]), np.array([35, 255, 255]))


# ============================================================================
# STEP 2 & 3: SHAPE CHECKING
# ============================================================================
def _is_solid_colour(hsv, cx, cy, radius, colour):
    """Return True if a circular region of the frame is overwhelmingly
    filled with the given colour.  Real prohibition / mandatory signs
    always contain a contrasting centre, so a solid-colour blob is more
    likely a helmet, jacket, taillight or a coloured marker on a larger
    sign.

    ``colour`` is either ``'red'`` or ``'blue'``.
    """
    x1 = max(0, int(cx - radius))
    y1 = max(0, int(cy - radius))
    x2 = min(hsv.shape[1], int(cx + radius))
    y2 = min(hsv.shape[0], int(cy + radius))
    if x2 <= x1 or y2 <= y1:
        return False
    roi = hsv[y1:y2, x1:x2]
    H, S, V = cv2.split(roi)
    total = max(1, H.size)
    if colour == 'red':
        target = (((H <= 14) | (H >= 158)) & (S >= 55) & (V >= 45)).sum()
    else:
        target = ((H >= 95) & (H <= 130) & (S >= 130) & (V >= 60)).sum()
    return float(target) / total > 0.65


def circle_score(area, perimeter, radius, bw, bh):
    """Combine the standard circularity metric with the radius and bounding
    box fill ratios.  A score of 1.0 means the contour looks like a
    perfectly round, fully filled disc."""
    if perimeter <= 0 or radius <= 0 or bw * bh <= 0:
        return 0.0
    circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
    circle_fill = area / (math.pi * radius * radius)
    bb_fill = area / float(bw * bh)
    return (0.45 * min(circularity / 0.75, 1.0) +
            0.35 * min(circle_fill / 0.65, 1.0) +
            0.20 * min(bb_fill / 0.70, 1.0))


def is_circular_candidate(cx, cy, radius, area, bw, bh, perimeter):
    """Strictly test whether a contour is shaped like a circle.

    The rules reject anything that looks like text inside a rectangle,
    a thin diagonal stroke, or a sky/road blob.  ``cx``/``cy`` are the
    centre returned by ``cv2.minEnclosingCircle``.
    """
    if perimeter <= 0 or radius <= 0 or bw <= 0 or bh <= 0:
        return False
    circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
    aspect = bw / float(bh)
    if aspect < 0.62 or aspect > 1.60:
        return False
    if circularity < 0.32:
        return False
    circle_fill = area / (math.pi * radius * radius)
    if circle_fill < 0.30:
        return False
    bb_fill = area / float(bw * bh)
    if bb_fill < 0.35:
        return False
    return True


# ============================================================================
# STEP 4: DETECTION - combining colour masks, contours and Hough circles
# ============================================================================
def find_circle_candidates(hsv, frame_shape):
    """Return de-duplicated boxes of likely circular traffic signs.

    Three independent sources are merged:

    1.  Red contour detection -- finds the red rim of "prohibition" signs.
    2.  Blue contour detection + Hough circles -- finds solid blue
        "mandatory direction" signs and blue rectangles that are NOT
        full-circle (handled by the rectangle-discriminator below).
    3.  Combined red+blue mask -- a sign with a red rim and a blue
        interior ("no parking") only reveals its true shape when both
        colours are considered together.
    """
    h, w = frame_shape[:2]
    red_mask = create_red_mask(hsv)
    blue_mask = create_blue_mask(hsv)

    # A larger kernel for red rim filling helps connect the rim pieces
    # of a partially occluded sign.
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, k_close)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, k_close)

    # Blue signs occasionally have white arrows inside, so close more
    # aggressively to fill the inside of the disc.
    k_close_blue = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blue_mask_clean = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, k_close_blue)
    blue_mask_clean = cv2.morphologyEx(blue_mask_clean, cv2.MORPH_OPEN, k_close)

    # Combined red+blue mask - aggressive close to merge rim and interior
    combined = cv2.bitwise_or(red_mask, blue_mask)
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_big)
    # Remove any blob touching the frame edge - those are sky/buildings.
    edge_mask = np.zeros_like(combined)
    cv2.rectangle(edge_mask, (3, 3), (w - 4, h - 4), 255, -1)
    combined = cv2.bitwise_and(combined, edge_mask)

    # Scale-relative size limits - the same code must work for both
    # 1920x1080 videos and any future resolution.
    min_r = max(18.0, min(w, h) * 0.016)
    max_r = min(160.0, min(w, h) * 0.18)

    candidates = []  # list of (cx, cy, radius, score, kind)

    # --- Source 1: red contours ---
    for c in cv2.findContours(red_mask, cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(c)
        if area < 200:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if radius < min_r or radius > max_r:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        perim = cv2.arcLength(c, True)
        if not is_circular_candidate(cx, cy, radius, area, bw, bh, perim):
            continue
        # Reject solid red blobs - real prohibition signs have a red
        # rim with a non-red (white/blue) interior.  A solid red
        # region is more likely a helmet, jacket or stop light.
        if _is_solid_colour(hsv, cx, cy, radius, 'red'):
            continue
        score = circle_score(area, perim, radius, bw, bh)
        candidates.append((cx, cy, radius, score, 'red'))

    # --- Source 2a: blue contours ---
    for c in cv2.findContours(blue_mask_clean, cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(c)
        if area < 200:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if radius < min_r or radius > max_r:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        perim = cv2.arcLength(c, True)

        # Reject rectangles - check bounding box aspect ratio directly
        bb_aspect = max(bw, bh) / float(min(bw, bh, 1))
        if bb_aspect > 1.5:
            continue

        if not is_circular_candidate(cx, cy, radius, area, bw, bh, perim):
            continue
        # Reject solid blue blobs - real mandatory direction signs
        # have a white arrow inside, not a uniform blue region.
        if _is_solid_colour(hsv, cx, cy, radius, 'blue'):
            continue
        score = circle_score(area, perim, radius, bw, bh)
        candidates.append((cx, cy, radius, score, 'blue'))

    # --- Source 2b: Hough circles on the blue mask (catches signs whose
    # interior arrow splits the blue contour).  Only keep circles that
    # contain a meaningful number of blue pixels. ---
    hough_circles = cv2.HoughCircles(
        blue_mask_clean, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
        param1=80, param2=18, minRadius=int(min_r), maxRadius=int(max_r * 0.85))
    if hough_circles is not None:
        for hc in hough_circles[0]:
            cx, cy, r = float(hc[0]), float(hc[1]), float(hc[2])
            if r < min_r or r > max_r * 0.85:
                continue
            # Check actual blue fill inside the Hough circle
            mask_circ = np.zeros_like(blue_mask_clean)
            cv2.circle(mask_circ, (int(cx), int(cy)), max(2, int(r * 0.9)),
                       255, -1)
            inside = cv2.countNonZero(cv2.bitwise_and(blue_mask_clean, mask_circ))
            fill = inside / float(max(1, math.pi * (r * 0.9) ** 2))
            if fill < 0.40:  # Increased from 0.30 - need more solid blue
                continue
            # Reject Hough circles that fall on a large rectangular sign.
            # A real circular sign has roughly equal width and height of
            # blue pixels around it; a billboard or painted wall has
            # blue stretching far beyond the circle in one direction.
            x = max(0, int(cx - r)); y = max(0, int(cy - r))
            x2 = min(w, int(cx + r)); y2 = min(h, int(cy + r))
            roi = blue_mask_clean[y:y2, x:x2]
            roi_pix = max(1, roi.size)
            if inside / roi_pix > 0.50:  # Stricter (was 0.55)
                continue  # Likely a rectangle, not a disc.
            # Wider scan: look at a 3x3 region centred on the candidate.
            # If the blue region there has an elongated bounding box,
            # the Hough circle is sitting on a billboard.
            scan = int(r * 2.5)
            xa = max(0, int(cx - scan)); ya = max(0, int(cy - scan))
            xb = min(w, int(cx + scan)); yb = min(h, int(cy + scan))
            big_roi = blue_mask_clean[ya:yb, xa:xb]
            contours_big, _ = cv2.findContours(
                big_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_big:
                biggest = max(contours_big, key=cv2.contourArea)
                bx, by, bbw, bbh = cv2.boundingRect(biggest)
                if bbw > 0 and bbh > 0:
                    ar = max(bbw, bbh) / float(min(bbw, bbh))
                    big_area = cv2.contourArea(biggest)
                    big_perim = cv2.arcLength(biggest, True)
                    if big_perim > 0:
                        big_circ = 4 * math.pi * big_area / (big_perim ** 2)
                        # Reject if the enclosing blob is highly
                        # elongated (a billboard) AND not very circular.
                        if ar > 1.3 and big_circ < 0.40:  # Much stricter
                            continue
                    # Also reject if ROI itself is very wide/tall
                    if ar > 1.5:
                        continue
            bw = bh = int(r * 2)
            score = 0.6 + 0.4 * min(fill / 0.7, 1.0)
            candidates.append((cx, cy, r, score, 'blue_hough'))

    # --- Source 3: combined red+blue (e.g. "no parking") ---
    for c in cv2.findContours(combined, cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(c)
        if area < 500:
            continue

        # IMPORTANT: Check circularity of the combined contour BEFORE minEnclosingCircle
        # Rectangles will have very low circularity even after morphological closing
        perimeter = cv2.arcLength(c, True)
        if perimeter > 0:
            contour_circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
            # Reject contours that are not circular enough
            if contour_circularity < 0.65:  # Strict circularity check
                continue

        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if radius < min_r or radius > max_r:
            continue
        x, y, bw, bh = cv2.boundingRect(c)

        # KEY CHECK: For a true circle, the enclosing circle should closely
        # match the bounding box. For a rectangle, the enclosing circle is
        # significantly larger than the bounding box (by ~sqrt(2) or more).
        half_diag = math.sqrt((bw/2)**2 + (bh/2)**2)
        if half_diag > 0 and radius / half_diag > 1.15:
            # Enclosing circle is much larger than BB diagonal = likely rectangle
            continue

        perim = cv2.arcLength(c, True)
        if not is_circular_candidate(cx, cy, radius, area, bw, bh, perim):
            continue
        # Reject blobs that extend to the frame edge - those are
        # sky/road/buildings rather than a sign.
        if x <= 2 or y <= 2 or (x + bw) >= w - 2 or (y + bh) >= h - 2:
            continue
        # Reject blobs whose bounding box is too elongated - we are
        # specifically hunting for circles. Stricter check here.
        aspect = max(bw, bh) / float(min(bw, bh)) if min(bw, bh) > 0 else 99
        if aspect > 1.25:  # Much stricter for combined (was 1.55)
            continue
        # Reject candidates that coincide with the big Hàng Xanh sign
        # (a large rectangle).  The rectangle has very low fill relative
        # to its bounding box.
        bb_fill = area / float(max(1, bw * bh))
        if bb_fill < 0.55:  # Stricter (was 0.45)
            continue
        # IMPORTANT: Reject large detections at top of frame - likely rectangular signs
        if cy < 0.25 * h and radius > 35:
            continue
        score = circle_score(area, perim, radius, bw, bh)
        candidates.append((cx, cy, radius, score, 'combined'))

    return candidates


def deduplicate_candidates(candidates, frame_shape):
    """NMS-like suppression across detection sources.

    The scene may contain dozens of small blue blobs (sky edges, painted
    stripes, icons inside large rectangular signs).  After NMS we still
    prune the leftovers by:

    * removing candidates whose score is below a per-source threshold,
    * dropping Hough circles whose radius is much smaller than the
      largest blue circle in the same frame (they tend to be noise),
    * removing the top-of-frame rectangles that hold car icons rather
      than circular signs.
    """
    h, w = frame_shape[:2]
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda c: c[3], reverse=True)
    kept = []
    for cx, cy, r, score, kind in candidates:
        overlap = False
        for kcx, kcy, kr, _, _ in kept:
            dist = math.hypot(cx - kcx, cy - kcy)
            if dist < max(0.7 * (r + kr), 14.0):
                overlap = True
                break
        if not overlap:
            kept.append((cx, cy, r, score, kind))

    # Prune weak Hough candidates - they often correspond to small
    # blue spots (icons inside a large rectangular sign) rather than
    # a circular traffic sign.
    filtered = []
    blue_hough = [k for k in kept if k[4] == 'blue_hough']
    if blue_hough:
        max_hough_r = max(k[2] for k in blue_hough)
    else:
        max_hough_r = 0
    for cx, cy, r, score, kind in kept:
        # Reject tiny Hough circles - they are not real signs.
        if kind == 'blue_hough' and r < 0.5 * max_hough_r and len(blue_hough) > 4:
            continue
        # Reject Hough circles that are clearly too large - they fit a
        # large rectangular billboard rather than a circular sign.
        if kind == 'blue_hough' and r > 70:
            continue
        # Reject Hough circles near the top of the frame - those are
        # usually icon symbols on a rectangular sign.
        if kind == 'blue_hough' and cy < 0.18 * h and score < 0.85:
            continue
        # Reject blue detections at top of frame (rectangular signs like Hàng Xanh)
        if cy < 0.20 * h and r > 60:
            continue
        # Real traffic signs are on poles, NOT in the lower 40% of frame
        # (that's where motorcycles, cars, people are)
        if cy > 0.60 * h:
            continue
        if cy > 0.50 * h and score < 0.85:
            continue
        # General minimum score to keep
        if score < 0.60:
            continue
        filtered.append((cx, cy, r, score, kind))
    return filtered


# ============================================================================
# STEP 5: TEMPORAL TRACKING - keep detections stable across frames
# ============================================================================
class TemporalSignFilter:
    """Smooth a detection list over time and reject one-frame colour blobs."""

    def __init__(self, min_hits=6, max_gap=3, max_tracks=999):
        self.tracks = []
        self.min_hits = min_hits
        self.max_gap = max_gap
        self.max_tracks = max_tracks

    def update(self, candidates):
        boxes = []
        for cx, cy, r, score, kind in candidates:
            x = int(cx - r)
            y = int(cy - r)
            bw = int(2 * r)
            bh = int(2 * r)
            boxes.append((x, y, bw, bh, score, kind, cx, cy, r))

        used = set()
        for t in self.tracks:
            tx, ty = t['cx'], t['cy']
            best, best_d = -1, float('inf')
            for i, b in enumerate(boxes):
                if i in used:
                    continue
                d = math.hypot(b[6] - tx, b[7] - ty)
                limit = max(35.0, 0.8 * max(t['r'], b[8]))
                if d < limit and d < best_d:
                    best, best_d = i, d
            if best >= 0:
                b = boxes[best]
                t['cx'] = 0.65 * t['cx'] + 0.35 * b[6]
                t['cy'] = 0.65 * t['cy'] + 0.35 * b[7]
                t['r'] = 0.65 * t['r'] + 0.35 * b[8]
                t['hits'] += 1
                t['gap'] = 0
                used.add(best)
            else:
                t['gap'] += 1

        for i, b in enumerate(boxes):
            if i not in used:
                self.tracks.append({
                    'cx': b[6], 'cy': b[7], 'r': b[8],
                    'hits': 1, 'gap': 0, 'score': b[4]})

        # Drop old tracks
        self.tracks = [t for t in self.tracks if t['gap'] <= self.max_gap]
        # Established tracks
        active = [t for t in self.tracks if t['hits'] >= self.min_hits]
        # Sort by score, then hits
        active.sort(key=lambda t: (t['score'], t['hits']), reverse=True)
        return active[:self.max_tracks]


# ============================================================================
# STEP 6: DRAW BOUNDING BOX
# ============================================================================
def draw_bounding_boxes(frame, tracks):
    output = frame.copy()
    for t in tracks:
        cx, cy, r = t['cx'], t['cy'], t['r']
        # Pad the box a little so the rim is fully inside the rectangle.
        pad = max(3, int(r * 0.10))
        x = int(cx - r) - pad
        y = int(cy - r) - pad
        w = int(2 * r) + 2 * pad
        h = int(2 * r) + 2 * pad
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(output, (int(cx), int(cy)), 4, (0, 255, 0), -1)
    return output


def add_student_id(frame, student_id):
    output = frame.copy()
    cv2.rectangle(output, (5, 5), (230, 42), (0, 0, 0), -1)
    cv2.putText(output, student_id, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    return output


# ============================================================================
# MAIN PROCESSING
# ============================================================================
def process_video(input_path, output_path):
    print(f"Processing: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
        return False

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Properties: {frame_width}x{frame_height}, {fps:.3f} fps, "
          f"{total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps,
                          (frame_width, frame_height))
    if not out.isOpened():
        cap.release()
        return False

    tracker = TemporalSignFilter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        candidates = find_circle_candidates(hsv, frame.shape)
        candidates = deduplicate_candidates(candidates, frame.shape)
        active = tracker.update(candidates)
        output = draw_bounding_boxes(frame, active)
        output = add_student_id(output, STUDENT_ID)
        out.write(output)

    cap.release()
    out.release()
    print(f"Done: {output_path}")
    return True


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video1_input = os.path.join(script_dir, "video1.mp4")
    video2_input = os.path.join(script_dir, "video2.mp4")
    video1_output = os.path.join(script_dir, "523H0054_video1.mp4")
    video2_output = os.path.join(script_dir, "523H0054_video2.mp4")

    print("=" * 60)
    print("Traffic Sign Detection - Circular Signs Only")
    print(f"Student ID: {STUDENT_ID}")
    print("=" * 60)

    success1 = process_video(video1_input, video1_output)
    print("\n" + "-" * 60 + "\n")
    success2 = process_video(video2_input, video2_output)

    print("\n" + "=" * 60)
    if success1 and success2:
        print("All videos processed successfully!")
    else:
        print("Some videos failed to process.")
    print("=" * 60)
