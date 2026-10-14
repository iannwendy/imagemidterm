#!/usr/bin/env python3
"""Debug script - extract frames and show detailed detection info"""
import cv2
import numpy as np
import math
import os

STUDENT_ID = "523H0054"

def create_red_mask(hsv):
    return cv2.inRange(hsv, np.array([0, 55, 45]), np.array([14, 255, 255])) | \
           cv2.inRange(hsv, np.array([158, 55, 45]), np.array([179, 255, 255]))

def create_blue_mask(hsv):
    return cv2.inRange(hsv, np.array([95, 130, 60]), np.array([130, 255, 255]))

def _is_solid_colour(hsv, cx, cy, radius, colour):
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
    if perimeter <= 0 or radius <= 0 or bw * bh <= 0:
        return 0.0
    circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
    circle_fill = area / (math.pi * radius * radius)
    bb_fill = area / float(bw * bh)
    return (0.45 * min(circularity / 0.75, 1.0) +
            0.35 * min(circle_fill / 0.65, 1.0) +
            0.20 * min(bb_fill / 0.70, 1.0))

def is_circular_candidate(cx, cy, radius, area, bw, bh, perimeter):
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

def find_circle_candidates(hsv, frame_shape):
    h, w = frame_shape[:2]
    red_mask = create_red_mask(hsv)
    blue_mask = create_blue_mask(hsv)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, k_close)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, k_close)

    k_close_blue = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blue_mask_clean = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, k_close_blue)
    blue_mask_clean = cv2.morphologyEx(blue_mask_clean, cv2.MORPH_OPEN, k_close)

    combined = cv2.bitwise_or(red_mask, blue_mask)
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_big)
    edge_mask = np.zeros_like(combined)
    cv2.rectangle(edge_mask, (3, 3), (w - 4, h - 4), 255, -1)
    combined = cv2.bitwise_and(combined, edge_mask)

    min_r = max(18.0, min(w, h) * 0.016)
    max_r = min(160.0, min(w, h) * 0.18)

    candidates = []

    # Red contours
    for c in cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
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
        if _is_solid_colour(hsv, cx, cy, radius, 'red'):
            continue
        score = circle_score(area, perim, radius, bw, bh)
        candidates.append((cx, cy, radius, score, 'red'))

    # Blue contours
    for c in cv2.findContours(blue_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
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
        if _is_solid_colour(hsv, cx, cy, radius, 'blue'):
            continue
        score = circle_score(area, perim, radius, bw, bh)
        candidates.append((cx, cy, radius, score, 'blue'))

    # Hough circles
    hough_circles = cv2.HoughCircles(
        blue_mask_clean, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
        param1=80, param2=18, minRadius=int(min_r), maxRadius=int(max_r * 0.85))
    if hough_circles is not None:
        for hc in hough_circles[0]:
            cx, cy, r = float(hc[0]), float(hc[1]), float(hc[2])
            if r < min_r or r > max_r * 0.85:
                continue
            mask_circ = np.zeros_like(blue_mask_clean)
            cv2.circle(mask_circ, (int(cx), int(cy)), max(2, int(r * 0.9)), 255, -1)
            inside = cv2.countNonZero(cv2.bitwise_and(blue_mask_clean, mask_circ))
            fill = inside / float(max(1, math.pi * (r * 0.9) ** 2))
            if fill < 0.40:
                continue
            x = max(0, int(cx - r)); y = max(0, int(cy - r))
            x2 = min(w, int(cx + r)); y2 = min(h, int(cy + r))
            roi = blue_mask_clean[y:y2, x:x2]
            roi_pix = max(1, roi.size)
            if inside / roi_pix > 0.50:
                continue
            scan = int(r * 2.5)
            xa = max(0, int(cx - scan)); ya = max(0, int(cy - scan))
            xb = min(w, int(cx + scan)); yb = min(h, int(cy + scan))
            big_roi = blue_mask_clean[ya:yb, xa:xb]
            contours_big, _ = cv2.findContours(big_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_big:
                biggest = max(contours_big, key=cv2.contourArea)
                bx, by, bbw, bbh = cv2.boundingRect(biggest)
                if bbw > 0 and bbh > 0:
                    ar = max(bbw, bbh) / float(min(bbw, bbh))
                    big_area = cv2.contourArea(biggest)
                    big_perim = cv2.arcLength(biggest, True)
                    if big_perim > 0:
                        big_circ = 4 * math.pi * big_area / (big_perim ** 2)
                        if ar > 1.3 and big_circ < 0.40:
                            continue
                    if ar > 1.5:
                        continue
            bw = bh = int(r * 2)
            score = 0.6 + 0.4 * min(fill / 0.7, 1.0)
            candidates.append((cx, cy, r, score, 'blue_hough'))

    # Combined
    for c in cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(c)
        if area < 500:
            continue

        # Check circularity of the combined contour BEFORE minEnclosingCircle
        perimeter = cv2.arcLength(c, True)
        if perimeter > 0:
            contour_circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
            if contour_circularity < 0.65:  # Stricter
                continue

        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if radius < min_r or radius > max_r:
            continue
        x, y, bw, bh = cv2.boundingRect(c)

        half_diag = math.sqrt((bw/2)**2 + (bh/2)**2)
        if half_diag > 0 and radius / half_diag > 1.15:
            continue

        perim = cv2.arcLength(c, True)
        if not is_circular_candidate(cx, cy, radius, area, bw, bh, perim):
            continue
        if x <= 2 or y <= 2 or (x + bw) >= w - 2 or (y + bh) >= h - 2:
            continue
        aspect = max(bw, bh) / float(min(bw, bh)) if min(bw, bh) > 0 else 99
        if aspect > 1.25:
            continue
        bb_fill = area / float(max(1, bw * bh))
        if bb_fill < 0.55:
            continue
        if cy < 0.25 * h and radius > 35:
            continue
        score = circle_score(area, perim, radius, bw, bh)
        candidates.append((cx, cy, radius, score, 'combined'))

    return candidates

def deduplicate_candidates(candidates, frame_shape):
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

    filtered = []
    blue_hough = [k for k in kept if k[4] == 'blue_hough']
    if blue_hough:
        max_hough_r = max(k[2] for k in blue_hough)
    else:
        max_hough_r = 0
    for cx, cy, r, score, kind in kept:
        if kind == 'blue_hough' and r < 0.5 * max_hough_r and len(blue_hough) > 4:
            continue
        if kind == 'blue_hough' and r > 70:
            continue
        if kind == 'blue_hough' and cy < 0.18 * h and score < 0.85:
            continue
        if cy < 0.20 * h and r > 60:
            continue
        if cy > 0.60 * h:
            continue
        if cy > 0.50 * h and score < 0.85:
            continue
        if score < 0.60:
            continue
        filtered.append((cx, cy, r, score, kind))
    return filtered

class TemporalSignFilter:
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

        self.tracks = [t for t in self.tracks if t['gap'] <= self.max_gap]
        active = [t for t in self.tracks if t['hits'] >= self.min_hits]
        active.sort(key=lambda t: (t['score'], t['hits']), reverse=True)
        return active[:self.max_tracks]

def draw_debug_frame(frame, candidates, deduped, tracks, frame_num, fps, h, w):
    output = frame.copy()

    # Draw ALL raw candidates in cyan (before any filtering)
    for cx, cy, r, score, kind in candidates:
        x, y = int(cx - r), int(cy - r)
        # Label with kind and score
        cv2.rectangle(output, (x-2, y-2), (x + int(2*r)+2, y + int(2*r)+2),
                     (255, 255, 0), 1)

    # Draw deduplicated candidates in yellow
    for cx, cy, r, score, kind in deduped:
        x, y = int(cx - r), int(cy - r)
        cv2.rectangle(output, (x-3, y-3), (x + int(2*r)+3, y + int(2*r)+3),
                     (0, 255, 255), 2)
        # Add text label
        label = f"{kind[0]}:{score:.2f}"
        cv2.putText(output, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Draw final tracks in green
    for t in tracks:
        cx, cy, r = t['cx'], t['cy'], t['r']
        pad = max(3, int(r * 0.10))
        x = int(cx - r) - pad
        y = int(cy - r) - pad
        w = int(2 * r) + 2 * pad
        h = int(2 * r) + 2 * pad
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Info
    cv2.putText(output, f"Frame: {frame_num} | Time: {frame_num/fps:.1f}s | H:{h} W:{w}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(output, f"Raw: {len(candidates)} | Deduped: {len(deduped)} | Tracks: {len(tracks)}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Legend
    cv2.rectangle(output, (frame.shape[1]-200, 10), (frame.shape[1]-10, 100), (0,0,0), -1)
    cv2.putText(output, "Cyan=Raw", (frame.shape[1]-190, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    cv2.putText(output, "Yellow=Dedup", (frame.shape[1]-190, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    cv2.putText(output, "Green=Final", (frame.shape[1]-190, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.rectangle(output, (5, 5), (230, 42), (0, 0, 0), -1)
    cv2.putText(output, STUDENT_ID, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return output

def extract_key_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Sample every 30 frames (1 second)
    sample_frames = list(range(0, total, 30))

    tracker = TemporalSignFilter()

    print(f"Extracting frames from {video_path}")
    print(f"FPS: {fps}, Total frames: {total}, Resolution: {w}x{h}")

    frame_num = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num in sample_frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            candidates = find_circle_candidates(hsv, frame.shape)
            deduped = deduplicate_candidates(candidates, frame.shape)
            tracks = tracker.update(candidates)

            # Print debug info
            if len(candidates) > 0 or len(deduped) > 0:
                print(f"\n--- Frame {frame_num} (t={frame_num/fps:.1f}s) ---")
                print(f"  Raw candidates ({len(candidates)}):")
                for cx, cy, r, score, kind in candidates:
                    cy_pct = cy / h * 100
                    print(f"    {kind}: cx={cx:.0f} cy={cy:.0f} ({cy_pct:.1f}%) r={r:.0f} score={score:.3f}")
                if len(deduped) > 0:
                    print(f"  Deduped ({len(deduped)}):")
                    for cx, cy, r, score, kind in deduped:
                        cy_pct = cy / h * 100
                        print(f"    {kind}: cx={cx:.0f} cy={cy:.0f} ({cy_pct:.1f}%) r={r:.0f} score={score:.3f}")

            debug_frame = draw_debug_frame(frame, candidates, deduped, tracks, frame_num, fps, h, w)
            fname = os.path.join(output_dir, f"debug_{frame_num:05d}.jpg")
            cv2.imwrite(fname, debug_frame)
            saved += 1

            if saved % 20 == 0:
                print(f"  Saved {saved} frames...")

        frame_num += 1

    cap.release()
    print(f"Done! Saved {saved} frames to {output_dir}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "debug_frames")

    extract_key_frames(os.path.join(script_dir, "video1.mp4"), output_dir)
    print()
    extract_key_frames(os.path.join(script_dir, "video2.mp4"), output_dir)
