import cv2
import numpy as np
import time

# =========================
# Mouse
# =========================
mouse_xy = (0, 0)
recalibrate_requested = False

def on_mouse(event, x, y, flags, param):
    global mouse_xy, recalibrate_requested
    mouse_xy = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        recalibrate_requested = True

# =========================
# Edges
# =========================
def compute_edges(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges = cv2.Canny(blur, 50, 200)
    return edges

# =========================
# Step 1: Find center from largest circle-ish contour
# =========================
def largest_circle_center_from_edges(edges, min_area=2000):
    """
    Returns (cx, cy) estimated from the largest 'round-ish' contour.
    If fails, returns image center.
    """
    H, W = edges.shape

    # Keep morphology mild (do NOT glue rings together)
    e = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_r = -1

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if r > best_r:
            best_r = r
            best = (int(round(x)), int(round(y)))

    if best is None:
        return (W // 2, H // 2), e

    return best, e

# =========================
# Step 2: Polar unwrap + radial profile -> radii peaks
# =========================
def radial_profile_from_edges(edges, center, r_max=None):
    """
    Unwrap edges to polar image around center.
    Returns profile[r] = mean edge strength at radius r.
    """
    H, W = edges.shape
    cx, cy = center

    if r_max is None:
        # max radius until we hit image border
        r_max = int(min(cx, cy, W - 1 - cx, H - 1 - cy))
        r_max = max(r_max, 10)

    # cv2.warpPolar output size: (radius, angle) if we set (angle, radius)??:
    # OpenCV uses dsize = (width, height). For WARP_POLAR_LINEAR:
    # width  = angle samples, height = radius samples.
    angle_samples = 720
    polar = cv2.warpPolar(
        edges,
        (angle_samples, r_max),
        (cx, cy),
        r_max,
        flags=cv2.WARP_POLAR_LINEAR
    )
    # polar shape: (r_max, angle_samples)
    # radial profile = mean over angles
    profile = polar.mean(axis=1)  # length r_max
    return profile, polar

def find_ring_radii(profile, n_rings=3, min_sep=25, smooth_ksize=21):
    """
    Find up to n_rings peak radii from radial profile.
    """
    prof = profile.astype(np.float32)

    # smooth to reduce noise
    if smooth_ksize % 2 == 0:
        smooth_ksize += 1
    prof_s = cv2.GaussianBlur(prof.reshape(-1, 1), (1, smooth_ksize), 0).ravel()

    # normalize
    if prof_s.max() > 1e-6:
        prof_s = prof_s / prof_s.max()

    # find local maxima
    peaks = []
    for r in range(1, len(prof_s) - 1):
        if prof_s[r] > prof_s[r - 1] and prof_s[r] > prof_s[r + 1]:
            peaks.append((prof_s[r], r))

    # sort by peak height
    peaks.sort(key=lambda t: t[0], reverse=True)

    # pick top peaks with minimum separation (avoid thick/double edge)
    chosen = []
    for val, r in peaks:
        if val < 0.12:  # ignore tiny peaks (tune if needed)
            continue
        if all(abs(r - rr) >= min_sep for rr in chosen):
            chosen.append(r)
        if len(chosen) == n_rings:
            break

    chosen.sort(reverse=True)  # biggest radius first
    return chosen, prof_s

# =========================
# Zone map from concentric circles
# =========================
def build_zone_map(H, W, center, radii_desc):
    """
    radii_desc = [R1, R2, R3] biggest->smallest (up to 3)
    zone:
      0 outside biggest
      1 inside biggest (outside 2nd)
      2 inside 2nd (outside 3rd)
      3 inside 3rd
    """
    zone = np.zeros((H, W), np.uint8)
    cx, cy = center

    if len(radii_desc) == 0:
        return zone

    masks = []
    for r in radii_desc[:3]:
        m = np.zeros((H, W), np.uint8)
        cv2.circle(m, (cx, cy), int(r), 255, -1)
        masks.append(m)

    zone[masks[0] == 255] = 1
    if len(masks) >= 2:
        zone[masks[1] == 255] = 2
    if len(masks) >= 3:
        zone[masks[2] == 255] = 3

    return zone

def zone_map_view(zone_map):
    return (zone_map * 80).astype(np.uint8)

# =========================
# Calibration on one frame
# =========================
def calibrate(bgr, debug=True):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = compute_edges(gray)

    center, edges_proc = largest_circle_center_from_edges(edges, min_area=2000)
    profile, polar = radial_profile_from_edges(edges, center, r_max=None)
    radii, prof_s = find_ring_radii(profile, n_rings=3, min_sep=25, smooth_ksize=21)

    H, W = gray.shape
    zone_map = build_zone_map(H, W, center, radii)

    print("Center:", center, "Radii:", radii)
    uniq, cnt = np.unique(zone_map, return_counts=True)
    print("Zone distribution:", dict(zip(uniq.tolist(), cnt.tolist())))

    dbg = None
    if debug:
        dbg = bgr.copy()
        cx, cy = center
        cv2.circle(dbg, (cx, cy), 3, (0, 0, 255), -1)
        for r in radii:
            cv2.circle(dbg, (cx, cy), int(r), (0, 255, 0), 2)
        cv2.putText(dbg, f"center=({cx},{cy}) radii={radii}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return zone_map, center, radii, edges, edges_proc, polar, dbg


# =========================
# Main
# =========================
if __name__ == "__main__":
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise RuntimeError("Cannot open camera")

    cv2.namedWindow("view")
    cv2.setMouseCallback("view", on_mouse)

    ret, bgr0 = vid.read()
    if not ret:
        raise RuntimeError("Camera read failed")

    zone_map, center, radii, edges, edges_proc, polar, dbg = calibrate(bgr0, debug=True)

    cv2.imshow("Edges", edges)
    cv2.imshow("Edges processed", edges_proc)
    cv2.imshow("Polar", polar)
    cv2.imshow("Zone map", zone_map_view(zone_map))
    if dbg is not None:
        cv2.imshow("Calibration", dbg)

    H, W = zone_map.shape

    while True:
        ret, bgr = vid.read()
        if not ret:
            break

        # Recalibrate on click
        if recalibrate_requested:
            recalibrate_requested = False
            zone_map, center, radii, edges, edges_proc, polar, dbg = calibrate(bgr, debug=True)
            H, W = zone_map.shape
            cv2.imshow("Edges", edges)
            cv2.imshow("Edges processed", edges_proc)
            cv2.imshow("Polar", polar)
            cv2.imshow("Zone map", zone_map_view(zone_map))
            if dbg is not None:
                cv2.imshow("Calibration", dbg)

        x, y = mouse_xy
        x = int(np.clip(x, 0, W - 1))
        y = int(np.clip(y, 0, H - 1))
        z = int(zone_map[y, x])

        vis = bgr.copy()
        cv2.circle(vis, (x, y), 6, (255, 0, 0), -1)
        cv2.putText(vis, f"x={x} y={y} zone={z} (click=recalibrate)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        # draw rings
        cx, cy = center
        cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
        for r in radii:
            cv2.circle(vis, (cx, cy), int(r), (0, 255, 0), 2)

        cv2.imshow("view", vis)

        if cv2.waitKey(1) == 27:
            break

    vid.release()
    cv2.destroyAllWindows()