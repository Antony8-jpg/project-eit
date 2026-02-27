import cv2
import numpy as np

# =========================
# CONFIG: jouw 3 stralen (pixels)
# Grootste eerst!
# =========================
RADII = [228, 150, 67]  # <-- VERVANG door jouw 3 radii in pixels (R1>R2>R3)

# detectie van witte punt (middelpunt)
DOT_THR = 240      # verlaag naar 220-235 als punt soms niet gevonden wordt
DOT_MIN_AREA = 8   # verhoog als er ruispuntjes zijn

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
# Center from white dot
# =========================
def detect_white_dot_center(bgr, thr=240, min_area=8):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)

    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = -1
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        if area > best_area:
            best_area = area
            best = c

    H, W = gray.shape
    if best is None:
        return (W // 2, H // 2), mask, False  # fallback

    M = cv2.moments(best)
    if abs(M["m00"]) < 1e-6:
        x, y, w, h = cv2.boundingRect(best)
        cx, cy = x + w // 2, y + h // 2
    else:
        cx = int(round(M["m10"] / M["m00"]))
        cy = int(round(M["m01"] / M["m00"]))

    cx = int(np.clip(cx, 0, W - 1))
    cy = int(np.clip(cy, 0, H - 1))
    return (cx, cy), mask, True

# =========================
# Zone map from known radii
# =========================
def build_zone_map(H, W, center, radii_desc):
    """
    radii_desc: [R1, R2, R3] grootste->kleinste
    zone:
      0 buiten R1
      1 binnen R1 maar buiten R2
      2 binnen R2 maar buiten R3
      3 binnen R3
    """
    cx, cy = center
    radii = sorted([int(r) for r in radii_desc], reverse=True)

    zone = np.zeros((H, W), np.uint8)
    masks = []
    for r in radii[:3]:
        m = np.zeros((H, W), np.uint8)
        cv2.circle(m, (cx, cy), r, 255, -1)
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
# Main
# =========================
if __name__ == "__main__":
    # sanity check radii
    if len(RADII) != 3:
        raise ValueError("RADII moet exact 3 waarden hebben.")
    if not (RADII[0] > RADII[1] > RADII[2]):
        print("LET OP: RADII is niet strikt dalend. Ik sorteer ze automatisch.")

    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise RuntimeError("Cannot open camera")

    cv2.namedWindow("view")
    cv2.setMouseCallback("view", on_mouse)

    # init
    ret, bgr0 = vid.read()
    if not ret:
        raise RuntimeError("Camera read failed")

    center, dot_mask, ok = detect_white_dot_center(bgr0, thr=DOT_THR, min_area=DOT_MIN_AREA)
    H, W = bgr0.shape[:2]
    zone_map = build_zone_map(H, W, center, RADII)

    print("Init center:", center, "dot_found:", ok, "RADII:", sorted(RADII, reverse=True))

    cv2.imshow("dot_mask", dot_mask)
    cv2.imshow("zone_map", zone_map_view(zone_map))
    cv2.waitKey(1)

    while True:
        ret, bgr = vid.read()
        if not ret:
            break

        # recalibrate center on click (same radii, new center)
        if recalibrate_requested:
            recalibrate_requested = False
            center, dot_mask, ok = detect_white_dot_center(bgr, thr=DOT_THR, min_area=DOT_MIN_AREA)
            zone_map = build_zone_map(H, W, center, RADII)
            print("Recalibrate center:", center, "dot_found:", ok)

            cv2.imshow("dot_mask", dot_mask)
            cv2.imshow("zone_map", zone_map_view(zone_map))

        # zone under mouse
        x, y = mouse_xy
        x = int(np.clip(x, 0, W - 1))
        y = int(np.clip(y, 0, H - 1))
        z = int(zone_map[y, x])

        # draw
        vis = bgr.copy()
        cx, cy = center

        # draw circles with SAME center using your radii
        for r in sorted(RADII, reverse=True):
            cv2.circle(vis, (cx, cy), int(r), (0, 255, 0), 2)

        cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)   # center dot marker
        cv2.circle(vis, (x, y), 6, (255, 0, 0), -1)     # cursor marker

        cv2.putText(vis, f"center=({cx},{cy})  x={x} y={y} zone={z} (click=recenter)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        cv2.imshow("view", vis)

        k = cv2.waitKey(1)
        if k == 27:
            break

    vid.release()
    cv2.destroyAllWindows()