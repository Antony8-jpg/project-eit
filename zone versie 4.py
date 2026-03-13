""" IMPORTS """

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import numpy as np
import pyrealsense2 as rs
import os


""" GLOBALS """
saved_edges = None
last_capture = None
last_contours = []
last_zones_vis = None
last_zone_masks = None
save_counter = 0
save_folder = "saved_edges"

""" Constanten """

ROI_X = 120
ROI_Y = 60
ROI_W = 400
ROI_H = 360


""" FUNCTIONS """

# edge detection
def computeEdges(frame):
    edges = cv2.GaussianBlur(frame, (3, 3), sigmaX=1.5, sigmaY=1.5)
    edges = cv2.Canny(edges, threshold1=50, threshold2=200, L2gradient=False)
    return edges # edges is de image in zwart-wit

# rectangle of interest: witte rechthoek -> image wordt gehouden, zwart -> image wordt zwart
def apply_roi_mask(image, x, y, w, h):
    mask = np.zeros_like(image)
    mask[y:y+h, x:x+w] = (255, 255, 255)
    return cv2.bitwise_and(image, mask) #combineer masker en beeld 


""" 
# witte rand van roi tekenen # houden? als het niet goed lukt om randen te tekenen?
def draw_roi_rectangle(image, x, y, w, h, color=255, thickness=2):
    output = image.copy()
    cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
    return output
"""

# edges duidelijker maken door close : gaten sluiten en dilate : randen dikker maken
def preprocess_for_contours(edges_roi):
    kernel_close = np.ones((3, 3), np.uint8)
    proc = cv2.morphologyEx(edges_roi, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    proc = cv2.dilate(proc, kernel_close, iterations=1)
    return proc 

# ruis weghalen en contour sorteren van klein naar groot
def select_main_contours(contours, min_area=100):
    valid_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            valid_contours.append(cnt)

    valid_contours.sort(key=cv2.contourArea)
    return valid_contours[-3:]

# zones maken adhv de contours
def build_zone_masks_from_contours(image_shape, contours, roi_x, roi_y, roi_w, roi_h): # roi mag weg reeds gedef -arne
    h, w = image_shape[:2]

    if len(contours) != 3:
	# raise ValueError('A very specific bad thing happened.')
        return None

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = 255
    
    # m = masker van contour (m1 is alles binnen contour 1)
    masks = []
    for cnt in contours:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)
        mask = cv2.bitwise_and(mask, roi_mask)
        masks.append(mask)

    m1, m2, m3 = masks

    # y_min en y_max van grootste contour
    y_licht, x_licht = np.where(m3>0)
    # global y_min,y_max
    y_min = np.min(y_licht)
    y_max = np.max(y_licht)

    zone1 = m1
    zone2 = cv2.subtract(m2, m1)
    zone3 = cv2.subtract(m3, m2)
    zone4 = cv2.subtract(roi_mask, m3)

    return (zone1, zone2, zone3, zone4) , y_min, y_max

"""
# visualisatie -> niet essentieel
def visualize_zones((zone1, zone2, zone3, zone4), contours, roi_x, roi_y, roi_w, roi_h): # roi mag weg reeds gedef. -arne
    h, w = zone1.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    vis[:] = (70, 70, 70)
    vis[zone3 > 0] = (130, 130, 130)
    vis[zone2 > 0] = (190, 190, 190)
    vis[zone1 > 0] = (255, 255, 255)

    if len(contours) >= 1:
        cv2.drawContours(vis, [contours[0]], -1, (0, 0, 255), 2)
    if len(contours) >= 2:
        cv2.drawContours(vis, [contours[1]], -1, (0, 255, 0), 2)
    if len(contours) >= 3:
        cv2.drawContours(vis, [contours[2]], -1, (255, 0, 0), 2)

    return vis
"""

# coordinaat -> zone
def get_zone_at_pixel(x, y, zone_masks):
    if zone_masks is None:
        return None

    zone1, zone2, zone3, zone4 = zone_masks # tuple met 4 elementen

    # check of x,y binnen het beeld zitten
    if y < 0 or y >= zone1.shape[0] or x < 0 or x >= zone1.shape[1]:
        return None

    if zone1[y, x] > 0:
        return 1
    elif zone2[y, x] > 0:
        return 2
    elif zone3[y, x] > 0:
        return 3
    elif zone4[y, x] > 0:
        return 4
    else:
        return None

""" oude versie
def analyze_capture(captured_edges):
    processed = preprocess_for_contours(captured_edges)
    contours, hierarchy = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ring_contours = select_main_contours(contours, min_area=100)
    #ring_contours = select_ring_contours(contours, ROI_X, ROI_Y, ROI_W, ROI_H)

    zones_vis = None
    zone_masks = None

    if len(ring_contours) == 3:
        zone_masks = build_zone_masks_from_contours(
            captured_edges.shape,
            ring_contours,
            ROI_X, ROI_Y, ROI_W, ROI_H
        )

        if zone_masks is not None:
            zone1, zone2, zone3, zone4 = zone_masks
            zones_vis = visualize_zones(
                zone1, zone2, zone3, zone4,
                ring_contours,
                ROI_X, ROI_Y, ROI_W, ROI_H
            )

    return ring_contours, zones_vis, zone_masks
"""

def init(activate_check_zone=True):
    # camera instellen
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # camera starten
    pipeline.start(config)

    try:
        for _ in range(10):
            pipeline.wait_for_frames() 

        frames = pipeline.wait_for_frames() # wachten tot nieuwe frame beschikbaar is, bevat meerdere type frames
        color_frame = frames.get_color_frame() # haalt color_frame eruit

        if not color_frame:
            raise RuntimeError("Geen color frame ontvangen van de RealSense.")

        # frame omzetten in Numpy ndarray : (480,640,3) : hoogte, breedte, rgb kleuren
        frame = np.asanyarray(color_frame.get_data())
        # frame grayscalen -> numpy.ndarray grootte (480,640)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

        # functies gebruiken om de grayscale om te vormen naar 3 duidelijke contours
        edges = computeEdges(frame_gray) # numpy.ndarray met waardes 0 of 255
        edges_roi = apply_roi_mask(edges, ROI_X, ROI_Y, ROI_W, ROI_H) # numpy.ndarray maar buiten ROI nu 0 -> alles buiten ROI weghalen
        processed = preprocess_for_contours(edges_roi) # edges processen ( duidelijkere contours )
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # contours vinden -> list van np.ndarray
        main_contours = select_main_contours(contours, min_area=100) # ruis weghalen -> lijst van np.ndarray met 3 elementen

	# deze code aanpassen, of de zone checker aanpassen want die geeft ook None als het buiten de zones valt of als andere gebruiken -Arne
        if len(main_contours) != 3:
	    # raise ValueError('A very specific bad thing happened.')
            return None # return init(True)
        

        # zones maken
        zone_masks, y_min,y_max = build_zone_masks_from_contours(
            edges_roi.shape,
            main_contours,
            ROI_X, ROI_Y, ROI_W, ROI_H
        )

	""" code arne voor check beeld
    # Save the image
	image_path = "beeld.png"
	beeld = visualize_zones(zone_masks, main_contours, roi_x, roi_y, roi_w, roi_h)
	cv2.imwrite(image_path, beeld)
	
	if activate_check_zone:
	# Step 2: Open the image using OpenCV
	loaded_image = cv2.imread(image_path)

	# Display the image in a window
	cv2.imshow("Loaded Image", loaded_image)
	cv2.waitKey(0)  # Wait for a key press to close the window
	cv2.destroyAllWindows()  # Close the window

	# Step 3: Request user input
	user_input = bool(input("1/0: "))
	print("You entered:", user_input) # het zo maken dat als het antwoord 1 is de init succesvol was en als het antwoord 0 is er een nieuwe poging komt om de zones te vinden (soms staat er iemand in de weg waardoor hij ze niet kan vinden
	if not user_intput: return init(activate_check_zone)"""
	
	



        #return zone_masks, y_min, y_max
        return zone_masks	# return statement hierboven nodig voor lednr. te bepalen -Arne  + zie boven return onder if statement

    finally:
        pipeline.stop()

"""
def init():
    saved_edges = None
    # last_capture = None
    # last_contours = []
    # last_zones_vis = None
    last_zone_masks = None
    # save_counter = 0
    # save_folder = "saved_edges"

    os.makedirs(save_folder, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    try:
        for _ in range(10):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            raise RuntimeError("Geen eerste color frame ontvangen van de RealSense.")

        # prev_frame = np.asanyarray(color_frame.get_data())
        # prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        #cv2.namedWindow("RealSense Color")
        cv2.namedWindow("Edges")
        cv2.namedWindow("Last Capture")
        cv2.namedWindow("Zones Screenshot")

        #cv2.setMouseCallback("Edges", mouse_callback_edges)
        #cv2.setMouseCallback("Zones Screenshot", mouse_callback_zones)

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            edges = computeEdges(frame_gray)

            edges_roi = apply_roi_mask(edges, ROI_X, ROI_Y, ROI_W, ROI_H)
            edges_display = draw_roi_rectangle(edges_roi, ROI_X, ROI_Y, ROI_W, ROI_H, color=255, thickness=2)

            saved_edges = edges_roi.copy()

            cv2.imshow("RealSense Color", frame)
            cv2.imshow("Edges", edges_display)

            if last_capture is not None:
                cv2.imshow("Last Capture", last_capture)


            if last_zones_vis is not None:
                cv2.imshow("Zones Screenshot", last_zones_vis)

            k = cv2.waitKey(1)
            if k == 27:
                break

            # prev_frame = frame_gray.copy()

            # filename = os.path.join(save_folder, f"edges_{save_counter:04d}.png")
            # cv2.imwrite(filename, saved_edges)
            # print("Saved:", filename)

            last_capture = saved_edges.copy()
            last_contours, last_zones_vis, last_zone_masks = analyze_capture(last_capture)
            #a,b = y_min, y_max
            return last_zone_masks

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
"""


if __name__ == '__main__':
    zones = intit()
    get_zone_at_pixel(300, 300, zones):