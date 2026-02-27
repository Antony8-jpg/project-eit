""" IMPORTS """

import pyrealsense2 as rs
import numpy as np
import cv2

""" RECHTHOEK """
    
ROI_CX, ROI_CY = 320, 240   # midden van beeld (voor 640x480)
ROI_W, ROI_H = 480, 360

""" FUNCTIES """

def create_blob_detector():
    # Parameters object for simple blob detector, and assign parameters
    params = cv2.SimpleBlobDetector_Params()

    # Filter by color
    params.filterByColor = True
    params.blobColor = 255  

    # Threshold (Binary pixel thresholding)
    params.minThreshold = 0
    params.thresholdStep = 10
    params.maxThreshold = 255

    # Filter by Area (number of pixels)
    params.filterByArea = True
    params.minArea = 200    # 10x10 pixels
    params.maxArea = 10000  # 500x500 pixels

    # Filter by Circularity (How circular the blob)
    params.filterByCircularity = True
    params.minCircularity = 0.5  # 0 is not a circle, 1 is a perfect circle

    # Create a detector with the parameters
    return cv2.SimpleBlobDetector_create(params)

def detect_blobs(src):
    # Converteer naar HSV
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    # HSV bereik voor geel (kan je tunen!)
    lower_yellow = np.array([30, 100, 100])
    upper_yellow = np.array([45, 255, 255])

    # Maak mask
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Optioneel: ruis verwijderen
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Detect blobs op mask (wit = 255)
    kpts = blob_detector.detect(mask)

    return kpts

"""

def check_thresholds(src, thresholdStep=20):

    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    
    for th in range(0, 255-thresholdStep, thresholdStep):
        mask = cv2.inRange(gray, th, th+thresholdStep)
        cv2.imshow(str(th) + "-" + str(th+thresholdStep), mask)

"""
    
def draw_blobs(img, kpts):
    return cv2.drawKeypoints(img, kpts, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def contour_roundness(cnt):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0:
        return 0

    return (4 * np.pi * area) / (perimeter * perimeter)

def detect_most_round_blob(src, roi):
    x, y, w, h = roi

    # Crop naar ROI
    roi_img = src[y:y+h, x:x+w]

    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([45, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue

        score = contour_roundness(cnt)

        if score > best_score:
            best_score = score
            best_cnt = cnt

    # Zet contour-coördinaten terug naar full image coördinaten
    if best_cnt is not None:
        best_cnt = best_cnt + np.array([[[x, y]]], dtype=best_cnt.dtype)

    return best_cnt, best_score, mask


def draw_most_round(img, cnt, score):

    if cnt is None:
        return img

    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)

    cv2.circle(img, center, radius, (0,0,255), 2)
    return img

def ball_pos_from_circle(cnt):
    if cnt is None:
        return None
    (x, y), r = cv2.minEnclosingCircle(cnt)
    return int(x), int(y), int(r)

def draw_text_topleft(img, text):
    cv2.putText(img, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return img



""" MAIN """

if __name__ == '__main__':

    webcam = False

    # open webcam video stream
    if webcam:
        cap = cv2.VideoCapture(0)
        
    else:
        # Create the pipeline which handles all connected realsense devices
        pipeline = rs.pipeline()
        
        # Create a configuration object
        config = rs.config()
        
        # Enable some sensors
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)      # Color map uint8 (x3 channels)
        
        # Start the pipeline
        profile = pipeline.start(config)


    # The blob detector object
    blob_detector = create_blob_detector()

   

    # Run detection
    try:
        while True:
            frames = pipeline.wait_for_frames()
            if not frames:
                continue
            
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
        
            if not depth_frame or not color_frame:
                continue
        
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
        
            ROI_X = int(ROI_CX - ROI_W/2)
            ROI_Y = int(ROI_CY - ROI_H/2)
            
            roi = (ROI_X, ROI_Y, ROI_W, ROI_H)
            
            cv2.rectangle(color_image,
                          (ROI_X, ROI_Y),
                          (ROI_X+ROI_W, ROI_Y+ROI_H),
                          (255, 0, 0), 2)
            cnt, score, mask = detect_most_round_blob(color_image, roi)
            color_image = draw_most_round(color_image, cnt, score)
            pos = ball_pos_from_circle(cnt)
            
            if pos is not None:
                x, y, r = pos
                color_image = draw_text_topleft(color_image, f"Ball: x={x}, y={y}, r={r}")
            else:
                color_image = draw_text_topleft(color_image, "Ball: not found")
            
            cv2.imshow("Mask ROI", mask)   # debug: toont enkel ROI-mask
            
        
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
        
            cv2.imshow("Depth Image", depth_colormap)
            cv2.imshow("Color Image", color_image)
        
            if cv2.waitKey(1) == 27:
                break
            
            if score > 0.7:
                # accept
                pass
    
    finally:
        if webcam:
            # When everything done, release the capture
            cap.release()
        else:
            pipeline.stop()
            # finally, close the window
            cv2.destroyAllWindows()