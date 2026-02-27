""" IMPORTS """

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import numpy as np 
import time


""" FUNCTIONS """

def computeEdges(frame):
    edges = cv2.GaussianBlur(frame, (3, 3), sigmaX=1.5, sigmaY=1.5)
    edges = cv2.Canny(edges, threshold1=50, threshold2=200, L2gradient=False)
    return edges

def computeFlow(prev_frame, frame):
    flow = cv2.calcOpticalFlowFarneback(prev_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def flowToColor(flow):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype='uint8')
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb


""" MAIN """

if __name__ == "__main__":
    
    vid = cv2.VideoCapture(0)
    ret, prev_frame = vid.read() 
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    while True:
        ret, frame = vid.read()
    
        begin = time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = computeFlow(prev_frame, frame)
        edges = computeEdges(frame)
        elapsed_time = time.time() - begin
        print('Elapsed time = ' + str(elapsed_time) + '(s)')
    
        rgb = flowToColor(flow)
    
        cv2.imshow("Optical Flow", rgb)
        cv2.imshow("Edges", edges)
        
        # Escape
        k = cv2.waitKey(1)
        if k == 27:
            break
    
        prev_frame = frame
    
    vid.release()
    cv2.destroyAllWindows()
