import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
import atexit

def nothing(val):
    pass

def exit_handler():
    plt.plot(ball_t, ball_x)
    plt.show()
    x_pos_arr = np.asarray(ball_x)
    time_arr = np.asarray(ball_t)
    np.savetxt("x_pos_pixels.csv", x_pos_arr, delimiter=",")
    np.savetxt("times.csv", time_arr, delimiter=",")
    #print(sum(ball_x)/len(ball_x))
    #print(sum(ball_y)/len(ball_y))
    #print(len(ball_r))
 
def initialize_trackbars():
    global hh, hl, sh, sl, vh, vl, wnd

    cv2.namedWindow("Colorbars") 
    
    hh='Hue High'
    hl='Hue Low'
    sh='Saturation High'
    sl='Saturation Low'
    vh='Value High'
    vl='Value Low'

    wnd = 'Colorbars'

    cv2.createTrackbar(hl, wnd,0,179,nothing)
    cv2.createTrackbar(hh, wnd,0,179,nothing)
    cv2.createTrackbar(sl, wnd,0,255,nothing)
    cv2.createTrackbar(sh, wnd,0,255,nothing)
    cv2.createTrackbar(vl, wnd,0,255,nothing)
    cv2.createTrackbar(vh, wnd,0,255,nothing)

def initialize():
    global cap
    #cap = cv2.VideoCapture(2)
    cap = cv2.VideoCapture("videos/length_videos/49.mp4")
    #initialize_trackbars()
    atexit.register(exit_handler)

def find_ball(frame, show_trackbars = False):
    frame = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hul, huh, sal, sah, val, vah = 0, 9, 230, 255, 85, 255

    #hul, huh, sal, sah, val, vah = 23, 40, 126, 255, 127, 255 #rob vals

    if show_trackbars:
        hul=cv2.getTrackbarPos(hl, wnd)
        huh=cv2.getTrackbarPos(hh, wnd)
        sal=cv2.getTrackbarPos(sl, wnd)
        sah=cv2.getTrackbarPos(sh, wnd)
        val=cv2.getTrackbarPos(vl, wnd)
        vah=cv2.getTrackbarPos(vh, wnd)

    HSVLOW = np.array([hul, sal, val])
    HSVHIGH = np.array([huh, sah, vah])
    thresh = cv2.inRange(hsv, HSVLOW, HSVHIGH)
    thresh = cv2.GaussianBlur(thresh, (7, 7), 0)
    floodfill = thresh.copy()
    h, w = thresh.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(floodfill, mask, (0,0), 255)
    floodfill_inv = cv2.bitwise_not(floodfill)
    thresh_filled = thresh | floodfill_inv    
    edged = cv2.Canny(thresh_filled, 30, 200)
    _, all_contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []

    for contour in all_contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if abs(max(w, h)/min(w, h)) < 1.1:
            if area > 400:
                filtered_contours.append(contour) 
                break

    if len(filtered_contours) < 1:
        return -1, -1, -1, thresh_filled

    ball_contour = filtered_contours[0]
    center, radius = cv2.minEnclosingCircle(ball_contour)
    
    return int(center[0]), int(center[1]), int(radius), thresh_filled
    
def main():
    global ball_t, ball_x, ball_r, ball_y
    ball_t, ball_x, ball_r, ball_y = [], [], [], []

    i = 0
    while(cap.isOpened()):
        i += 1
        ret,frame = cap.read()
        time_stamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        x_px, y_px, radius, thresh = find_ball(frame, show_trackbars = False)

        if x_px > -1:
            ball_t.append(time_stamp)
            ball_x.append(x_px)
            #ball_r.append(radius)
            #ball_y.append(y_px)
            cv2.circle(frame, (x_px, y_px), radius, (0, 255, 0), 2)
            cv2.circle(frame, (x_px, y_px), 2, (255, 0, 0), 2)
        
        cv2.imshow("contours", frame)
        cv2.imshow("thresh", thresh)
        if(cv2.waitKey(5) & 0xFF == ord("q")):
             break
    
    cap.release()
    cv2.destroyAllWindows()

def analyze_video(path_in, path_out):
    cap = cv2.VideoCapture(path_in + ".mp4")

    ball_t, ball_x, ball_r, ball_y = [], [], [], []

    i = 0
    ret = True
    while(cap.isOpened()):
        i += 1
        ret,frame = cap.read()
        if not ret: break
        #cv2.imshow("contours", frame)
        time_stamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        x_px, y_px, radius, thresh = find_ball(frame, show_trackbars = False)
        if x_px > -1:
            ball_t.append(time_stamp)
            ball_x.append(x_px)
            ball_r.append(radius)
            ball_y.append(y_px)
            cv2.circle(frame, (x_px, y_px), radius, (0, 255, 0), 2)
            cv2.circle(frame, (x_px, y_px), 2, (255, 0, 0), 2)
        
        cv2.imshow("contours", frame)
        #cv2.imshow("thresh", thresh)
        if(cv2.waitKey(5) & 0xFF == ord("q")):
             break
    
    cap.release()
    cv2.destroyAllWindows()
    
    plt.plot(ball_t, ball_x)
    plt.show()
    
    x_pos_arr = np.asarray(ball_x)
    time_arr = np.asarray(ball_t)
    y_pos_arr = np.asarray(ball_y)
    r_arr = np.asarray(ball_r)

    np.savetxt(path_out + "_x.csv", x_pos_arr, delimiter=",")
    np.savetxt(path_out + "_y.csv", y_pos_arr, delimiter=",")
    np.savetxt(path_out + "_r.csv", r_arr, delimiter=",")
    np.savetxt(path_out + "_t.csv", time_arr, delimiter=",")

def print_progress(arr, i):
    print("#"*(i+1) + "_"*(len(arr)-i-1)) 
    
if __name__ == "__main__":  
    #initialize_trackbars()
    masses = [95, 190, 285, 380, 570, 760]
    #masses = [95]
    for i in range(len(masses)): 
        analyze_video("videos/mass_videos/" + str(masses[i]), "raw_data/mass/" + str(masses[i]))
        print_progress(masses, i)  
    """
    lengths = [49, 61, 68, 79, 91, 104, 113, 123, 132, 142, 152, 167]
    for i in range(len(lengths)): 
        analyze_video("videos/length_videos/" + str(lengths[i]), "raw_data/length/" + str(lengths[i]))
        print_progress(lengths, i)
    """

    