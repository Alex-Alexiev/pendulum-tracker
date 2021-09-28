import cv2
import numpy as np
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

input_file = "videos/rob_01.m4v"
output_file = "output.avi"

x_start, x_end = 0, 100
y_start, y_end = 0, 200
rotate = True

start_time, end_time = 2, 4

ffmpeg_extract_subclip(input_file, start_time, end_time, targetname="temp.m4v")

cap = cv2.VideoCapture("temp.m4v")
fourcc = cv2.VideoWriter_fourcc(*'XVID')


out = cv2.VideoWriter(output_file,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (200, 100))

i = 0
while i < 100:
    i += 1
    ret, frame = cap.read()
    sky = frame[y_start:y_end, x_start:x_end]
    cv2.imshow('Video', sky)

    out.write(sky)

    if cv2.waitKey(1) == 27:
        exit(0)

out.release()