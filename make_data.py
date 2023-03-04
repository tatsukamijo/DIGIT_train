# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from digit_interface.digit import Digit
import cv2
import time

# Set id for your DIGIT e.g. D20221
DIGIT_ID = "YOUR_DIGIT_ID"

def make_datasets():
    count = 0
    while True:
        frame = digit.get_frame()
        cv2.imshow("Frame", frame)
        # Save frame to specified path
        # Change here to collect frames for different classes
        digit.save_frame(f"Dxxxx/notouch/{count}_notouch.png")
        count += 1
        print(count)
        # Change count to the number of frames you want to collect
        if (cv2.waitKey(1) & 0xFF == ord("q")) or count == 1500:
            break

if __name__ == "__main__":
    digit = Digit(f"{DIGIT_ID}", "digit_name")
    digit.connect()
    digit.set_intensity(Digit.LIGHTING_MIN)
    time.sleep(1)
    digit.set_intensity(Digit.LIGHTING_MAX)
    # Change DIGIT resolution to QVGA
    qvga_res = Digit.STREAMS["QVGA"]
    digit.set_resolution(qvga_res)
    # Change DIGIT frame rate to 30fps
    fps_30 = Digit.STREAMS["QVGA"]["fps"]["30fps"]
    digit.set_fps(fps_30)
    make_datasets()
