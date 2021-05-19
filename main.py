import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    import logging
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    logger.info("info")

    vid = cv2.VideoCapture(0)
    ret, frame = vid.read()
 
    while(ret):
        ret, frame = vid.read()
        frame = cv2.resize(frame, (400,300))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret,frame = cv2.threshold(frame,128,255,cv2.THRESH_BINARY)

        plt.imshow(frame)
        plt.pause(0.01)

        if cv2.waitKey(1)  & 0xFF == ord('q'):
            vid.release()
            ret = False
            break

    plt.close()        


if __name__ == "__main__":
    main()
