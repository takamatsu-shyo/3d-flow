import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    import logging
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    logger.info("info")

    vid = cv2.VideoCapture(0)
    ret, frame = vid.read()

    stacked_frame = np.zeros((300,400),)
    logger.debug(f"0 {stacked_frame.shape}")


    while(ret):
        ret, frame = vid.read()
        frame = cv2.resize(frame, (400,300))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #ret,frame = cv2.threshold(frame,128,255,cv2.THRESH_BINARY)

        stacked_frame = np.dstack((stacked_frame, frame))
        logger.debug(f"loop {stacked_frame.shape}")
        last_frame_number = stacked_frame.shape[2]
        cv2.imshow("frame", frame)

        if last_frame_number > 100:
            np.save("stacked_frame", stacked_frame)
            logger.debug(stacked_frame.shape)
            stacked_frame = stacked_frame[:,:,1:]
            logger.debug(stacked_frame.shape)

        sf_var = (np.var(stacked_frame, axis=2))
        #cv2.imshow("sf_var", sf_var)"
        sf_var = min_max(sf_var)
        plt.imshow(sf_var)
        #plt.show()
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

        if cv2.waitKey(1)  & 0xFF == ord('q'):
            vid.release()
            ret = False
            break

    plt.close()        

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-0)/(max-0)
    return result


if __name__ == "__main__":
    main()
