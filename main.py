import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    import logging
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    logger.info("info")

    analysis_frame_depth = 10 
    input_frame_size = [600,800]

    vid = cv2.VideoCapture(0)
    ret, frame = vid.read()

    stacked_frame = np.zeros(input_frame_size,)
    logger.debug(f"0 {stacked_frame.shape}")


    while(ret):
        ret, frame = vid.read()
        frame = cv2.resize(frame, (input_frame_size[1], input_frame_size[0]))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        stacked_frame = np.dstack((stacked_frame, frame))
        logger.debug(f"loop {stacked_frame.shape}")
        last_frame_number = stacked_frame.shape[2]
        cv2.imshow("frame", frame)

        if last_frame_number > analysis_frame_depth:
            np.save("stacked_frame", stacked_frame)
            logger.debug(stacked_frame.shape)
            stacked_frame = stacked_frame[:,:,1:]
            logger.debug(stacked_frame.shape)

        sf_var = (np.var(stacked_frame, axis=2))
        sf_var = min_max(sf_var)
        plt.imshow(sf_var)
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
