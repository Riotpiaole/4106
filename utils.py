import sys
import cv2
import numpy as np
from PIL import Image
from multiprocessing import Process


def multi_process_wrapper(iterable):
    '''multi_process_wrapper
        Wrapping given function and a constant variable for CPUs wise
        multi-processingself.

    Arguments:
        iterable: iterable chunkable array
        func: function that will be execute each iteration

        inner_args  *args for function arguments
                    step log_size for each step
    >>> @multi_process_wrapper([1,2,3])
    >>> def dosomething(*args):
    ...     arg , iteration = args
    ...     print(arg , iteration)
    >>> dosomething(10)
    10 1
    10 2
    10 3

    >>> # generate 10 process for executing the soemthing_over_array
    >>> @multi_process_wrapper( list(chunk([i for i in range 1000] , 100) ) )
    >>> def something_over_array(*args)
    ...     iteration = args
    ...     print( iteartion)
    >>> something_over_array()
    10

    '''
    def wrapper(func):
        def inner_wrapper(*args, **kwargs):
            processes = []
            step = 0
            for index, item in enumerate(iterable):
                if index != 0:
                    step = step + len(item)

                process = Process(target=func, args=[*args, item, step])
                process.start()
                print("Starting process.....")
            for process in processes:
                process.join()
            print("Process completed.....")
            return
        return inner_wrapper
    return wrapper


def bgr_to_rgb(img):
    '''Convert image from bgr to rgb'''
    b, g, r = np.dsplit((img), 3)
    return np.dstack((r, g, b))


def rgb_to_bgr(img):
    '''Convert image from rgb to bgr'''
    r, g, b = np.dsplit((img), 3)
    return np.dstack((b, g, r))


def bgr_to_ycrcb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)


def showImage(img, name="Output_Image"):
    cv2.imshow(name, img)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()


def rgb_to_ycrcb(img):
    return bgr_to_ycrcb(rgb_to_bgr(img))


def rgb_to_ycrcb_channel_first(image, upscale=2):

    yCrCb_image = cv2.cvtColor(
        image.astype(np.uint8),
        cv2.COLOR_RGB2YCrCb)
    y, Cr, Cb = np.dsplit((yCrCb_image), 3)
    h, w = y.shape[:2]
    y_train = np.array([cv2.resize(y, (h // 2, w // 2))])
    return y_train.astype(np.float64), Cr, Cb, y.transpose((2, 0, 1))


def ycrcb2rgb(yf, cr, cb):
    result = []
    for (y_c, cr_c, cb_c) in zip(yf, cr, cb):

        y = y_c.detach().cpu().numpy() * 255.0
        cr = cr_c.detach().cpu().numpy()
        cb = cb_c.detach().cpu().numpy()

        y = y.clip(0, 255).astype(np.uint8)

        image = np.vstack((y, cb, cr)).astype(np.float32)
        image -= (255.0 / 2)

        result.append(image)
    return np.array(result)
