import sys
import cv2
import torch
import numpy as np
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


def rgb_to_ycrcb_channel_first(image):

    yCrCb_image = cv2.cvtColor(
        image.astype(np.uint8),
        cv2.COLOR_RGB2YCrCb)
    y, Cr, Cb = np.dsplit((yCrCb_image), 3)
    h, w = y.shape[:2]
    y_train = np.array([cv2.resize(y, (h // 2, w // 2))])
    naive = cv2.resize(
        # shrink the image by it's sizes//2
        cv2.resize(image, y_train.shape[:2]),
        y.shape[:2])  # Upscale the image by twice to present the result

    naive_mean = np.mean(naive, axis=0).astype(np.float32)
    naive_std = np.std(naive, axis=0).astype(np.float32)

    naive -= naive_mean
    naive /= naive_std

    return y_train.astype(np.float64), Cr, Cb, naive.astype(np.float32)


def ycrcb2rgb(yf, cr, cb):
    # detach from cuda into numpy
    yf = yf.detach().cpu().numpy()
    cr = cr.detach().cpu().numpy().transpose((0, 3, 1, 2))
    cb = cb.detach().cpu().numpy().transpose((0, 3, 1, 2))

    y = yf.clip(0, 255).astype(np.uint8)
    result = np.hstack(
        (y, cr, cb)
    ).astype(np.float32)

    result /= 255

    return result


def toTensor(arr, cuda=False):
    val = torch.from_numpy(arr)
    return val.cuda() if cuda else val


def visualizes(image, model):
    y, cr, cb, y_origin = rgb_to_ycrcb_channel_first(image)
    y = np.array([y])

    y = toTensor(y, True)
    y_pred = model.forward(y.float()).detach().cpu().numpy()
    y_pred = y_pred[0].reshape(cb.shape)

    result = np.dstack((y_pred, cr, cb)).clip(0, 255)
    naive = cv2.resize(
        # shrink the image by it's sizes//2
        cv2.resize(image, y.shape[2:]),
        result.shape[:2])  # Upscale the image by twice to present the result
    return naive, cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_YCrCb2RGB)


if __name__ == "__main__":
    img = np.zeros((400, 400, 3), dtype = "uint8")
    cv2.rectangle(img, (0, 0), (200, 200), (0, 255, 0), 5)
    b , g , r = np.dsplit(img)

    showImage(img)
    # Creating line



