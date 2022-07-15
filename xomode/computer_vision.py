import cv2
import numpy as np


def kernel_deconvolve(x):
    m = np.max(x)
    k = np.zeros_like(x)
    cx = int(x.shape[0]/2-4)
    cy = int(x.shape[1]/2-1.5)
    k[cx:cx+4, cy:cy+3] = np.array([[0.0, 0.1, 0.0],
                                    [0.4, 1.0, 0.4],
                                    [0.4, 1.0, 0.4],
                                    [0.0, 0.1, 0.0]]) * m
    x_fft = np.fft.fftshift(np.fft.fft2(x))
    k_fft = np.fft.fftshift(np.fft.fft2(k))
    y = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x_fft/k_fft)))
    y = np.real(y)
    return y


def morphology(x, n=1, blur=(3, 3), erode=(3, 3), dilate=(5, 5)):
    """

    Parameters
    ----------
        x : ndarray float
            (frequency bins, range bins) signal intensity.
        n : int
            number of morphology passes.
        blur : tuple int
            morphology kernel size (x, y).
        erode : tuple int
            morphology kernel size (x, y).
        dilate : tuple int
            morphology kernel size (x, y).

    Returns
    -------
        x : ndarray float
            (frequency bins, range bins) signal intensity image morphed to show lines.
    """
    for i in range(0, n, 1):
        if blur is not None:
            x = cv2.GaussianBlur(x, blur, 0)
        if erode is not None:
            k = cv2.getStructuringElement(cv2.MORPH_CROSS, erode, anchor=(0, 0))
            x = cv2.erode(x, k)
        if dilate is not None:
            k = cv2.getStructuringElement(cv2.MORPH_CROSS, dilate, anchor=(0, 0))
            x = cv2.dilate(x, k)
    return x




