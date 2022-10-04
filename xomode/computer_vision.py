import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter


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


def ellipses(img):
    # Load picture, convert to grayscale and detect edges
    image_gray = img
    edges = canny(image_gray, sigma=2.0,
                  low_threshold=0.55, high_threshold=0.8)

    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    result = hough_ellipse(edges, accuracy=20, threshold=250,
                           min_size=100, max_size=120)
    result.sort(order='accumulator')

    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]

    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    img[cy, cx] = (0, 0, 255)
    # Draw the edge (white) and the resulting ellipse (red)
    # edges = color.gray2rgb(img_as_ubyte(edges))
    edges[cy, cx] = (250, 0, 0)

    fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
                                    sharex=True, sharey=True)

    ax1.set_title('Original picture')
    ax1.imshow(img)

    ax2.set_title('Edge (white) and result (red)')
    ax2.imshow(edges)

    plt.show()
    return


