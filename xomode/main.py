import numpy as np
import argparse
import h5py
import cv2
import matplotlib.pyplot as plt
# From this project
import data_handler
import computer_vision


def main():
    f = h5py.File('/home/arl203/iono/2022.05.02/oul/2022-05-02/lfm_ionogram-Oulu-001-1651449714.00.h5', 'r')
    # f = h5py.File('/home/arl203/iono/2022.05.02/oul/2022-05-02/lfm_ionogram-Oulu-001-1651458774.00.h5', 'r')
    freq = f['freqs'][()] / 1e6
    rng = f['ranges'][()] / 1e3
    S = f['S'][()]
    S = np.swapaxes(S, 0, 1)
    S = data_handler.normalize(S, cutoff=0.1)
    S = data_handler.normalize_range(S, cutoff=0.1)
    g = data_handler.despeckle(S, std=50.0, size=1)
    S = data_handler.intensity_db(S)
    g = data_handler.intensity_db(g)

    X, Y = np.meshgrid(freq, rng)
    plt.figure(figsize=[12, 5])
    plt.subplot(121)
    plt.pcolormesh(X, Y, S, shading='auto', cmap='inferno', vmin=0.0)
    plt.colorbar(label='SNR [dB]')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('One-way range offset [km]')
    # plt.grid()
    plt.subplot(122)
    plt.pcolormesh(X, Y, g, shading='auto', cmap='inferno', vmin=0.0)
    plt.colorbar(label='Amplitude')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('One-way range offset [km]')
    # plt.grid()
    plt.tight_layout()
    plt.show()

    # convert to a 255 bit image
    img = np.flipud(g)
    img = img / np.max(img) * 255
    img = img.astype(np.uint8)
    cv2.imshow('initial stage', img)

    # apply a blur + erode + dilate morphology
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 5)
    img = computer_vision.morphology(img, n=3)
    cv2.imshow('morphology stage', img)

    # apply a partial ellipse Hough transform
    # todo: make this work
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    plt.subplot(121), plt.imshow(img)
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(cimg)
    plt.title('Hough Transform'), plt.xticks([]), plt.yticks([])
    plt.show()

    # try canny edge detect
    edges = cv2.Canny(img, 100, 200)
    cv2.imshow('edges', edges)

    # contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
    #     if len(cnt) > 10:
    #         cv2.drawContours(pnts, cnt, -1, (255, 255, 255), 1)
    #
    # cv2.imshow('traced', pnts)
    cv2.waitKey()
    return None


if __name__ == '__main__':
    main()




