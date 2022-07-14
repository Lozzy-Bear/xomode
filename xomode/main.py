import numpy as np
import argparse
import data_handler
import h5py
import cv2
import scipy.signal as sig
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

# def main():
#     data = data_handler.load_data(filepath)
#     return None
#
#
# if __name__ == '__main__':
#     main()


def normalize_intensity(x, cutoff=0.1, default=1e-3):
    noise_floor = np.mean(x, axis=0)
    noise_floor = np.tile(noise_floor, (650, 1))
    std_floor = np.mean(np.abs(x - noise_floor), axis=0)
    std_floor = np.tile(std_floor, (650, 1))
    x = (x - noise_floor) / std_floor
    x[x < cutoff] = default
    return x


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
    plt.figure()
    plt.subplot(231)
    plt.imshow(x, origin='lower')
    plt.colorbar()
    plt.subplot(232)
    plt.imshow(k, origin='lower')
    plt.colorbar()
    plt.subplot(233)
    plt.imshow(y, origin='lower')
    plt.colorbar()

    plt.subplot(234)
    plt.imshow(np.real(x_fft), origin='lower')
    plt.colorbar()
    plt.subplot(235)
    plt.imshow(np.real(k_fft), origin='lower')
    plt.colorbar()
    plt.subplot(236)
    plt.imshow((np.real(np.fft.ifftshift(x_fft/k_fft))), origin='lower')
    plt.colorbar()
    plt.show()
    return y


f = h5py.File('/home/arl203/iono/2022.05.02/oul/2022-05-02/lfm_ionogram-Oulu-001-1651449654.00.h5', 'r')
keys = f.keys()
for key in keys:
    d = np.asarray(f[key][()])
    print(key, type(d), d.shape)
freq = f['freqs'][()] / 1e6
rng = f['ranges'][()] / 1e3
S = f['S'][()]
S = np.swapaxes(S, 0, 1)

S = normalize_intensity(S, cutoff=1.0)
S = 20*np.log10(S)
S[S == np.nan] = 1e-3
S[S <= 0.0] = 0.0


gaussian = sig.windows.gaussian(S.shape[0], 50.0, sym=True)
g = np.zeros_like(S)
for l in range(S.shape[1]):
    fft = np.fft.fft(S[:, l])
    fft = np.fft.fftshift(fft)
    # fft[gaussian > 0.1] = 0.0
    g[:, l] = np.fft.ifft(np.fft.fftshift(fft*gaussian))
    # plt.figure()
    # plt.plot(fft)
    # plt.plot(gaussian * np.max(fft))
    # plt.show()

h = np.zeros((g.shape[0], g.shape[0]))
h[0:g.shape[0], 0:g.shape[1]] = g
h = ndi.median_filter(h, size=2)
g = h[0:g.shape[0], 0:g.shape[1]]
g[g < 10.0] = 0.0
g[g > 0.0] = 255
g[0:100, :] = 0
g[500::, :] = 0
g[:, 200::] = 0

# g = kernel_deconvolve(g)
# S[S > np.max(S)-10.0] = 1e-3

X, Y = np.meshgrid(freq, rng)
plt.figure(figsize=[12, 5])
plt.subplot(121)
plt.pcolormesh(X, Y, S, shading='auto', cmap='inferno', vmin=0.0, vmax=30.0)
plt.colorbar(label='SNR [dB]')
plt.xlabel('Frequency [MHz]')
plt.ylabel('One-way range offset [km]')

plt.subplot(122)
plt.pcolormesh(X, Y, g, shading='auto', cmap='inferno', vmin=0.0, vmax=30.0)
plt.colorbar(label='Amplitude')
plt.xlabel('Frequency [MHz]')
plt.ylabel('One-way range offset [km]')
plt.grid()
plt.show()

plt.tight_layout()

S = np.flipud(g)
img = np.zeros((S.shape[0], S.shape[1], 3))
img[:, :, 0] = S/np.max(S) * 255
img = img.astype(np.uint8)
# img_blur = cv2.GaussianBlur(img, (1, 5), 0)
# cv2.imshow('blur', img_blur)
# cv2.waitKey()

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5), anchor=(0, 0))
edges = cv2.dilate(img, kernel)
edges = cv2.Canny(edges, 100, 200)
cv2.imshow('edges', edges)

# linesP = cv2.HoughLines(edges, 1, np.pi / 180, 1, None, 0, 0)
# if linesP is not None:
#     print('adasd')
#     for i in range(0, len(linesP)):
#         l = linesP[i][0]
#         cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0, 255, 255), 3, cv2.LINE_AA)
# cv2.imshow('lines', img)

# flag = False
pnts = np.zeros_like(edges)
# print(np.min(pnts))
# for f in range(edges.shape[1]):
#     flag = False
#     r1 = np.nan
#     f1 = np.nan
#     for r in range(edges.shape[0]):
#         if edges[r, f] > 0 and flag:
#             rr = int(r1 + (r - r1)/2)
#             ff = int(f1 + (f - f1)/2)
#             pnts[rr, ff] = 255
#             flag = False
#         if edges[r, f] > 0 and not flag:
#             r1 = r
#             f1 = f
#             flag = True

contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours), contours[0])
for cnt in contours:
    if len(cnt) > 10:
        cv2.drawContours(pnts, cnt, -1, (255,255,255), 1)

cv2.imshow('traced', pnts)
cv2.waitKey()



# def trace_line(x):
#     for f in range(x.shape[1]):
#         for r in range(x.shape[0]):




