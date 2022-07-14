import numpy as np
import argparse
import data_handler
import h5py
import cv2


# def main():
#     data = data_handler.load_data(filepath)
#     return None
#
#
# if __name__ == '__main__':
#     main()
def normalize_intensity(x, cutoff=0.1, default=1e-3):
    # noise_floor = np.median(x, axis=0)
    # std_floor = np.median(np.abs(x - noise_floor))
    # print(x.shape, noise_floor.shape, std_floor.shape)
    # x = (x - noise_floor) / std_floor
    # x[x < cutoff] = default
    for l in range(x.shape[1]):
        m = np.median(x[:, l])
        s = np.median(np.abs(x[:, l] - m))
        x[:, l] = (x[:, l] - m) / s
    x[x < cutoff] = default
    return x


import matplotlib.pyplot as plt
f = h5py.File('F:\\projects\\iono\\2022.05.02\\oul\\2022-05-02\\lfm_ionogram-Oulu-001-1651449654.00.h5', 'r')
keys = f.keys()
for key in keys:
    d = np.asarray(f[key][()])
    print(key, type(d), d.shape)
freq = f['freqs'][()] / 1e6
rng = f['ranges'][()] / 1e3
S = f['S'][()]
S = np.swapaxes(S, 0, 1)


g = np.zeros_like(S)
for l in range(S.shape[1]):
    fft = np.fft.fft(np.abs(S[:, l]))
    fft = np.fft.fftshift(fft)
    g[:, l] = np.abs(fft)


S = normalize_intensity(S, 3.0)
S = 20*np.log10(S)
S[S < 0.0] = 0.0
S[S > np.max(S)-10.0] = 1e-3
S[S == np.nan] = 1e-3
X, Y = np.meshgrid(freq, rng)
print(X.shape, Y.shape, S.shape)
plt.figure(1)
plt.pcolormesh(X, Y, S, shading='auto', cmap='inferno')
plt.colorbar(label='SNR [dB]')
plt.xlabel('Frequency [MHz]')
plt.ylabel('One-way range offset [km]')


plt.figure(2)
plt.pcolormesh(X, Y, g, shading='auto', cmap='inferno')
plt.colorbar()

# S[g > 10e3] = 1e-3
g = g/np.max(g)
plt.figure(3)
plt.pcolormesh(X, Y, S*g, shading='auto', cmap='inferno')
plt.colorbar()

plt.show()

# S = np.flipud(S)
# img = np.zeros((S.shape[0], S.shape[1], 3))
# img[:, :, 0] = S/np.max(S) * 255
# img_blur = cv2.GaussianBlur(img, (1, 5), 0)
# cv2.imshow('blur', img_blur)
# cv2.waitKey()
#
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5), anchor=(0, 0))
# edges = cv2.dilate(img_blur, kernel)
# # edges = cv2.Canny(img_blur, 100, 200)
# cv2.imshow('edges', edges)
# cv2.waitKey()


# data_dir = "/home/arl203/iono/2022.05.02/oul/2022-05-02"
# fl = glob.glob(f"{data_dir}/lfm*.h5")
# fl.sort()

# for n in range(20):
#     f0 = x / 2
#     for idx, f in enumerate(fl):
#         try:
#             h = h5py.File(f, "r")
#             distances[idx, n, :] = h["ranges"][()]
#             times[idx, n, :] = h["t0"][()]
#             freq = h["freqs"][()] / 1e6
#             S = h["S"][()]
#             fidx = np.argmin(np.abs(f0 - freq))
#             S0[idx, n, :] = S[fidx, :] / np.median(np.abs(S[fidx, :]))
#
#
# times = (np.asarray(times) - epoch) / 60 / 60
#
# colormaps = ['Greys', 'Purples', 'Blues', 'Oranges', 'Greens', 'Reds']
# legend_elements = []
# for x in range(20):
#     alpha_values = S0[x, :, :] / 20.0
#     alpha_values[np.where(alpha_values > 1.0)] = 1.0
#     plot_num = plt.pcolormesh(times[x, :, :], distances[x, :, :] / 1e3, S0[x, :, :], cmap=colormaps[int(x % 6)], vmin=0,
#                               vmax=20, alpha=alpha_values)
#     cmap = matplotlib.cm.get_cmap(colormaps[int(x % 6)])
#     legend_elements.append(Patch(facecolor=cmap(0.75), label=f'{x / 2:.1f} MHz'))
# norm = matplotlib.colors.Normalize(vmin=0, vmax=20)
# sm = plt.cm.ScalarMappable(cmap='Greys', norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm)
# cbar.set_label('Intensity')
# plt.title('Color Coded Frequency Stack - 2022-05-02 - Oulu')
# plt.xlabel('Time UT (Hours)')
# plt.ylabel('One-way Distance (km)')
# plt.legend(handles=legend_elements, loc='upper left')
# plt.show()
