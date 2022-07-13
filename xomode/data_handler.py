import dataclasses
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib
import h5py
import numpy as np
import datetime
import pretty_plots


# 'S', 'ch', 'freqs', 'id', 'ranges', 'rate', 'sr', 'station_name', 't0'
@dataclasses.dataclass
class Data:
    signal: float        # intensity shape = (freqs, ranges)
    channel: int         # channel name from digital_rf
    freq: float          # transmitting freqs in Hz shape = (freqs, )
    id: int              # link identification integer?
    ranges: int          # range gates in meters shape = (ranges, )
    chirp_rate: float    # chirp rate in Hz per 1 second
    sampling_rate: str   # sampling rate in Hz
    station: str         # station name string
    time0: float         # initial time in second since epoch


def normalize(S):
    for fi in range(S.shape[0]):
        noise_floor = np.median(S[fi, :])
        std_floor = np.median(n.abs(S[fi, :] - noise_floor))
        S[fi, :] = (S[fi, :] - noise_floor) / std_floor
    S[S < 0] = 1e-3
    return (S)


data_dir = "/home/arl203/iono/2022.05.02/oul/2022-05-02"
fl = glob.glob(f"{data_dir}/lfm*.h5")
fl.sort()

epoch = datetime.datetime.utcfromtimestamp(0)
dt = datetime.datetime.strptime("2022-05-02", "%Y-%m-%d")
epoch = int((dt - epoch).total_seconds())
print(epoch)

fof2 = []
hmf = []
t = []

S0 = np.zeros([20, len(fl), 650])
distances = np.zeros([20, len(fl), 650])
times = np.zeros([20, len(fl), 650])

for x in range(20):
    f0 = x / 2
    print(f0)

    for fi, f in enumerate(fl):
        h = h5py.File(f, "r")
        keys = h.keys()
        print(keys)
        for key in keys:
            d = h[f'{key}'][()]
            print(key, type(d), d)
        print('exiting...')
        exit()
        try:
            h = h5py.File(f, "r")
            #        S=normalize(h["S"][()])
            S = h["S"][()]
            fr = h["freqs"][()] / 1e6
            tdist = h["ranges"][()]
            ttime = h["t0"][()]

            fidx = np.argmin(np.abs(f0 - fr))
            # print(S.shape)
            # print(fidx)
            S0[x, fi, :] = S[fidx, :] / np.median(np.abs(S[fidx, :]))
            distances[x, fi, :] = tdist
            times[x, fi, :] = ttime
            h.close()
            # print(f)
        except:
            distances[x, fi, :] = tdist
            times[x, fi, :] = times[x, fi - 1, :] + 60
            pass
            h.close()

colormaps = ['Greys', 'Purples', 'Blues', 'Oranges', 'Greens', 'Reds']
legend_elements = []

times = (np.asarray(times) - epoch) / 60 / 60

print(times[0, np.arange(len(fl) - 1), 0])  # -times[0,np.arange(len(fl)-1)+1,0])

print(times.shape, distances.shape)

for x in range(20):
    alpha_values = S0[x, :, :] / 20.0
    alpha_values[np.where(alpha_values > 1.0)] = 1.0

    plot_num = plt.pcolormesh(times[x, :, :], distances[x, :, :] / 1e3, S0[x, :, :], cmap=colormaps[int(x % 6)], vmin=0,
                              vmax=20, alpha=alpha_values)

    cmap = matplotlib.cm.get_cmap(colormaps[int(x % 6)])
    legend_elements.append(Patch(facecolor=cmap(0.75), label=f'{x / 2:.1f} MHz'))

norm = matplotlib.colors.Normalize(vmin=0, vmax=20)
sm = plt.cm.ScalarMappable(cmap='Greys', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Intensity')

plt.title('Color Coded Frequency Stack - 2022-05-02 - Oulu')
plt.xlabel('Time UT (Hours)')
plt.ylabel('One-way Distance (km)')
plt.legend(handles=legend_elements, loc='upper left')
plt.show()
