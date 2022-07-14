import dataclasses
import glob
import h5py
import numpy as np


@dataclasses.dataclass
class Data:
    signal: float           # intensity shape = (freqs, ranges)
    channel: int            # channel name from digital_rf
    frequency: float        # transmitting freqs in Hz shape = (freqs, )
    id: int                 # link identification integer?
    ranges: int             # range gates in meters shape = (ranges, )
    chirp_rate: float       # chirp rate in Hz per 1 second
    sampling_rate: float    # sampling rate in Hz
    station: str            # station name string
    time0: float            # initial time in second since epoch


def load_data(filepath, station='*', year='*', month='*', day='*', epoch='*', freq_bins=20, range_bins=650):
    files = glob.glob(f"{filepath}/{year}.{month}.{day}/{station}/lfm*{epoch}.00.h5")
    files.sort()
    n = len(files)
    data = Data
    data.signal = np.zeros((n, freq_bins, range_bins), dtype=float)
    data.channel = np.chararray((n, ), itemsize=3)
    data.frequency = np.zeros((n, freq_bins), dtype=float)
    data.id = np.zeros((n, ), dtype=int)
    data.ranges = np.zeros((n, range_bins), dtype=float)
    data.chirp_rate = np.zeros((n, ), dtype=float)
    data.sampling_rate = np.zeros((n, ), dtype=float)
    data.station = np.chararray((n, ), itemsize=3)
    data.time0 = np.zeros((n, ), dtype=float)
    for idx, file in enumerate(files):
        # 'S', 'ch', 'freqs', 'id', 'ranges', 'rate', 'sr', 'station_name', 't0'
        f = h5py.File(file, 'r')
        data.signal[idx, :, :] = normalize_intensity(f['S'][()])
        data.channel[idx] = f['ch'][()]
        data.frequency[idx, :] = f['freqs'][()]/1e6
        data.id[idx] = f['id'][()]
        data.ranges[idx, :] = f['ranges'][()]
        data.chirp_rate[idx] = f['rate'][()]
        data.sampling_rate[idx] = f['sr'][()]
        data.station[idx] = f['station_name'][()]
        data.time0[idx] = f['t0'][()]
    return data


def normalize_intensity(x, cutoff=0.0, default=1e-3):
    noise_floor = np.median(x)
    std_floor = np.median(np.abs(x - noise_floor))
    x = (x - noise_floor) / std_floor
    x[x < cutoff] = default
    return x


def shit():
    for x in range(20):
        f0 = x / 2
        for fi, f in enumerate(fl):
            h = h5py.File(f, "r")
            S = h["S"][()]
            fr = h["freqs"][()] / 1e6
            fidx[:] = np.argmin(np.abs(f0 - fr[:]))
            S0[x, fi, :] = S[fidx, :] / np.median(np.abs(S[fidx, :]))

