import dataclasses
import glob
import h5py
import numpy as np
import scipy.signal as sig
import scipy.ndimage as ndi


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
        data.signal[idx, :, :] = normalize(f['S'][()])
        data.channel[idx] = f['ch'][()]
        data.frequency[idx, :] = f['freqs'][()]/1e6
        data.id[idx] = f['id'][()]
        data.ranges[idx, :] = f['ranges'][()]
        data.chirp_rate[idx] = f['rate'][()]
        data.sampling_rate[idx] = f['sr'][()]
        data.station[idx] = f['station_name'][()]
        data.time0[idx] = f['t0'][()]
    return data


def normalize(x, cutoff=0.1, default=1e-3):
    """
    Normalize intensity data per frequency bin and take the log10.

    Parameters
    ----------
        x : ndarray float
            (frequency bins, range bins) signal intensity.
        cutoff : float
            noise floor intensity cutoff under which is mapped to default.
        default : float
            the default minimum intensity value; ideally above 0.0 for taking log10.

    Returns
    -------
        x : ndarray float
            (frequency bins, range bins) frequency bin normalized signal intensity.
    """
    noise_floor = np.mean(x, axis=0)
    noise_floor = np.tile(noise_floor, (x.shape[0], 1))
    std_floor = np.mean(np.abs(x - noise_floor), axis=0)
    std_floor = np.tile(std_floor, (x.shape[0], 1))
    x = (x - noise_floor) / std_floor
    x[x < cutoff] = default
    return x


def normalize_range(x, cutoff=0.1, default=1e-3):
    """
    Normalize intensity data per frequency bin and take the log10.

    Parameters
    ----------
        x : ndarray float
            (frequency bins, range bins) signal intensity.
        cutoff : float
            noise floor intensity cutoff under which is mapped to default.
        default : float
            the default minimum intensity value; ideally above 0.0 for taking log10.

    Returns
    -------
        x : ndarray float
            (frequency bins, range bins) frequency bin normalized signal intensity.
    """
    noise_floor = np.mean(x, axis=1)
    noise_floor = np.tile(noise_floor, (x.shape[1], 1)).T
    std_floor = np.mean(np.abs(x - noise_floor), axis=1)
    std_floor = np.tile(std_floor, (x.shape[1], 1)).T
    x = (x - noise_floor) / std_floor
    x[x < cutoff] = default
    return x


def intensity_db(x, min=1e-3):
    x = 20 * np.log10(x)
    x[x == np.nan] = min
    x[x <= 0.0] = min
    return x


def despeckle(x, std=50.0, size=2, notch=10):
    # Apply a gaussian window per frequency bin
    gaussian = sig.windows.gaussian(x.shape[0], std, sym=True)[:, np.newaxis]
    fft = np.fft.fftshift(np.fft.fft(x, axis=0))
    mid = int(fft.shape[0]/2 + 1)
    fft[(mid-notch):(mid+notch), :] = 0.0
    x = np.fft.ifft(np.fft.ifftshift(fft * gaussian), axis=0)
    x = np.abs(x)
    # Apply a median filter
    h = np.zeros((x.shape[0], x.shape[0]))
    h[0:x.shape[0], 0:x.shape[1]] = x
    h = ndi.median_filter(h, size=size)
    x = h[0:x.shape[0], 0:x.shape[1]]
    return x


def kernel_density_filter(f, r, w):
    """
    todo: Make this work. What I am trying to do is make a density plot of the data to define percentile
            regions which I can then crop to. So I need Freq, Range, and S for weighting. Probs run this
            after we despeckle.
    """
    from scipy import stats
    import matplotlib.pyplot as plt
    lat = r
    lon = f
    lat_min = np.min(lat)
    lat_max = np.max(lat)
    lon_min = np.min(lon)
    lon_max = np.max(lon)

    # Peform the kernel density estimate
    xx, yy = np.mgrid[lon_min:lon_max:100j, lat_min:lat_max:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([lon, lat])
    kernel = stats.gaussian_kde(values, weights=w)
    z = np.reshape(kernel(positions).T, xx.shape)

    fig, ax = plt.subplots()
    ax.imshow(np.rot90(z), cmap=plt.cm.gist_earth_r, extent=[lon_min, lon_max, lat_min, lat_max])
    ax.plot(lon, lat, 'k.', markersize=2)
    # ax.set_ylim([lat_min, lat_max])
    # ax.set_xlim([lon_min, lon_max])
    plt.show()

    return

