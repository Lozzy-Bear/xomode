# Pretty plot configuration.
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib
from matplotlib import rc, pyplot
rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 12
pyplot.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
pyplot.rc('axes', titlesize=BIGGER_SIZE)  # font size of the axes title
pyplot.rc('axes', labelsize=MEDIUM_SIZE)  # font size of the x and y labels
pyplot.rc('xtick', labelsize=SMALL_SIZE)  # font size of the tick labels
pyplot.rc('ytick', labelsize=SMALL_SIZE)  # font size of the tick labels
pyplot.rc('legend', fontsize=SMALL_SIZE)  # legend font size
pyplot.rc('figure', titlesize=BIGGER_SIZE)  # font size of the figure title



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
