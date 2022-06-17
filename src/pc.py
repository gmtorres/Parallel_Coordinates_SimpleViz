# Inspired by https://stackoverflow.com/a/60401570

from functools import partial
import functools
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
from matplotlib.widgets import Slider, Button

from sklearn import datasets

iris = datasets.load_iris()

values = iris.data # dada values for visualization

ynames = iris.feature_names # change this to appropriate labells
ys = values
ymins = ys.min(axis=0)
ymaxs = ys.max(axis=0)
dys = ymaxs - ymins
ymins -= dys * 0.05  # add 5% padding below and above
ymaxs += dys * 0.05

aspiration = np.array(ymaxs)
host, zs = None, None
fig, host = plt.subplots(figsize=(10,4))

#ymaxs[1], ymins[1] = ymins[1], ymaxs[1]  # reverse axis 1 to have less crossings
def draw():
    global host, zs, ymins, ymaxs, dys
    ymins = ys.min(axis=0)
    ymaxs = ys.max(axis=0)
    dys = ymaxs - ymins
    ymins -= dys * 0.05  # add 5% padding below and above
    ymaxs += dys * 0.05

    dys = ymaxs - ymins

    # transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]


    axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))

    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    host.set_xticklabels(ynames, fontsize=14)
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()
    host.set_title('Parallel Coordinates Plot — Sectorizarion', fontsize=18, pad=12)
    
    draw_lines()
    

#host.legend(legend_handles, iris.target_names,
#            loc='lower center', bbox_to_anchor=(0.5, -0.18),
#            ncol=len(iris.target_names), fancybox=True, shadow=True)

def draw_lines():
    global host, zs
    #colors = plt.cm.Set2.colors
    #legend_handles = [None for _ in iris.target_names]
    cmap = plt.cm.get_cmap('cool_r')
    color_id = 0
    norm = matplotlib.colors.Normalize(vmin=ymins[color_id], vmax=ymaxs[color_id])
    for j in range(ys.shape[0]):
        if not np.less(ys[j],aspiration).all():
            continue
        # create bezier curves
        verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                        np.repeat(zs[j, :], 3)[1:-1]))
        codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
        path = Path(verts, codes)
        t = ys[j, color_id]
        patch = patches.PathPatch(path, facecolor='none', lw=1, alpha=0.5, edgecolor=cmap(norm(t)))
        #legend_handles[iris.target[j]] = patch
        host.add_patch(patch)
    
    aspiration_arr = np.array([aspiration])

    aspiration_arrs = np.zeros_like(aspiration_arr)
    aspiration_arrs[:, 0] = aspiration_arr[:, 0]
    aspiration_arrs[:, 1:] = (aspiration_arr[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                        np.repeat(aspiration_arrs[0, :], 3)[1:-1]))
    codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', lw=1, alpha=0.5, edgecolor='red')
    #legend_handles[iris.target[j]] = patch
    host.add_patch(patch)
    

def swap(id1, id2, event):
    #ax0, ax1 = axes[id1], axes[id2]
    ymins[id1], ymins[id2] = ymins[id2], ymins[id1]
    ymaxs[id1], ymaxs[id2] = ymaxs[id2], ymaxs[id1]
    ynames[id1], ynames[id2] = ynames[id2], ynames[id1]

    #zs[:, [id2, id1]] = zs[:, [id1, id2]]
    ys[:, [id2, id1]] = ys[:, [id1, id2]]

    print(aspiration)
    aspiration[id1], aspiration[id2] = aspiration[id2], aspiration[id1]
    """sliders[id1].valmax, sliders[id2].valmax = sliders[id2].valmax, sliders[id1].valmax
    sliders[id1].valmin, sliders[id2].valmin = sliders[id2].valmin, sliders[id1].valmin
    sliders[id1].val, sliders[id2].val = sliders[id2].val, sliders[id1].val
    sliders[id1].ax.set_xlim(sliders[id1].valmin,sliders[id1].valmax)
    sliders[id2].ax.set_xlim(sliders[id2].valmin,sliders[id2].valmax)
    sliders[id1].set_val(sliders[id1].val)
    sliders[id2].set_val(sliders[id2].val)
    sliders[id2].set_text("a")
    """

    #print(aspiration)

    #ax0.set_ylim(ymins[id1], ymaxs[id1])
    #ax1.set_ylim(ymins[id2], ymaxs[id2])
    #host.set_xticklabels(ynames, fontsize=14)
    #[p.remove() for p in reversed(host.patches)]
    plt.clf()
    draw()

    plt.show()
"""
bswaps = []
for i in range(ys.shape[1] - 2):
    axswap = plt.axes([0.48 + 0.27 * (i), 0.01, 0.05, 0.075])
    bswap = Button(axswap, '<>')
    bswap.on_clicked(functools.partial(swap, i + 1, i + 2))
    bswaps.append(bswap)
"""

def update_value(i, val):
    aspiration[i] = val
    #print(aspiration)
    [p.remove() for p in reversed(host.patches)]
    draw_lines()
    plt.draw()

ymins_t = ys.min(axis=0)
def generate_slider(i):
    axfreq = plt.axes([0.20, 0.30 - 0.08 * i, 0.65, 0.03])
    freq_slider = Slider(
        ax=axfreq,
        label=ynames[i],
        valmin=ymins_t[i],
        valmax=ymaxs[i],
        valinit=ymaxs[i],
    )
    freq_slider.on_changed(functools.partial(update_value,i))
    return freq_slider

sliders = []
for i in range(ys.shape[1]):
    freq_slider = generate_slider(i)
    sliders.append(freq_slider)

draw()

plt.tight_layout()
plt.subplots_adjust(bottom=0.35)

plt.show()


