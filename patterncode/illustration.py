import random

import numpy as np
from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from skimage.exposure import rescale_intensity

from patterncode.config import *

GAP_LIMS = 10, 15


def find_pattern_indices(sequence, pattern):
    indices = []
    for i in range(len(sequence) - len(pattern) + 1):
        if sequence[i:i + len(pattern)] == pattern:
            indices.append(i)
    return indices


def draw_dna_sequence(ax, sequence, pattern=CTTAAG, offset=0.5):
    pattern_indices = find_pattern_indices(sequence, pattern)

    for i, base in enumerate(sequence):
        for idx in pattern_indices:
            if idx == i:
                color = 'black'
                fontweight = 'bold'
                break
            elif idx <= i < idx + len(pattern):
                color = 'black'
                fontweight = 'bold'
                break
        else:
            color = 'black'
            fontweight = 'normal'
        x = i * 5
        if GAP_LIMS[0] <= i <= GAP_LIMS[1]:
            continue
        ax.text(x, offset, base, color=color, fontsize=5, ha='center', va='center', fontweight=fontweight)



def plot_double_helix(ax, sequence, length=500, amplitude=10, period=10 * 2 * np.pi, linewidth=1):
    t = np.linspace(0, length, num=1000)
    y1 = amplitude * np.sin(period * t / length)
    y2 = -y1
    scale = 5
    valid = (t / scale) >= GAP_LIMS[1]
    ax.plot(t[valid], y1[valid], color="blue", linewidth=linewidth)
    ax.plot(t[valid], y2[valid], color="red", linewidth=linewidth)

    valid = (t / scale) <= GAP_LIMS[0]
    ax.plot(t[valid], y1[valid], color="blue", linewidth=linewidth)
    ax.plot(t[valid], y2[valid], color="red", linewidth=linewidth)

    dots_x = np.linspace(GAP_LIMS[0] * scale, GAP_LIMS[1] * scale, num=5)[1:-1]
    ax.plot(dots_x, np.zeros_like(dots_x), '.', color="k", markersize=1)

    for i in range(length):
        if i % 5 == 0:
            connector_x = np.full(50, i)
            connector_y = np.linspace(y1[i * 2], y2[i * 2], num=50)
            if GAP_LIMS[0] <= i / scale <= GAP_LIMS[1]:
                continue
            ax.plot(connector_x, connector_y, color="green", linewidth=.3)

    # Draw the DNA sequence on top of the double helix
    draw_dna_sequence(ax, sequence, offset=amplitude * 1.5)

    ax.set_aspect("equal")
    ax.set_xlim(0, length)
    ax.axis("off")


def plot_dna_illustration():
    random.seed(0)
    pattern = 'CTTAAG'
    sequence_length = 50
    k = 5
    dna_sequence = ''.join([random.choice(['A', 'C', 'G', 'T']) for _ in range(sequence_length - len(pattern) * k)])
    insert_positions = random.sample(range(len(dna_sequence)), k)

    for pos in sorted(insert_positions, reverse=True):
        dna_sequence = dna_sequence[:pos] + pattern + dna_sequence[pos:]
    ax = plt.gca()
    plot_double_helix(ax, dna_sequence)

    plt.xlim(0, 250)
    scalebar_kw = dict(color="k", box_alpha=0, scale_loc="top", location="upper right",
                       width_fraction=.05,
                       length_fraction=.1,
                       sep=0,
                       font_properties=dict(weight="bold", size=6)
                       )

    scalebar_scale = .5e-10
    plt.gca().add_artist(ScaleBar(scalebar_scale, **scalebar_kw))


def plot_image_illustration(image, locs, ref):
    fig, axs = plt.subplots(
        5, 1,
        figsize=(FIG_SIZE, 1.5),
        height_ratios=[2, 1, 1, 1, .1]
    )
    label_kw = dict(ha='right', va='center', fontsize=6)
    plt.xlim(0, image.shape[-1])
    plt.sca(axs[0])

    plt.text(-.01, .5, 'a)', transform=plt.gca().transAxes, **label_kw, )
    plot_dna_illustration()

    plt.sca(axs[1])

    plt.axis('off')
    axs = axs[1:]

    axs[0].sharex(axs[1])
    axs[2].sharex(axs[1])

    plt.sca(axs[1])

    plt.text(-.01, .5, 'b)', transform=plt.gca().transAxes, **label_kw, )
    plt.axis('off')
    width = 11
    crop = image[
           image.shape[0] // 2 - width // 2 + 1:
           image.shape[0] // 2 + width // 2 + 2
           ]
    crop = rescale_intensity(crop,
                             in_range=(0, 1000),
                             out_range=(0, 1)
                             )
    plt.imshow(crop, cmap='gray', interpolation='none', aspect='auto')
    scalebar_kw = dict(color="ghostwhite", box_alpha=0, scale_loc="top", location="upper right", width_fraction=.05,
                       length_fraction=.1, sep=0,
                       font_properties=dict(weight="normal", size=6))

    scalebar_scale = 130e-9
    plt.gca().add_artist(ScaleBar(scalebar_scale, **scalebar_kw))

    plt.sca(axs[0])

    plt.axis('off')
    fontsize = 5

    def xy_label(label):
        plt.text(0, .5, label, fontsize=fontsize, va='center', ha='right', color='k', transform=plt.gca().transAxes)

    xy_label('y: ')

    s = .25

    pixels_per_bp = .985 * (locs[-1] - locs[0]) / (ref[-1] - ref[0])
    # locs = locs - offset
    locs = locs[locs < image.shape[-1]]
    locs = locs[locs >= 0]

    bin_size_pixels = DEFAULT_BIN_SIZE * pixels_per_bp

    ref = pixels_per_bp * (ref - ref[0]) + locs[0]
    ref = ref[ref < image.shape[-1]]

    kw = dict(linewidth=0.5, alpha=.8)
    for loc in locs:
        plt.axvline(loc, 0, s, color='red', **kw)
    M = int(image.shape[-1] / bin_size_pixels)
    xv = np.bincount((ref / bin_size_pixels).astype(int), minlength=M)
    yv = np.bincount((locs / bin_size_pixels).astype(int), minlength=M)
    bin_edges = np.arange(len(xv) + 1) * bin_size_pixels

    def annotate_bincounts(xv, *ylim):
        for i, edge in enumerate(bin_edges):
            plt.axvline(edge, *ylim, ls=':', color='k', linewidth=0.3)
            if i < len(bin_edges) - 1:
                plt.annotate(xv[i], (edge + bin_size_pixels / 2, .5), ha='center', va='center', color='k',
                             fontsize=fontsize)

    bs = .75
    annotate_bincounts(yv, 0, bs)

    plt.sca(axs[2])

    xy_label('x: ')
    plt.axis('off')

    for r in ref:
        plt.axvline(r, 1, 1 - s, color='g', **kw)

    annotate_bincounts(xv, 1 - bs, 1)

    xlim = plt.xlim(15, None)

    plt.sca(axs[-1])
    # plt.axis('off')

    delta = xlim[1] - xlim[0]
    plt.xlim(0, delta / pixels_per_bp / 1000)

    plt.gca().axes.get_yaxis().set_visible(False)
    # axes box
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('Position in DNA fragment (kilobase)')
    plt.subplots_adjust(hspace=0)
