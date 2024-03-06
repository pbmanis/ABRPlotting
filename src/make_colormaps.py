# build color map where each SPL is a color (cycles over 12 levels)
import numpy as np
import seaborn as sns
import matplotlib as mpl

def make_spl_freq_colormap():
    spl_bounds = np.linspace(0, 120, 25)  # 0 to 120 db inclusive, 5 db steps
    color_labels = list(set(spl_bounds))
    rgb_values = list(sns.color_palette("Set2", 25))
    # Map label to RGB
    color_map = dict(list(zip(color_labels, rgb_values)))
    freqs = np.arange(500, 48100, 100)
    spectral = mpl.colormaps['Spectral'].resampled(len(freqs))
    freq_colormap = {}
    for i, fr in enumerate(freqs):
        freq_colormap[int(fr)] = spectral(fr/1000)
    #     if i < 5:
    #         print(freq_colormap[fr])
            
    # exit()


    # freq_colormap = {
    #     1000: "r",
    #     2000: "g",
    #     3000: "lime",
    #     4000: "b",
    #     8000: "k",
    #     12000: "m",
    #     16000: "c",
    #     24000: "y",
    #     32000: "orange",
    #     48000: "brown",
    # }
    return color_map, freq_colormap

def make_subject_colormap(subjects):

    color_labels = list(set(subjects))
    rgb_values = list(sns.color_palette("Set2",len(color_labels)))
    # Map label to RGB
    color_map = dict(list(zip(color_labels, rgb_values)))
    return color_map

def make_group_colormap(groups):
    color_labels = list(set(groups))
    rgb_values = list(sns.color_palette("Set2", len(color_labels)))
    # Map label to RGB
    color_map = dict(list(zip(color_labels, rgb_values)))
    return color_map

