"""
================================================================================

Triplot v1.0 - Wolfgang Enzi 2021

Inspired by python packages such as
getdist (https://getdist.readthedocs.io/en/latest/) and
corner.py (https://corner.readthedocs.io/en/latest/),
this code allows to create a triangle plot to visualize the relations among
different parameters in high dimensional samples. I hope that the simplicity
of this code makes it slightly more accessible than other software and
potentially slightly faster.

================================================================================
"""

import numpy as np
import matplotlib.pylab as plt
from scipy.misc import logsumexp as lse
from scipy.stats import gaussian_kde as gkde
from scipy import integrate as integ
import sys


# function to add mirrored points to for the given samples,
# to account for boundaries
def get_mirrorpoints(y, w, ranges):

    if y.shape[0] == 1:

        y_new = np.concatenate([2 * ranges[0][0] - y[0], y[0],
                                2 * ranges[0][1] - y[0]])[np.newaxis, :]
        w_new = np.concatenate([w, w, w])

    if y.shape[0] == 2:

        yl = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                yij = np.copy(y)
                if i != 0:
                    yij[0, :] = 2 * ranges[0][(i + 1) / 2] - yij[0, :]
                if j != 0:
                    yij[1, :] = 2 * ranges[1][(j + 1) / 2] - yij[1, :]
                yl += [yij]

        y_new = np.concatenate(yl, axis=1)
        w_new = np.concatenate([w, w, w, w, w, w, w, w, w])

    return y_new, w_new


# function to determine the probability mass of the ind*sigma region
def contour_percentile(ind):
    return (1 - np.exp(- (ind * 1.0) ** 2 / 2.0))


# function to determine percentiles
def get_percentiles(plot_data, x, percs):

    lvla = []
    for si in range(len(plot_data)):
        cumsum_si = np.cumsum(plot_data[si] / np.sum(plot_data[si]))
        lvla += [np.array([np.argmin((cumsum_si - percs[i]) ** 2)
                           for i in range(len(percs))])]

    percentiles_plot_1d = [[x[lvla[si][0]],
                            x[lvla[si][1]],
                            x[lvla[si][2]]] for si in range(len(plot_data))]

    return percentiles_plot_1d, lvla


# get the kernel density estimation (and reflect at boundaries if requested)
def get_kde_refl(y, w, ranges, refl=0):

    L = ranges[:, 1] - ranges[:, 0]
    y = np.atleast_2d(y).T
    ranges = np.atleast_2d(ranges)

    if refl == 1:
        # covariance estimation following silverman,
        # Monographs on Statistics and Applied Probability,
        # Chapman and Hall, London, 1986
        neff = np.sum(w) ** 2 / (np.sum(w * w))
        covf = (np.power(neff * (y.shape[0] + 2) / 4.0,
                         -1.0 / (y.shape[0] + 4)))
        covy = np.cov(y, aweights=w)
        y, w = get_mirrorpoints(y, w, ranges)

    q0 = gkde(y, weights=w / np.sum(w), bw_method="silverman")

    if refl == 1:
        q0.covariance = covy * np.power(covf, 2)
        q0.inv_cov = np.linalg.inv(np.atleast_2d(q0.covariance))

    if refl == 0 or refl == 1:
        def log_pdens(m):
            return np.log(q0(m) + 1e-30)

    return q0, q0.covariance, log_pdens


# function to create a colormap with increasing alpha for a given color
def get_alpha_colormap(auxc):

    import matplotlib.colors as colors
    u = colors.hex2color(auxc)
    from matplotlib.colors import LinearSegmentedColormap
    cdict = {'red':   [[0.0, u[0], u[0]],
                       [1.0, u[0], u[0]]],
             'green':  [[0.0, u[1], u[1]],
                        [1.0, u[1], u[1]]],
             'blue':   [[0.0, u[2], u[2]],
                        [1.0, u[2], u[2]]],
             'alpha':   [[0.0, 0.0, 0.0],
                         [1e-3, 0.0, 0.0],
                         [1.0, 0.6, 0.6]]}
    newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)

    return newcmp


# calculate the 2d array that is plotted
def get_2d_array(s, i, j, ranges, refl, N):

    fits2d = [get_kde_refl(np.take(s[si][0], [i, j], axis=1), s[si][1],
                           ranges, refl=refl) for si in range(len(s))]
    x = np.linspace(ranges[0, 0], ranges[0, 1], N)
    y = np.linspace(ranges[1, 0], ranges[1, 1], N)
    xx = np.outer(np.ones(y.shape), x)
    yy = np.outer(y, np.ones(x.shape))
    log_plot_data = []
    for si in range(len(s)):
        xy = np.array([xx.flatten(), yy.flatten()])
        log_plot_data_si = fits2d[si][2](xy).reshape(xx.shape)
        log_plot_data += [log_plot_data_si]

    return fits2d, x, y, log_plot_data, xx, yy


# plot the 2d array
def plot_2d(s, i, j, rangesx, N, refl, prangesx, nsig=3):

    plt.ticklabel_format(axis="both", style="plain", useOffset=False)
    ranges_ij = np.take(rangesx, [i, j], axis=0)
    pranges_ij = np.take(prangesx, [i, j], axis=0)

    ccycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fits2d, x, y, log_plot_data, xx, yy = get_2d_array(s, i, j, ranges_ij,
                                                       refl, N)

    plot_list = []
    for si in range(len(s)):

        log_plot_data_si = np.copy(log_plot_data[si])
        log_plot_data_si_sort = np.sort(log_plot_data_si.flatten())[::-1]
        plot_data_si_sort_cumsum = np.cumsum(np.exp(log_plot_data_si_sort
                                             - lse(log_plot_data_si_sort)))

        lvla_si = []
        for i in range(nsig):

            ind_lvl = np.argmin(np.power(plot_data_si_sort_cumsum
                                         - contour_percentile(i + 1), 2.0))
            lvla_si += [log_plot_data_si_sort[ind_lvl]]

        plot1 = plt.imshow(np.exp(log_plot_data_si),
                           cmap=get_alpha_colormap(ccycle[si]),
                           extent=(np.min(x), np.max(x), np.min(y),
                           np.max(y)), origin="lower")

        extent_si = (np.min(x), np.max(x), np.min(y), np.max(y))
        plot2 = plt.contour(x, y, log_plot_data_si, levels=np.sort(lvla_si),
                            colors=ccycle[si],
                            extent=extent_si,
                            origin="lower", linestyles="solid")

        plt.xlim(pranges_ij[0, 0], pranges_ij[0, 1])
        plt.ylim(pranges_ij[1, 0], pranges_ij[1, 1])

        plot_list += [[plot1, plot2]]


# calculate the 1d array that is plotted
def get_1d_array(s, i, ranges, refl, N):

    fits1d = [get_kde_refl(np.take(s[si][0], [i], axis=1),
                           s[si][1], ranges, refl) for si in range(len(s))]
    x = np.linspace(ranges[0, 0], ranges[0, 1], N)
    plot_aux = [fits1d[si][0](x) for si in range(len(s))]
    plot_data = []
    for si in range(len(s)):
        plot_data_si = plot_aux[si] / np.trapz(plot_aux[si], x)
        plot_data += [plot_data_si]

    return fits1d, x, plot_data


# plot the 1d array
def plot_1d(s, i, rangesx, N, refl, prangesx, slabels=[]):

    plt.ticklabel_format(axis='y', style='sci',
                         scilimits=(0, 2), useOffset=False)
    plt.ticklabel_format(axis='x', style='plain', useOffset=False)

    ranges_i = np.atleast_2d(np.take(rangesx, i, axis=0))
    pranges_i = np.atleast_2d(np.take(prangesx, i, axis=0))

    ccycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fits2d, x, plot_data = get_1d_array(s, i, ranges_i, refl, N)

    percs_list = [0.5 - 0.683 / 2.0, 0.5, 0.5 + 0.683 / 2.0]

    percentiles_plot_1d, lvla = get_percentiles(plot_data, x, percs_list)

    plot_list = [[
            plt.fill_between(x, plot_data[si], color=ccycle[si], alpha=0.15),
            plt.plot(x, plot_data[si], color=ccycle[si], label=slabels[si]),
            plt.plot(np.array([x[lvla[si][0]], x[lvla[si][0]]]),
                     np.array([0, plot_data[si][lvla[si][0]]]),
                     color=ccycle[si], alpha=0.25),
            plt.plot(np.array([x[lvla[si][1]], x[lvla[si][1]]]),
                     np.array([0, plot_data[si][lvla[si][1]]]),
                     color=ccycle[si], ls=":", alpha=0.25),
            plt.plot(np.array([x[lvla[si][2]], x[lvla[si][2]]]),
                     np.array([0, plot_data[si][lvla[si][2]]]),
                     color=ccycle[si], alpha=0.25)] for si in range(len(s))]

    plt.xlim(pranges_i[0, 0], pranges_i[0, 1])

    return percentiles_plot_1d


# main function of this code, defines the routine to create the plots
def triangl_plot(s, i, rangesx, N, prangesx=[], labels=[], slabels=[], refl=0):

    fig = plt.figure()

    if (len(prangesx) == 0):
        prangesx = np.copy(rangesx)

    for u in range(rangesx.shape[0]):
        prangesx[u] = np.clip(prangesx[u], rangesx[u, 0], rangesx[u, 1])

    ni = len(np.atleast_1d(i))

    percs = []
    axes = []

    """
    Loop over the indices requested to be plotted
    """
    for u in range(ni):
        for v in range(u, ni):

            nii = i[u]
            nij = i[v]

            ax = plt.subplot2grid((ni, ni), (v, u))
            axes += [ax]

            if (u == v):  # 1D plots for diagonals
                print("plot 1d : %d " % nii + "...")
                persc_nii = plot_1d(s, nii, rangesx, N,
                                    refl, prangesx, slabels)
                percs += [persc_nii]
                plt.tick_params(axis='y', labelleft=False, labelright=True)
                plt.gca().yaxis.set_ticks_position("right")
                if v == 0:
                    plt.legend(bbox_to_anchor=(1.0*ni+0.1, 1.0))
                print("done")

            else:  # 2D plots for off diagonals
                print("plot 2d : %d, %d " % (nij, nii) + "...")
                plot_2d(s, nii, nij, rangesx, N, refl, prangesx)
                print("done")

            """
            Remove y ticks for plots that are not in the most left column
            and add labels to only the most left column
            """
            if ((u != 0) and (u != v)):
                plt.yticks([])

            else:
                if (u != v):
                    if (len(labels) > 0):
                        plt.ylabel(labels[nij])

            if (u == ni - 1) and (v == ni - 1):
                plt.xlabel(labels[nii])

            """
            Remove x ticks for plots that are not in the bottom row
            and add labels to only the bottom row
            """
            if (v != ni - 1):
                plt.xticks([])

            else:
                if (u != v):
                    if (len(labels) > 0):
                        plt.xlabel(labels[nii])

            plt.xticks(rotation=45)
            ax.set_aspect('auto')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15, wspace=0.15)

    print("\n")

    fig.align_ylabels(axes)
    fig.align_xlabels(axes)

    for si in range(len(s)):

        print("Sample %d" % si)
        string_args = ((0.5 - 0.683 / 2.0) * 100,
                       0.5 * 100,
                       (0.5 + 0.683 / 2.0) * 100)
        print("P\t%.2f%%\t\t\t%.2f%%\t\t\t%.2f%%" % string_args)

        for u in range(ni):
            nii = i[u]
            string_args = (nii, percs[u][si][0],
                           percs[u][si][1], percs[u][si][2])
            print("%d\t%.2e\t\t%.2e\t\t%.2e" % string_args)

        print("\n")
