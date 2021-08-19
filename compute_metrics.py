import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import itertools
from ae_package import pca
# from ae_package import autoencoder2_0
from ae_package import useful_functions
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy import stats
from scipy.spatial import distance
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from sklearn.manifold import TSNE
from ae_package.pca import PrincipalComponentAnalysis
from matplotlib.ticker import FormatStrFormatter
from sklearn.cross_decomposition import CCA


# # double exponential with constant
# def func(x, a, b, c, d, e):
#     return a * np.exp(-b * x) + c * np.exp(-d * x) + e


# # double exponential
# def func(x, a, b, c, d):
#     return a * np.exp(-b * x) + c * np.exp(-d * x)

# single exponential with constant
def func(x, a, b, c):
    return a * np.exp(-b * x) + c


def append_nan(metric_list, tot_trial):
    """
    convert list to numpy array and add eventual NaNs
    :param metric_list: list to be converted in np array. Nans are added if necessary
    :param tot_trial: total number of trails
    :return:
    """
    metric_list = np.array(metric_list)
    if metric_list.shape[0] != tot_trial:
        metric_list = np.append(metric_list, np.zeros(tot_trial - len(metric_list)) + np.nan)
    metric_list = np.reshape(metric_list, (metric_list.shape[0],))
    metric_list = metric_list.astype(float)

    return metric_list


def compute_ee(df_reach, names, trial_list, idx, tgt, tgt_list):
    """
    This function computes the error between the final position of the cursor and the current target during blind trials
    :param df_reach: dataframe with reaching data
    :param names: list with subjects' names
    :param trial_list: list with number of trials per each subject
    :param idx: indexslice for handling dataframes
    :param tgt: array containing target positions during training
    :param tgt: array containing order of targets during the session (same for all participants)
    :return:
    """

    df_reach = df_reach.reset_index()
    df_reach = df_reach.set_index(["Subject", "trial"])

    ee_list = [[] for i in range(len(names))]

    # f, ax = plt.subplots(1, len(names))
    # f.suptitle('Endpoint Error trend during Training for ' + typ + ' group', fontsize=18)

    # compute endpoint error for training  blocks
    for sub in range(len(names)):
        ee = []

        for trial in range(1, trial_list[sub] + 1):
            # exclude everything but blind trials
            if 1 in df_reach.loc[idx[sub + 1, trial], ['isBlind']].values:
                curs_end_tmp = df_reach.loc[idx[sub + 1, trial], ['cursor_x', 'cursor_y', 'count_mouse']]
                curs_end = curs_end_tmp.loc[curs_end_tmp['count_mouse'] == 100].values[0, 0:2]
                tgt_curr = tgt_list[trial-1]
                ee.append(distance.euclidean(curs_end, tgt[:, tgt_curr]) * 0.0265)

        # prepare list of lists for all participants
        ee = np.array(ee)
        if ee.shape[0] != 32:
            ee = np.append(ee, np.zeros(32 - len(ee)) + np.nan)
        ee = np.reshape(ee, (ee.shape[0],))
        ee = ee.astype(float)
        ee_list[sub] = ee

    #     # exponential fit on ee
    #     x = np.arange(ee.shape[0])
    #     y = ee
    #     # id = np.isfinite(x) & np.isfinite(y)
    #     # popt, pcov = curve_fit(func, x[id], y[id], p0=[max(y), 1 / 5, min(y)], maxfev=2000)
    #     # tau.append(1 / popt[1])
    #     # est = func(x, popt[0], popt[1], popt[2])
    #     # res = y - est
    #     # ss_res = np.sum(res ** 2)
    #     # ss_tot = np.sum((y - np.mean(y)) ** 2)
    #     # r_sq.append(1 - (ss_res / ss_tot))
    #
    #     # plot everything
    #     # ax[sub].plot(loss * (max(ee)/max(loss)), color='orange', label='loss J')
    #     # ax[sub].plot(est, color='green', label='fit')
    #     ax[sub].scatter(x, y, color='black', label='Endpoint error', s=5 * 2 ** 1e-4)
    #     ax[sub].set_xlabel('trials', fontsize=14)
    #     if sub == len(names) - 1:
    #         ax[sub].legend(frameon=False, fontsize=12, loc='upper right')
    #     elif sub == 0:
    #         ax[sub].set_ylabel('[pxl]', fontsize=12)
    #     ax[sub].set_title('Subject ' + str(sub + 1), fontsize=16)
    #     ax[sub].spines['right'].set_color('none')
    #     ax[sub].spines['top'].set_color('none')
    #     ax[sub].set_yticks([0, 4, 8, 12, 16, 20])
    #     ax[sub].set_ylim([0, 20])
    #     ax[sub].set_xlim([-1, 33])
    #     # set fontsize for xticks and yticks
    #     for tick in ax[sub].xaxis.get_major_ticks():
    #         tick.label.set_fontsize(12)
    #     for tick in ax[sub].yaxis.get_major_ticks():
    #         tick.label.set_fontsize(12)
    #
    # plt.show()

    return ee_list


def compute_ee_repetition(names, ee):
    """
        plot loss, ee and its fit. Mean across repetitions
        :param names: list with subjects' names
        :param ee: list of list containing ee during training per each subject
        :return: plot loss against ee and its fit, corr and values between loss and ee
    """

    # maria(sub 10 coad) was not paying attention during 16th trial (15 pyhton). Exclude it
    ee[9][15] = float('NaN')

    ci = 1.96
    ee_list = [[] for i in range(len(names))]

    # f, ax = plt.subplots(1, len(names))
    # f.suptitle('Endpoint error during blind trials for ' + typ + ' group', fontsize=18)

    for sub in range(len(names)):
        ee_mu = []
        tmp_mu = 0
        tot_rep = int(len(ee[sub]) / 4)

        # averaging reaching time per repetition
        for i in range(tot_rep):
            # first delete outliers, then append
            curr = ee[sub][tmp_mu: tmp_mu + 4]
            mu = np.mean(curr)
            std = np.std(curr)
            for index, item in enumerate(curr):
                if not mu - ci*std <= item < mu + ci*std:
                    curr[index] = float('NaN')
            ee_mu.append(np.nanmean(curr))
            tmp_mu += 4

        # prepare movement time array
        ee_mu = np.array(ee_mu)
        ee_mu = np.reshape(ee_mu, (tot_rep,))
        ee_mu = ee_mu.astype(float)
        ee_list[sub] = ee_mu

    #     x = np.arange(tot_rep)
    #     y = ee_mu
    #
    #     # plot everything
    #     # ax[sub].plot(loss * (max(ee)/max(loss)), color='orange', label='loss J')
    #     ax[sub].scatter(x, y, color='black', label='Endpoint error', s=5 * 2 ** 1e-4)
    #     ax[sub].set_xlabel('repetitions', fontsize=14)
    #     if sub == len(names)-1:
    #         ax[sub].legend(frameon=False, fontsize=12)
    #     if sub == 0:
    #         ax[sub].set_ylabel('[cm]', fontsize=14)
    #     ax[sub].set_title('Subject ' + str(sub + 1), fontsize=16)
    #     # get rid of the frame
    #     ax[sub].spines['right'].set_color('none')
    #     ax[sub].spines['top'].set_color('none')
    #     ax[sub].set_yticks([0, 4, 8, 12, 16, 20])
    #     ax[sub].set_ylim([0, 20])
    #     ax[sub].set_xlim([-5, tot_rep+5])
    #     # set fontsize for xticks and yticks
    #     for tick in ax[sub].xaxis.get_major_ticks():
    #         tick.label.set_fontsize(12)
    #     for tick in ax[sub].yaxis.get_major_ticks():
    #         tick.label.set_fontsize(12)
    #
    # # plt.show()

    return ee_list


def compute_cca(pdf, names, df_reach, df_reach_static, df_w1, df_b1, df_w2, df_b2, idx):
    """
    Function that computes CCA on neuron activation vectors (2nd layer, 1st still possible)
    :return:
    """
    col_imu = ['imu00', 'imu01', 'imu02', 'imu03', 'imu10', 'imu11', 'imu12', 'imu13']
    col_w2 = ['w1', 'w2']
    col_b1 = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8']
    col_b2 = ['b1', 'b2']

    # first, create a dataset based on MVG with mean and cov from a static experiment
    data = df_reach_static.loc[idx[1.0], col_imu]
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)
    data_sim = np.random.multivariate_normal(mean, cov, 3000)

    # data_sim = [[] for i in range(len(names))]
    # for sub in range(len(names)):
    #     data = df_reach.loc[idx[sub+1], col_imu]
    #     mean = np.mean(data, axis=0)
    #     cov = np.cov(data.T)
    #     data_sim[sub] = np.random.multivariate_normal(mean, cov, 100)

    # then compute neuron activation vectors (1st, 2nd layers) for each batch
    activations_1st = [[] for i in range(len(names))]
    activations_2nd = [[] for i in range(len(names))]

    # obtain activations with sim data for each subject and batch (indexes of df_w1). id[0] is subj and id[1] is batch
    for id, df_select in df_w1.groupby(df_w1.index):
        w1 = df_select.values
        b1 = df_b1.loc[idx[id[0], id[1]], col_b1].values
        w2 = df_w2.loc[idx[id[0], id[1]], col_w2].values
        b2 = df_b2.loc[idx[id[0], id[1]], col_b2].values
        # h1 = np.tanh(np.dot(data_sim[int(id[0] - 1)], w1) + b1)
        h1 = np.tanh(np.dot(data_sim, w1) + b1)
        cu = np.dot(h1, w2) + b2
        activations_1st[int(id[0] - 1)].append(np.array(h1))
        activations_2nd[int(id[0] - 1)].append(np.array(cu))

    # get number of batches of python thread
    last = df_w1.reset_index().groupby('Subject', as_index=False).last()
    batch_list_py = last['batch'].values.tolist()
    batch_list_py = [int(i) for i in batch_list_py]
    svcca_end_conf = [[] for i in range(len(names))]  # conf matrix that contains svcca values between start and end
    svcca_2min = [[] for i in range(len(names))]  # list of lists that contains svcca every 2 min for each subject
    svcca_2min_base = [[] for i in
                       range(len(names))]  # list of lists containing svcca every 2 min for each subject wrt start

    # initilize CCA object from sklearn
    comp_cca = 2
    cca = CCA(n_components=comp_cca, max_iter=1000)
    n_batch = 1  # compute CCA every n_batch*2s

    # compute CCA and similarity index (SVCCA) every n_batch sec wrt baseline each subject
    for sub in range(len(names)):
        # compute cca every 10s (or equivalently n batches)
        for batch in range(n_batch, batch_list_py[sub], n_batch):
            X = activations_2nd[sub][0] - np.mean(activations_2nd[sub][0], axis=0)
            Y = activations_2nd[sub][batch] - np.mean(activations_2nd[sub][batch], axis=0)
            cca.fit(X, Y)
            # # X_s and Y_s ane n_samples x n_comp matrices. Each column of U and V is a different order of correlation
            # X_s, Y_s = cca.transform(X, Y)
            # corr = []
            # for i in range(comp_cca):
            #     corr.append(np.corrcoef(X_s[:, i], Y_s[:, i])[0, 1])
            # corr.append(np.corrcoef(X_s.T, Y_s.T).diagonal(offset=comp_cca))  # gives same results
            corr = np.diag(np.corrcoef(cca.x_scores_, cca.y_scores_, rowvar=False)[:comp_cca, comp_cca:])
            svcca_2min_base[sub].append(np.mean(corr))  # using dsvcca
            # vaf = []
            # for i in range(comp_cca):
            #     vaf.append(useful_functions.compute_vaf(cca.x_scores_[:, i], X[:, i]))
            # vaf = [float(i) / sum(vaf) for i in vaf]
            # svcca_2min_base[sub].append(1 - np.sum(vaf*corr))  # using dpwcca
        fig_handle = plt.figure()
        plt.plot(np.arange(n_batch * 2, n_batch * 2 * len(svcca_2min_base[sub]) + 1, n_batch * 2), svcca_2min_base[sub])
        plt.title('Similarity every 2s wrt base for subject ' + str(sub + 1))
        plt.xlabel('Time [s]')
        plt.ylabel('dSVCCA')
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        pdf.savefig(fig_handle, dpi=300, papertype='letter', orientation='landscape', pad_inches=0)
        plt.close()

    # # there seems to be just a fckin offset....CCA is always one
    # for chn in range(comp_cca):
    #     plt.figure()
    #     plt.plot(activations_2nd[sub][0][:, chn], color='red')
    #     plt.plot(activations_2nd[sub][batch_list_py[sub]-1][:, chn], color='green')

    # compute CCA and similarity index (SVCCA) every 2 min each subject
    for sub in range(len(names)):
        # compute cca every 10s (or equivalently n_batches)
        for batch in range(n_batch, batch_list_py[sub], n_batch):
            X = activations_2nd[sub][batch - n_batch] - np.mean(activations_2nd[sub][batch - n_batch], axis=0)
            Y = activations_2nd[sub][batch] - np.mean(activations_2nd[sub][batch], axis=0)
            cca.fit(X, Y)
            corr = np.diag(np.corrcoef(cca.x_scores_, cca.y_scores_, rowvar=False)[:comp_cca, comp_cca:])
            svcca_2min[sub].append(np.mean(corr))
            # vaf = []
            # for i in range(comp_cca):
            #     vaf.append(useful_functions.compute_vaf(cca.x_scores_[:, i], X[:, i]))
            # vaf = [float(i) / sum(vaf) for i in vaf]
            # svcca_2min[sub].append(1 - np.sum(vaf * corr))  # using dpwcca
        fig_handle = plt.figure()
        plt.plot(np.arange(n_batch * 2, n_batch * 2 * len(svcca_2min[sub]) + 1, n_batch * 2), svcca_2min[sub])
        plt.title('Similarity every 2s for subject ' + str(sub + 1))
        plt.xlabel('Time [s]')
        plt.ylabel('dSVCCA')
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        pdf.savefig(fig_handle, dpi=300, papertype='letter', orientation='landscape', pad_inches=0)
        plt.close()

    # compute CCA and similarity index (SVCCA) between subjects at the end
    for subx in range(len(names)):
        for suby in range(len(names)):
            X = activations_2nd[subx][batch_list_py[subx] - 1] - np.mean(activations_2nd[subx][batch_list_py[subx] - 1],
                                                                         axis=0)
            Y = activations_2nd[suby][batch_list_py[suby] - 1] - np.mean(activations_2nd[suby][batch_list_py[suby] - 1],
                                                                         axis=0)
            cca.fit(X, Y)
            corr = np.diag(np.corrcoef(cca.x_scores_, cca.y_scores_, rowvar=False)[:comp_cca, comp_cca:])
            svcca_end_conf[subx].append(np.mean(corr))
            # vaf = []
            # for i in range(comp_cca):
            #     vaf.append(useful_functions.compute_vaf(cca.x_scores_[:, i], X[:, i]))
            # vaf = [float(i) / sum(vaf) for i in vaf]
            # svcca_end_conf[subx].append(1 - np.sum(vaf * corr))  # using dpwcca
    svcca_end_conf = np.array(svcca_end_conf)

    # order matrix by sum(similarity)
    ord = np.flip(np.argsort(np.sum(svcca_end_conf, axis=0)))
    old = svcca_end_conf.copy()
    new = np.zeros((10, 10))
    for row in range(10):
        for col in range(10):
            new[row, col] = old[ord[row], ord[col]]
    svcca_end_conf = new.copy()

    fig_handle = plt.figure()
    plt.imshow(svcca_end_conf)
    plt.title('Similarity of final activation vectors between subjects')
    plt.xlabel('Subject')
    plt.ylabel('Subject')
    plt.colorbar()
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    # pdf.savefig(fig_handle, dpi=300, papertype='letter', orientation='landscape', pad_inches=0)
    # plt.close()

    plt.show()

    return svcca_2min_base, svcca_2min, svcca_end_conf


def compute_corr_cca_vaf(names, vaf, svcca_2min):
    f, ax = plt.subplots(2, 5)
    f2, ax2 = plt.subplots(2, 5)
    # f, ax = plt.subplots(1, len(names))
    f.suptitle('Loss vs similarity trend', fontsize=14)
    f2.suptitle('Loss vs similarity trend', fontsize=14)
    axx = 0  # index to iterate rows of subplot
    axy = 0  # index to iterate cols of subplot

    corr_cca_vaf = []
    start_corr = 5  # start correlation only after x batches

    for sub in range(len(names)):
        # handle index of subplot
        if sub == 5:
            axx = 1
            axy = 0

        x = np.array(svcca_2min[sub][start_corr-1:])
        y = np.array(vaf[sub][start_corr:])

        flt_order = 3
        flt_fc = 1
        flt_fs = 50
        # x = useful_functions.filter(flt_order, 24.999, flt_fs, 'lowpass', x)
        # y = useful_functions.filter(flt_order, flt_fc, flt_fs, 'lowpass', y)

        # corr_tmp = pearsonr(x, y)
        # corr_cca_vaf.append(corr_tmp[0])

        # plot everything
        # ax[axx, axy].plot(np.array(x) * (max(y) / max(x)), color='green', label='similarity')
        # ax[axx, axy].plot(np.log10(x) * 1000, color='green', label='similarity')
        # ax[axx, axy].semilogy(x, color='green', label='similarity')
        ax[axx, axy].plot(x, color='green', label='similarity')


        # ax[axx, axy].plot(y, color='orange', label='VAF')
        # ax[axx, axy].plot(np.array(y) * (max(x) / max(y)), color='orange', label='VAF')
        if sub == 7:
            ax[axx, axy].set_xlabel('time [s]', fontsize=10)
        if sub == len(names) - 1:
            ax[axx, axy].legend(frameon=False, fontsize=8, loc='upper right')
        elif sub == 0:
            ax[axx, axy].set_ylabel("", fontsize=12)
        ax[axx, axy].set_title('Subject ' + str(sub + 1), fontsize=10)
        ax[axx, axy].spines['right'].set_color('none')
        ax[axx, axy].spines['top'].set_color('none')
        ax[axx, axy].set_ylim(0.9999, 1)
        # ax[axx, axy].set_xlim(0, len(y))
        # # ax[axx, axy].set_ylim(0, max(y) + 0.1*max(y))
        # # set fontsize for xticks and yticks
        # for tick in ax[axx, axy].xaxis.get_major_ticks():
        #     tick.label.set_fontsize(8)
        # ax[axx, axy].set_yticklabels([])
        # ax[axx, axy].text(0.6, 0.9, r'$R^2$: ' + str(np.ceil(corr_cca_vaf[sub] * 100) / 100),
        #                   fontsize=8, transform=ax[axx, axy].transAxes)

        # plot everything
        # ax[axx, axy].plot(np.array(x) * (max(y) / max(x)), color='green', label='similarity')
        # ax[axx, axy].plot(x, color='green', label='similarity')
        ax2[axx, axy].plot(y, color='orange', label='VAF')
        # ax2[axx, axy].plot(np.array(y) * (max(x) / max(y)), color='orange', label='VAF')
        if sub == 7:
            ax2[axx, axy].set_xlabel('time [s]', fontsize=10)
        if sub == len(names) - 1:
            ax2[axx, axy].legend(frameon=False, fontsize=8, loc='upper right')
        elif sub == 0:
            ax2[axx, axy].set_ylabel("", fontsize=12)
        ax2[axx, axy].set_title('Subject ' + str(sub + 1), fontsize=10)
        ax2[axx, axy].spines['right'].set_color('none')
        ax2[axx, axy].spines['top'].set_color('none')
        ax2[axx, axy].set_xlim(0, len(y))
        # set fontsize for xticks and yticks
        for tick in ax2[axx, axy].xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        ax2[axx, axy].set_ylim(0, 90)
        # ax2[axx, axy].set_yticklabels([])
        # ax2[axx, axy].text(0.6, 0.9, r'$R^2$: ' + str(np.ceil(corr_cca_vaf[sub] * 100) / 100),
        #                   fontsize=8, transform=ax2[axx, axy].transAxes)

        axy += 1

    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    # pdf.savefig(f, dpi=300, papertype='letter', orientation='landscape', pad_inches=0)
    # plt.close()

    plt.show()

    return corr_cca_vaf


def compute_loss_static(names, sess, df, idx, mainPath):
    """
    Function that computes loss offline for static group. Takes a while to run bc it trains AE every 2s of data
    """
    # remove unnecessary rows from df_reach. you don't need out-center, blind and test trials
    df = df.reset_index()
    df = df[df['comeback'] == 0]
    df = df[df['isBlind'] == 0]
    df = df[~df['block'].isin([1, 6, 11])]

    # need index for cycling through subjects
    df = df.set_index(["Subject"])

    # import pca and autoencoder from ae_package
    AE = autoencoder2_0.Autoencoder(10, 0.0001, 2, struc="non_linear", nh1=8)
    col_imu = ['imu00', 'imu01', 'imu02', 'imu03', 'imu10', 'imu11', 'imu12', 'imu13']
    window = 100  # sample window to cycle through all samples (i.e. train AE every x samples)
    batch_hist = 3000  # number of samples that you have to take every time you trian the network

    for sub in range(len(names)):
        # start from the AE parameters stored in start folder
        startPath = mainPath + names[sub] + sess[sub] + "AE/IMU/start/"

        # create path where batch AE parameters will be saved
        batchPath = mainPath + names[sub] + sess[sub] + "AE/IMU/batches/"
        if not os.path.exists(batchPath):
            os.mkdir(batchPath)

        # get imu array for current subject
        imu_tot = df.loc[idx[sub + 1], col_imu].values

        # start by training network pretrained with parameters in start folder.
        # then save newly obtained parameters in batchPath
        imu = imu_tot[0: batch_hist, :]
        hist, w, b, imu_rec_ae = AE.pre_train_network_sim(startPath, imu)
        vaf = useful_functions.compute_vaf(imu, imu_rec_ae) * 100

        # Save AE parameters in a file in current batch folder (after training)
        for layer in range(len(w)):
            np.savetxt(batchPath + "/weights" + str(layer + 1) + ".txt", w[layer])
            np.savetxt(batchPath + "/biases" + str(layer + 1) + ".txt", b[layer])

        # Append current batch to weights and biases
        w_hist = []
        b_hist = []
        vaf_hist = []
        for layer in range(len(w)):
            nw = len(w[layer])
            aw = np.ones((nw, 1)) * 0
            w_hist.append(np.hstack((w[layer], aw)))
            b_hist.append(np.hstack((b[layer], np.array([0, hist.history['loss'][-1]]))).reshape(
                (1, len(b[layer]) + 2)))
        vaf_hist.append(vaf)

        # Save AE parameters in history file (after training)
        for layer in range(len(w)):
            fw = open(batchPath + "/w" + str(layer + 1) + "_history.txt", 'ab')
            np.savetxt(fw, w_hist[layer])
            fw.close()
            fb = open(batchPath + "/b" + str(layer + 1) + "_loss_history.txt", 'ab')
            np.savetxt(fb, b_hist[layer])
            fb.close()
        fv = open(batchPath + "/vaf_history.txt", 'ab')
        np.savetxt(fv, vaf_hist)
        fv.close()

        # simulate online adaptation after first update
        for batch in range(1, int(len(imu_tot)/window)):
            imu = imu_tot[batch * window:batch * window + batch_hist, :]
            hist, w, b, imu_rec_ae = AE.pre_train_network_sim(batchPath, imu)
            vaf = useful_functions.compute_vaf(imu, imu_rec_ae) * 100

            # Save AE parameters in a file in current batch folder (after training)
            for layer in range(len(w)):
                np.savetxt(batchPath + "/weights" + str(layer + 1) + ".txt", w[layer])
                np.savetxt(batchPath + "/biases" + str(layer + 1) + ".txt", b[layer])

            # Append current batch to weights and biases
            w_hist = []
            b_hist = []
            vaf_hist = []
            for layer in range(len(w)):
                nw = len(w[layer])
                aw = np.ones((nw, 1)) * batch
                w_hist.append(np.hstack((w[layer], aw)))
                b_hist.append(np.hstack((b[layer], np.array([batch, hist.history['loss'][-1]]))).reshape(
                    (1, len(b[layer]) + 2)))
            vaf_hist.append(vaf)

            # Save AE parameters in history file (after training)
            for layer in range(len(w)):
                fw = open(batchPath + "/w" + str(layer + 1) + "_history.txt", 'ab')
                np.savetxt(fw, w_hist[layer])
                fw.close()
                fb = open(batchPath + "/b" + str(layer + 1) + "_loss_history.txt", 'ab')
                np.savetxt(fb, b_hist[layer])
                fb.close()
            fv = open(batchPath + "/vaf_history.txt", 'ab')
            np.savetxt(fv, vaf_hist)
            fv.close()

            print("Training completed for subject " + str(sub + 1) + " and batch "
                  + str(batch) + "/" + str(int(len(imu_tot)/window)))

        print("Training completed for subject " + str(sub + 1))
        print("\n")


def compute_metric_repetition(typ, pdf, names, metric, metric_test, tc_trial, y_label, repetition):
    """
        plot loss, rt and its fit. Mean across repetitions
        :param typ: string, either adaptive or static to plot in title
        :param names: list with subjects' names
        :param metric: list of list containing metric values during training er each subject
        :param metric_test: list of list containing metric values during test er each subject
        :param y_label: y label for plot containing name of the metric
        :return: plot metric and its fit, return fit learning rate
    """
    ci = 1.96
    r_sq = []
    tau = []
    metric_list = [[] for i in range(len(names))]
    metric_list_base = [[] for i in range(len(names))]
    metric_test_list = [[] for i in range(len(names))]

    # f, ax = plt.subplots(1, len(names))
    f, ax = plt.subplots(2, 5, sharex=True, sharey=True)
    f.suptitle('Performance with single exp fit for ' + typ + ' group', fontsize=14)
    axx = 0
    axy = 0

    for sub in range(len(names)):
        # handle index of subplot
        if sub == 5:
            axx = 1
            axy = 0

        metric_mu = []
        metric_mu_base = []
        tmp_mu = 0
        is_base = True  # bool that controls if you are in baseline
        tot_rep = int(len(metric[sub]) / repetition)

        # get number of trials completed during baseline (60s). Those will be excluded from the analysis
        tc_base = tc_trial[sub][0]

        # averaging reaching time per repetition
        for i in range(tot_rep):
            # first delete outliers, then append
            curr = metric[sub][tmp_mu: tmp_mu + repetition]
            mu = np.mean(curr)
            std = np.std(curr)
            for index, item in enumerate(curr):
                if not mu - ci*std <= item < mu + ci*std:
                    curr[index] = float('NaN')
            metric_mu.append(np.nanmean(curr))
            # in metric_mu_base exclude baseline (60s)
            if tc_base < tmp_mu + repetition and is_base is True:
                metric_mu_base.append(np.nanmean(curr[int(tc_base-repetition):]))
                is_base = False
            else:
                if is_base is True:
                    metric_mu_base.append(float('NaN'))
                else:
                    metric_mu_base.append(np.nanmean(curr))
            tmp_mu += repetition

        metric_test_mu = []
        tmp_test_mu = 0
        tot_rep_test = int(len(metric_test[sub]) / 8)
        # averaging reaching time per repetition
        for i in range(tot_rep_test):
            # first delete outliers, then append
            curr = metric_test[sub][tmp_test_mu: tmp_test_mu + 8]
            mu = np.mean(curr)
            std = np.std(curr)
            for index, item in enumerate(curr):
                if not mu - ci * std <= item < mu + ci * std:
                    curr[index] = float('NaN')
            metric_test_mu.append(np.nanmean(curr))
            tmp_test_mu += 8

        # prepare movement time array
        metric_mu = np.array(metric_mu)
        metric_mu = np.reshape(metric_mu, (tot_rep,))
        metric_mu = metric_mu.astype(float)
        metric_list[sub] = metric_mu
        metric_mu_base = np.array(metric_mu_base)
        metric_mu_base = np.reshape(metric_mu_base, (tot_rep,))
        metric_mu_base = metric_mu_base.astype(float)
        metric_list_base[sub] = metric_mu_base
        metric_test_mu = np.array(metric_test_mu)
        metric_test_mu = np.reshape(metric_test_mu, (tot_rep_test,))
        metric_test_mu = metric_test_mu.astype(float)
        metric_test_list[sub] = metric_test_mu

        # exponential fit on metric
        x = np.arange(tot_rep)
        y = metric_mu
        id = np.isfinite(x) & np.isfinite(y)
        # popt, pcov = curve_fit(func, x, y, p0=[max(y), 1/5, min(y)], maxfev=2000)
        popt, pcov = curve_fit(func, x[id], y[id], maxfev=2000)
        tau.append(1/popt[1])
        est = func(x, popt[0], popt[1], popt[2])
        res = y - est
        ss_res = np.sum(res**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_sq.append(1 - (ss_res / ss_tot))

        # plot everything
        # ax[sub].plot(loss * (max(metric)/max(loss)), color='orange', label='loss J')
        ax[axx, axy].plot(est, color='green', label='fit')
        ax[axx, axy].scatter(x, y, color='black', s=5 * 2 ** 1e-4)
        if sub == 7:
            ax[axx, axy].set_xlabel('repetitions', fontsize=10)
        if sub == len(names)-1:
            ax[axx, axy].legend(frameon=False, fontsize=9)
        if sub == 0:
            ax[axx, axy].set_ylabel(y_label, fontsize=10)
        ax[axx, axy].set_title('Subject ' + str(sub + 1), fontsize=12)
        # get rid of the frame
        ax[axx, axy].spines['right'].set_color('none')
        ax[axx, axy].spines['top'].set_color('none')
        ax[axx, axy].set_ylim([0, 15])
        ax[axx, axy].set_yticks([0, 2.5, 5, 7.5, 10, 12.5, 15])
        # ax[axx, axy].set_ylim([0, np.nanmax(np.nanmax(metric))/2])
        ax[axx, axy].set_xlim([-2, tot_rep+2])
        # set fontsize for xticks and yticks
        for tick in ax[axx, axy].xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        for tick in ax[axx, axy].yaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        ax[axx, axy].text(0.5, 0.75, r'$R^2$: ' + str(np.ceil(r_sq[sub] * 100) / 100),
                          fontsize=9, transform=ax[axx, axy].transAxes)
        # ax[axx, axy].text(len(x)*60/100, 30*90/100, r'$R^2$: ' + str(np.ceil(r_sq[sub] * 100) / 100), fontsize=9)

        axy += 1

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    # pdf.savefig(f, dpi=300, papertype='letter', orientation='landscape', pad_inches=0)
    # plt.close()

    return metric_list, metric_list_base, metric_test_list, r_sq, tau


def compute_metrics_trial(df_reach, names, trial_list, tgt, tgt_test, tgt_list, idx, elapsed):
    """
        compute reaching time (rt) for every trial ,trials completed (tc) every 2 mins
        :param df_reach: dataframe with reaching data
        :param names: list with subjects' names
        :param trial_list: list with number of trials per each subject
        :param idx: indexslice for handling dataframes
        :return: plot loss against rt and its fit, corr and values between loss and rt
    """
    df_reach = df_reach.reset_index()
    df_reach = df_reach.set_index(["Subject", "trial", "comeback"])

    rt_list = [[] for i in range(len(names))]
    rt_test_list = [[] for i in range(len(names))]
    re_list = [[] for i in range(len(names))]
    re_test_list = [[] for i in range(len(names))]
    asr_list = [[] for i in range(len(names))]
    asr_test_list = [[] for i in range(len(names))]
    pks_list = [[] for i in range(len(names))]
    pks_test_list = [[] for i in range(len(names))]
    tc_list = [[] for i in range(len(names))]
    tc_end_list = [[] for i in range(len(names))]
    tc_li_list = [[] for i in range(len(names))]
    tc_li_end_list = [[] for i in range(len(names))]
    tc_ms_list = [[] for i in range(len(names))]
    tc_ms_end_list = [[] for i in range(len(names))]

    # compute reaching time for training and test blocks
    for sub in range(len(names)):
        # initializing lists for all metrics
        rt = []  # reaching time
        rt_test = []
        re = []  # reaching error
        re_test = []
        asr = []  # aspect ratio
        asr_test = []
        pks = []  #number of peaks
        pks_test = []
        tc = []  # list containing trial completed in 60s windows from beginning
        tc_end = []  # list containing trial completed in 60s windows from end
        tc_li = []  # list containing linearity index in 60s windows from beginning
        tc_li_end = []  # list containing linearity index in 60s windows from end
        tc_ms = []  # list containing smoothness index in 60s windows from beginning
        tc_ms_end = []  # list containing smoothness index in 60s windows from end

        first_trials = [9, 37, 65, 93, 129, 157, 185, 213]   # list containing first trails of each training block.
        good_trials = []   # list containing only trials of training wout blind trials
        good_trials.append(np.arange(9, 33))
        good_trials.append(np.arange(37, 61))
        good_trials.append(np.arange(65, 89))
        good_trials.append(np.arange(93, 117))
        good_trials.append(np.arange(129, 153))
        good_trials.append(np.arange(157, 181))
        good_trials.append(np.arange(185, 209))
        good_trials.append(np.arange(213, 237))
        good_trials = [item for sublist in good_trials for item in sublist]

        good_trials_s2 = good_trials[:168]

        # append metric values for each trial (single subject)
        for trial in range(1, trial_list[sub]):
            # exclude blind trials
            if 1 not in df_reach.loc[idx[sub + 1, trial, 0], ['is_blind']].values:
                # reaching time
                start = df_reach.loc[idx[sub + 1, trial, 0], ['time']].values[0]
                end = df_reach.loc[idx[sub + 1, trial, 0], ['time']].values[-1]
                # path length
                traj = df_reach.loc[idx[sub + 1, trial, 0], ['cursor_x', 'cursor_y']].values
                dist = np.sum(np.sqrt(np.diff(traj[:, 0])**2 + np.diff(traj[:, 1])**2))
                # smoothness index
                vel_x = savgol_filter(traj[:, 0], 13, 4, 1)
                vel_y = savgol_filter(traj[:, 1], 13, 4, 1)
                speed = np.sqrt(vel_x**2 + vel_y**2)
                i_pks, y_pks = find_peaks(speed, height=np.max(speed)*0.15, distance=25)
                npk = len(i_pks)
                # this is for training
                if [1, 6, 11] not in df_reach.loc[idx[sub + 1, trial, 0], ['block']].values:
                    # RT
                    rt.append((end - start) / 1000)
                    # RE
                    tgt_curr = tgt[:, tgt_list[trial - 1]]
                    try:
                        curs = df_reach.loc[idx[sub + 1, trial, 0], ['cursor_x', 'cursor_y']].values[elapsed]
                    except:
                        curs = tgt_curr
                    re_tmp = distance.euclidean(curs, tgt_curr) * 0.02767
                    re.append(re_tmp)
                    # ASR
                    if trial in first_trials:
                        tgt_prev = [960, 540]
                    else:
                        tgt_prev = tgt[:, tgt_list[trial - 2]]
                    dist_norm = distance.euclidean(tgt_curr, tgt_prev)
                    # asr.append(useful_functions.compute_asr(traj, dist, tgt_curr))
                    asr.append(dist / dist_norm)
                    # PKS
                    pks.append(npk)
                    # this is for test
                else:
                    # reaching time
                    rt_test.append((end - start) / 1000)
                    # reaching error
                    tgt_curr = tgt_test[:, tgt_list[trial - 1]]
                    try:
                        curs = df_reach.loc[idx[sub + 1, trial, 0], ['cursor_x', 'cursor_y']].values[elapsed]
                    except:
                        curs = tgt_curr
                    re_tmp = distance.euclidean(curs, tgt_curr) * 0.02767
                    re_test.append(re_tmp)
                    # ASR
                    tgt_prev = [960, 540]
                    dist_norm = distance.euclidean(tgt_curr, tgt_prev)
                    asr_test.append(dist / dist_norm)
                    # PKS
                    pks_test.append(npk)

        # use cumsum to compute trials completed every 60s (starting from the beginning)
        rt_cum = np.cumsum(rt)
        count_time = 1
        count_trial = 0
        for id, value in enumerate(rt_cum):
            if value / 60 > count_time:
                tc.append(count_trial)
                count_time += 1
                count_trial = 1
            else:
                count_trial += 1
            if not count_time - 1 < value / 60 < count_time and count_time != 1:
                tc.append(0)
                count_time += 1

        # take number of trials completed each 60s and compute linearity/smoothness
        for id, value in enumerate(tc):
            # update ids to get init and end trial using good_trials
            if id == 0:
                init_id = 0
                end_id = value
            else:
                init_id = end_id + 1
                end_id = end_id + value
            tmp_li = []
            tmp_ms = []
            # evaluate only trials within good_trials list
            eval_trials = good_trials[init_id:end_id+1]
            for _, count_trial in enumerate(eval_trials):
                # aspect ratio
                traj = df_reach.loc[idx[sub + 1, count_trial, 0], ['cursor_x', 'cursor_y']].values
                dist = np.sum(np.sqrt(np.diff(traj[:, 0]) ** 2 + np.diff(traj[:, 1]) ** 2))
                # smoothness index
                vel_x = savgol_filter(traj[:, 0], 13, 4, 1)
                vel_y = savgol_filter(traj[:, 1], 13, 4, 1)
                speed = np.sqrt(vel_x ** 2 + vel_y ** 2)
                i_pks, y_pks = find_peaks(speed, height=np.max(speed) * 0.15, distance=25)
                npk = len(i_pks)
                # ASR
                tgt_curr = tgt[:, tgt_list[count_trial - 1]]
                if count_trial in first_trials:
                    tgt_prev = [960, 540]
                else:
                    tgt_prev = tgt[:, tgt_list[count_trial - 2]]
                dist_norm = distance.euclidean(tgt_curr, tgt_prev)
                # asr.append(useful_functions.compute_asr(traj, dist, tgt_curr))
                tmp_li.append(dist / dist_norm)
                # PKS
                tmp_ms.append(npk)
            tc_li.append(np.nanmean(tmp_li))
            # PKS
            tc_ms.append(np.nanmean(tmp_ms))

        # use cumsum to compute trials completed every 60s (starting from the end)
        rt_cum_end = np.flip(rt_cum)
        rt_cum_end = rt_cum_end - rt_cum_end[0]
        rt_cum_end = np.abs(rt_cum_end)
        count_time = 1
        count_trial = 0
        for id, value in enumerate(rt_cum_end):
            if value / 60 > count_time:
                tc_end.append(count_trial)
                count_time += 1
                count_trial = 1
            else:
                count_trial += 1
            if not count_time - 1 < value / 60 < count_time and count_time != 1:
                tc_end.append(0)
                count_time += 1

        # take number of trials_end completed each 60s and compute linearity/smoothness
        for id, value in enumerate(tc_end):
            # update ids to get init and end trial using good_trials
            if id == 0:
                end_id = -1
                init_id = -1-value
            else:
                end_id = init_id
                init_id = init_id - value
                if init_id < -192:
                    init_id = -192
            tmp_li = []
            tmp_ms = []
            # evaluate only trials within good_trials list
            if sub == 1:
                eval_trials = good_trials_s2[init_id:end_id]
            else:
                eval_trials = good_trials[init_id:end_id]
            for _, count_trial in enumerate(eval_trials):
                # aspect ratio
                traj = df_reach.loc[idx[sub + 1, count_trial, 0], ['cursor_x', 'cursor_y']].values
                dist = np.sum(np.sqrt(np.diff(traj[:, 0]) ** 2 + np.diff(traj[:, 1]) ** 2))
                # smoothness index
                vel_x = savgol_filter(traj[:, 0], 13, 4, 1)
                vel_y = savgol_filter(traj[:, 1], 13, 4, 1)
                speed = np.sqrt(vel_x ** 2 + vel_y ** 2)
                i_pks, y_pks = find_peaks(speed, height=np.max(speed) * 0.15, distance=25)
                npk = len(i_pks)
                # ASR
                tgt_curr = tgt[:, tgt_list[count_trial - 1]]
                if count_trial in first_trials:
                    tgt_prev = [960, 540]
                else:
                    tgt_prev = tgt[:, tgt_list[count_trial - 2]]
                dist_norm = distance.euclidean(tgt_curr, tgt_prev)
                # asr.append(useful_functions.compute_asr(traj, dist, tgt_curr))
                tmp_li.append(dist / dist_norm)
                # PKS
                tmp_ms.append(npk)
            tc_li_end.append(np.nanmean(tmp_li))
            # PKS
            tc_ms_end.append(np.nanmean(tmp_ms))

        # count_trial = 0
        # # append reaching time for each trial (single subject)
        # for trial in range(1, trial_list[sub] + 1):
        #     # exclude blind trials
        #     if 1 not in df_reach.loc[idx[sub + 1, trial, 0], ['isBlind']].values:
        #         # only training
        #         if [1, 6, 11] not in df_reach.loc[idx[sub + 1, trial, 0], ['block']].values:
        #             if trial in [9, 37, 65, 93, 129, 157, 195, 213]:
        #                 start_train = df_reach.loc[idx[sub + 1, trial, 0], ['time']].values[0]
        #             elapsed_time = df_reach.loc[idx[sub + 1, trial, 0], ['time']].values[0]
        #             if ((elapsed_time - start_train) / 60000) < 1:
        #                 count_trial += 1
        #             else:
        #                 trial_coae.append(count_trial)
        #                 count_trial = 0

        # prepare list of lists for all participants

        # reaching time
        rt_list[sub] = append_nan(rt, 192)
        rt_test_list[sub] = append_nan(rt_test, 24)
        # reaching error
        re_list[sub] = append_nan(re, 192)
        re_test_list[sub] = append_nan(re_test, 24)
        # aspect ratio
        asr_list[sub] = append_nan(asr, 192)
        asr_test_list[sub] = append_nan(asr_test, 24)
        # number of peaks
        pks_list[sub] = append_nan(pks, 192)
        pks_test_list[sub] = append_nan(pks_test, 24)

        tc = np.array(tc)
        tc = np.reshape(tc, (tc.shape[0],))
        tc = tc.astype(float)
        tc_list[sub] = tc
        tc_li = np.array(tc_li)
        tc_li = np.reshape(tc_li, (tc_li.shape[0],))
        tc_li = tc_li.astype(float)
        tc_li_list[sub] = tc_li
        tc_ms = np.array(tc_ms)
        tc_ms = np.reshape(tc_ms, (tc_ms.shape[0],))
        tc_ms = tc_ms.astype(float)
        tc_ms_list[sub] = tc_ms

        tc_end = np.array(tc_end)
        tc_end = np.reshape(tc_end, (tc_end.shape[0],))
        tc_end = tc_end.astype(float)
        tc_end_list[sub] = tc_end
        tc_li_end = np.array(tc_li_end)
        tc_li_end = np.reshape(tc_li_end, (tc_li_end.shape[0],))
        tc_li_end = tc_li_end.astype(float)
        tc_li_end_list[sub] = tc_li_end
        tc_ms_end = np.array(tc_ms_end)
        tc_ms_end = np.reshape(tc_ms_end, (tc_ms_end.shape[0],))
        tc_ms_end = tc_ms_end.astype(float)
        tc_ms_end_list[sub] = tc_ms_end

    return rt_list, rt_test_list, re_list, re_test_list, asr_list, asr_test_list, pks_list, pks_test_list, \
           tc_list, tc_end_list, tc_li_list, tc_li_end_list, tc_ms_list, tc_ms_end_list


def compute_rt_resample(df_reach, df_b2, names, trial_list, idx, fs):
    """
        plot loss and rt resampled with specific time bin
        :param df_reach: dataframe with reaching data
        :param df_b2: dataframe with encoder bias that contains loss history
        :param names: list with subjects' names
        :param trial_list: list with number of trials per each subject
        :param idx: indexslice for handling dataframes
        :param fs: new sampling frequency set for each trial
        :return: plot loss against rt, corr and xcorr values between loss and rt (resampled at fs)
    """
    df_reach = df_reach.reset_index()
    df_reach = df_reach.set_index(["Subject", "trial", "comeback"])

    corr = []
    rt_list = [[] for i in range(len(names))]

    f, ax = plt.subplots(2, 5)
    # f, ax = plt.subplots(1, len(names))
    f.suptitle('Loss vs similarity trend', fontsize=14)

    axx = 0  # index to iterate rows of subplot
    axy = 0  # index to iterate cols of subplot

    for sub in range(len(names)):
        # handle index of subplot
        if sub == 5:
            axx = 1
            axy = 0

        rt = []
        index_rt = []
        count = 0
        prev = 0
        for trial in range(1, trial_list[sub] + 1):
            # exclude blind trials
            if 1 not in df_reach.loc[idx[sub + 1, trial, 0], ['isBlind']].values:
                # exclude test blocks
                if [1, 6, 11] not in df_reach.loc[idx[sub + 1, trial, 0], ['block']].values:
                    count += 1
                    start = df_reach.loc[idx[sub + 1, trial, 0], ['time']].values[0]
                    end = df_reach.loc[idx[sub + 1, trial, 0], ['time']].values[-1]
                    curr = (end - start) / 1000
                    rt.append(curr)
                    index_rt.append(prev + curr)
                    prev += curr

        # prepare rt and time vector array
        rt = np.array(rt)
        rt = np.reshape(rt, (rt.shape[0],))
        rt = rt.astype(float)
        index_rt = np.array(index_rt)
        index_rt = np.reshape(index_rt, (index_rt.shape[0],))
        index_rt = index_rt.astype(float)

        # prepare loss array
        loss = df_b2.loc[idx[sub+1, :], ['loss']].values

        # resample rt to prefixed ts
        rt_res, index_rt_res = resample(index_rt, rt, new_sampling_freq=fs)
        rt_res = np.reshape(rt_res, (rt_res.shape[0], 1))
        rt_list[sub] = rt_res
        # make sure that loss has the same size of rt_res
        loss = resample_brutal(loss, len(rt_res), fs)

        x = np.arange(len(rt_res))
        y = rt_res

        # compute correaltion between loss and rt
        # corr_tmp = np.corrcoef(loss, y)
        loss = loss.reshape(len(loss),)
        y = y.reshape(len(y), )
        corr_tmp = pearsonr(loss, y)
        corr.append(corr_tmp[0])

        # plot everything
        ax[axx, axy].plot(loss * (max(rt)/max(loss)), color='orange', label='loss J')
        ax[axx, axy].plot(x, y, color='black', label='Reaching time')
        # ax[axx, axy].scatter(x, y, color='black', label='Reaching time', s=5*2**1e-4)
        ax[axx, axy].set_xlabel('time bin (' + str(int(1/fs)) + 's)', fontsize=10)
        ax[axx, axy].set_title('Subject ' + str(sub + 1), fontsize=10)
        ax[axx, axy].spines['right'].set_color('none')
        ax[axx, axy].spines['top'].set_color('none')
        ax[axx, axy].set_yticks([0, 10, 20, 30, 40])
        ax[axx, axy].set_ylim([0, 40])
        if sub == len(names)-1:
            ax[axx, axy].legend(frameon=False, fontsize=8, loc='upper right')
        elif sub == 0:
            ax[axx, axy].set_ylabel('[s]', fontsize=10)
        # set fontsize for xticks and yticks
        for tick in ax[axx, axy].xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        for tick in ax[axx, axy].yaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        # write correlation values
        ax[axx, axy].text(0.6, 0.9, r'$R^2$: ' + str(np.ceil(corr[sub] * 100) / 100),
                          fontsize=8, transform=ax[axx, axy].transAxes)

        axy += 1

    plt.show()

    return corr, rt_list


def compute_tSNE(pdf, names, df_w1, df_w2, df_b1, df_b2, idx):
    """
    This function compute tSNE to high dimensional data (stack of encoder parameters)
    :param pdf: path for prinitng figures in pdf format
    :param names: list of names for the study
    :param df_w1: dataframe containing weights of the first layer
    :param df_w2: dataframe containing weights of the second layer
    :param df_b1: dataframe containing biases of the first layer
    :param df_b2: dataframe containing biases of the second layer
    :param idx: idx for pandas
    :return:
    """
    # creating dataframes for t-SNE
    df_w1_tsne = df_w1.groupby(df_w1.index).agg(lambda x: list(x))
    df_w1_tsne = (df_w1_tsne['w1'] + df_w1_tsne['w2'] + df_w1_tsne['w3'] + df_w1_tsne['w4'] +
                  df_w1_tsne['w5'] + df_w1_tsne['w6'] + df_w1_tsne['w7'] + df_w1_tsne['w8']).to_frame()
    df_w1_tsne.index = pd.MultiIndex.from_tuples(df_w1_tsne.index)

    df_b1_tsne = df_b1.groupby(df_b1.index).agg(lambda x: list(x))
    df_b1_tsne = (df_b1_tsne['b1'] + df_b1_tsne['b2'] + df_b1_tsne['b3'] + df_b1_tsne['b4'] +
                  df_b1_tsne['b5'] + df_b1_tsne['b6'] + df_b1_tsne['b7'] + df_b1_tsne['b8']).to_frame()
    df_b1_tsne.index = pd.MultiIndex.from_tuples(df_b1_tsne.index)

    df_w2_tsne = df_w2.groupby(df_w2.index).agg(lambda x: list(x))
    df_w2_tsne = (df_w2_tsne['w1'] + df_w2_tsne['w2']).to_frame()
    df_w2_tsne.index = pd.MultiIndex.from_tuples(df_w2_tsne.index)

    df_b2_tsne = df_b2.groupby(df_b2.index).agg(lambda x: list(x))
    df_b2_tsne = (df_b2_tsne['b1'] + df_b2_tsne['b2']).to_frame()
    df_b2_tsne.index = pd.MultiIndex.from_tuples(df_b2_tsne.index)

    df_tsne = df_w1_tsne + df_w2_tsne + df_b1_tsne + df_b2_tsne

    encoder_param_list = [[] for i in range(len(names))]
    x_tsne_list = [[] for i in range(len(names))]

    for sub in range(len(names)):
        x_tuple = df_tsne.loc[idx[sub + 1, :], :].values

        # need to convert a tuple into list of lists, then numpy array
        x_list = []
        for item in x_tuple:
            x_list.extend(item)
        x = np.array([np.array(xi) for xi in x_list])
        # x = np.diff(x)

        # # pre-applying PCA
        # PCA = PrincipalComponentAnalysis(50)
        # x = PCA.train_pca(x)

        # training tSNE
        x_tsne = TSNE(n_components=2).fit_transform(x)
        # plot
        # fig_handle = plt.figure()
        plt.figure()
        plt.plot(x_tsne[:, 0], x_tsne[:, 1])
        plt.scatter(x_tsne[0, 0], x_tsne[0, 1], color='green')
        plt.scatter(x_tsne[-1, 0], x_tsne[-1, 1], color='red')
        plt.axis('equal')
        plt.title('Subject ' + str(sub + 1))
        # manager = plt.get_current_fig_manager()
        # manager.window.showMaximized()
        # pdf.savefig(fig_handle, dpi=300, papertype='letter', orientation='landscape', pad_inches=0)
        # plt.close()

        # # velocity control
        # vel = TSNE(n_components=2).fit_transform(x)
        #
        # speed = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2)
        # dir = np.arctan2(vel[:, 1], vel[:, 0])
        #
        # # compute workspace. start from initial pos and then apply velocity
        # pos = []
        # pos.append(np.array([0, 0]))
        # for i in range(len(vel)):
        #     px = pos[i][0] + 10*speed[i] * np.cos(dir[i])
        #     py = pos[i][1] + 10*speed[i] * np.sin(dir[i])
        #     pos.append(np.array([px, py]))
        # pos = np.array(pos)
        #
        # # define colormap for each velocity point
        # start = 0.0
        # stop = 1.0
        # number_of_lines = int(len(pos))
        # cm = np.linspace(start, stop, number_of_lines)
        #
        # # order speed to match colors
        # id_speed = speed.argsort()
        # id_id_speed = id_speed.argsort()
        #
        # # B = sorted(range(len(speed)), key=lambda x:speed[x], reverse=True)
        # # C = sorted(range(len(speed)), key=lambda x:B[x])
        #
        # cm_sort = cm[id_id_speed[::-1]]
        # cmap = plt.cm.get_cmap('Reds')
        # # create final vector that contains colormap
        # # colors = [plt.cm.jet(x) for x in cm_sort]
        # colors = [cmap(x) for x in cm_sort]
        #
        # plt.figure()
        # plt.plot(speed, color='red', label='speed')
        # plt.legend()
        #
        # # plot
        # # fig_handle = plt.figure()
        # # plt.figure()
        # fig, ax1 = plt.subplots()  # setup the plot
        # sc = ax1.scatter(pos[1:, 0], pos[1:, 1], color=colors)
        # fig.colorbar(sc, ax=ax1)
        # # plt.scatter(pos[0, 0], pos[0, 1], color='green')
        # # plt.scatter(pos[-1, 0], pos[-1, 1], color='red')
        # plt.axis('equal')
        # plt.title('Subject ' + str(sub + 1))
        # # manager = plt.get_current_fig_manager()
        # # manager.window.showMaximized()
        # # pdf.savefig(fig_handle, dpi=300, papertype='letter', orientation='landscape', pad_inches=0)
        # # plt.close()

        encoder_param_list[sub] = x
        x_tsne_list[sub] = x_tsne

    return encoder_param_list, x_tsne_list


def compute_vaf_batch(path, names, df_reach, df_w1, df_w2, df_w3, df_w4, df_b1, df_b2, df_b3, df_b4, idx):
    """

    :param names:
    :param df_reach:
    :param df_w1:
    :param df_w2:
    :param df_w3:
    :param df_w4:
    :param df_b1:
    :param df_b2:
    :param df_b3:
    :param df_b4:
    :param idx:
    :return:
    """

    col_imu = ['imu00', 'imu01', 'imu02', 'imu03', 'imu10', 'imu11', 'imu12', 'imu13']
    col_b8 = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8']
    blind_list = [33, 34, 35, 36, 61, 62, 63, 64, 89, 90, 91, 92, 117, 118, 119, 120,
                  153, 154, 155, 156, 181, 182, 183, 184, 209, 210, 211, 212, 237, 238, 239, 240]
    col_b2 = ['b1', 'b2']

    # delete blind trials and test blocks
    df_reach = df_reach.reset_index()
    df_reach = df_reach[~df_reach['block'].isin([1, 6, 11])]
    df_reach = df_reach[~df_reach['trial'].isin(blind_list)]

    df_reach = df_reach.set_index(["Subject", "batch"])
    vaf_list = [[] for i in range(len(names))]  # list of lists that contains vaf every batch for each subject
    vaf_list_res = [[] for i in range(len(names))]  # list of lists that contains resampled vaf every batch for each subject

    # load initial AE map
    w1 = pd.read_csv(path + '/AE_start/weights1.txt', sep=' ', header=None)
    w2 = pd.read_csv(path + '/AE_start/weights2.txt', sep=' ', header=None)
    w3 = pd.read_csv(path + '/AE_start/weights3.txt', sep=' ', header=None)
    w4 = pd.read_csv(path + '/AE_start/weights4.txt', sep=' ', header=None)
    b1 = pd.read_csv(path + '/AE_start/biases1.txt', sep=' ', header=None).values
    b1 = b1.reshape((len(b1),))
    b2 = pd.read_csv(path + '/AE_start/biases2.txt', sep=' ', header=None).values
    b2 = b2.reshape((len(b2),))
    b3 = pd.read_csv(path + '/AE_start/biases3.txt', sep=' ', header=None).values
    b3 = b3.reshape((len(b3),))
    b4 = pd.read_csv(path + '/AE_start/biases4.txt', sep=' ', header=None).values
    b4 = b4.reshape((len(b4),))

    # cycle participants
    for sub in range(len(names)):
        # cycle through batches
        for batch in range(len(df_b1.loc[idx[sub+1], :])):

            if batch == 0:
                # take batch of imu data from df_reach
                imu_prev = df_reach.loc[idx[sub + 1, batch + 1], col_imu].values[-2900:]
            else:
                # take AE parameters for each batch
                w1 = df_w1.loc[idx[sub + 1, batch + 1], :].values
                w2 = df_w2.loc[idx[sub + 1, batch + 1], :].values
                w3 = df_w3.loc[idx[sub + 1, batch + 1], :].values
                w4 = df_w4.loc[idx[sub + 1, batch + 1], :].values
                b1 = df_b1.loc[idx[sub + 1, batch + 1], col_b8].values
                b2 = df_b2.loc[idx[sub + 1, batch + 1], col_b2].values
                b3 = df_b3.loc[idx[sub + 1, batch + 1], col_b8].values
                b4 = df_b4.loc[idx[sub + 1, batch + 1], col_b8].values

                imu_curr = df_reach.loc[idx[sub + 1, batch + 1], col_imu].values
                imu_prev = np.vstack((imu_prev, imu_curr))
                if len(imu_prev) > 3000:
                    imu_prev = imu_prev[-3000:]

            # apply AE model and compute VAF
            h1 = np.tanh(np.dot(imu_prev, w1) + b1)
            cu = np.dot(h1, w2) + b2
            h3 = np.tanh(np.dot(cu, w3) + b3)
            imu_rec = np.dot(h3, w4) + b4
            vaf = useful_functions.compute_vaf(imu_prev, imu_rec)

            vaf_list[sub].append(vaf)

        # make sure that vaf_resample has 200 samples to be averaged
        vaf_list_res[sub] = resample_brutal(vaf_list[sub], 200, 0.5)

    return vaf_list, vaf_list_res


def compute_vaf_batch_static(path, names, df_reach, idx):
    """

    :param names:
    :param df_reach:
    :param df_w1:
    :param df_w2:
    :param df_w3:
    :param df_w4:
    :param df_b1:
    :param df_b2:
    :param df_b3:
    :param df_b4:
    :param idx:
    :return:
    """

    col_imu = ['imu00', 'imu01', 'imu02', 'imu03', 'imu10', 'imu11', 'imu12', 'imu13']
    col_b8 = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8']
    blind_list = [33, 34, 35, 36, 61, 62, 63, 64, 89, 90, 91, 92, 117, 118, 119, 120,
                  153, 154, 155, 156, 181, 182, 183, 184, 209, 210, 211, 212, 237, 238, 239, 240]
    col_b2 = ['b1', 'b2']

    df_reach = df_reach.reset_index()
    df_reach = df_reach[~df_reach['block'].isin([1, 6, 11])]
    df_reach = df_reach[~df_reach['trial'].isin(blind_list)]

    df_reach = df_reach.set_index(["Subject"])
    vaf_list = [[] for i in range(len(names))]  # list of lists that contains vaf every batch for each subject
    vaf_list_res = [[] for i in range(len(names))]  # list of lists that contains resampled vaf every batch for each subject

    # load initial AE map
    w1 = pd.read_csv(path + '/AE_start/weights1.txt', sep=' ', header=None)
    w2 = pd.read_csv(path + '/AE_start/weights2.txt', sep=' ', header=None)
    w3 = pd.read_csv(path + '/AE_start/weights3.txt', sep=' ', header=None)
    w4 = pd.read_csv(path + '/AE_start/weights4.txt', sep=' ', header=None)
    b1 = pd.read_csv(path + '/AE_start/biases1.txt', sep=' ', header=None).values
    b1 = b1.reshape((len(b1),))
    b2 = pd.read_csv(path + '/AE_start/biases2.txt', sep=' ', header=None).values
    b2 = b2.reshape((len(b2),))
    b3 = pd.read_csv(path + '/AE_start/biases3.txt', sep=' ', header=None).values
    b3 = b3.reshape((len(b3),))
    b4 = pd.read_csv(path + '/AE_start/biases4.txt', sep=' ', header=None).values
    b4 = b4.reshape((len(b4),))

    # cycle participants
    for sub in range(len(names)):
        # get number of batches
        len_df = 1 + int((len(df_reach.loc[idx[sub+1], :]) - 3000) / 100)
        # cycle through batches
        for batch in range(len_df):

            if batch == 0:
                # take batch of imu data from df_reach
                imu_curr = df_reach.loc[idx[sub + 1], col_imu].values[0:3000]
                batch_start = 100
            else:
                imu_curr = df_reach.loc[idx[sub + 1], col_imu].values[batch_start:batch_start+3000]
                batch_start = batch_start + 100

            # apply AE model and compute VAF
            h1 = np.tanh(np.dot(imu_curr, w1) + b1)
            cu = np.dot(h1, w2) + b2
            h3 = np.tanh(np.dot(cu, w3) + b3)
            imu_rec = np.dot(h3, w4) + b4
            vaf = useful_functions.compute_vaf(imu_curr, imu_rec) * 100

            vaf_list[sub].append(vaf)

        # make sure that vaf_resample has 200 samples to be averaged
        vaf_list_res[sub] = resample_brutal(vaf_list[sub], 200, 0.5)

    return vaf_list, vaf_list_res


def compute_vaf_time(df, names, idx):
    """
        compute vaf every 2 mins
        :param df_reach: dataframe with reaching data
        :param names: list with subjects' names
        :param idx: indexslice for handling dataframes
        :return:vaf
    """
    # remove unnecessary rows from df_reach. you don't need out-center, blind and test trials
    df = df.reset_index()
    df = df[df['comeback'] == 0]
    df = df[df['isBlind'] == 0]
    df = df[~df['block'].isin([1, 6, 11])]

    # need index for cycling through subjects
    df = df.set_index(["Subject"])

    # prepare vaf list for all subjects
    vaf_pca_list = [[] for i in range(len(names))]
    vaf_ae_list = [[] for i in range(len(names))]

    # import pca and autoencoder from ae_package
    AE = autoencoder2_0.Autoencoder(1000, 0.02, 2, struc="non_linear", nh1=8)
    PCA = pca.PrincipalComponentAnalysis(2)
    col_imu = ['imu00', 'imu01', 'imu02', 'imu03', 'imu10', 'imu11', 'imu12', 'imu13']

    fs = 50  # sampling rate
    t = 120  # compute every vaf every t sec
    x_range = fs * t
    for sub in range(len(names)):
        vaf_pca = []  # list containing VAF PCA for one subject
        vaf_ae = []  # list containing VAF AE for one subject

        # get imu array for current subject
        imu_tot = df.loc[idx[sub+1], col_imu].values

        # compute vaf for baseline
        imu = imu_tot[0:int(x_range/2), :]
        imu_rec_pca = PCA.train_pca(imu)
        vaf_pca.append(useful_functions.compute_vaf(imu, imu_rec_pca) * 100)
        _, _, imu_rec_ae, _ = AE.train_network(imu)
        vaf_ae.append(useful_functions.compute_vaf(imu, imu_rec_ae) * 100)

        # remove baseline
        imu_tot = imu_tot[int(x_range/2):, :]

        # compute PCA every tot minute after baseline
        for i in range(int(len(imu_tot) / x_range)+1):
            imu = imu_tot[i*x_range:(i+1)*x_range, :]
            imu_rec_pca = PCA.train_pca(imu)
            vaf_pca.append(useful_functions.compute_vaf(imu, imu_rec_pca) * 100)
            _, _, imu_rec_ae, _ = AE.train_network(imu)
            vaf_ae.append(useful_functions.compute_vaf(imu, imu_rec_ae) * 100)

            print("Training completed for subject " + str(sub + 1) + " and time "
                  + str(i) + "/" + str(int(len(imu_tot) / x_range)))
            print("\n")

        # compute vaf for last tot minute of training
        imu = imu_tot[-int(x_range / 2):, :]
        imu_rec_pca = PCA.train_pca(imu)
        vaf_pca.append(useful_functions.compute_vaf(imu, imu_rec_pca) * 100)
        _, _, imu_rec_ae, _ = AE.train_network(imu)
        vaf_ae.append(useful_functions.compute_vaf(imu, imu_rec_ae) * 100)

        print("\n")
        print("Training completed for subject " + str(sub + 1))
        print("\n")

        # assemble list of lists for vaf
        vaf_pca = np.array(vaf_pca)
        vaf_pca = np.reshape(vaf_pca, (vaf_pca.shape[0],))
        vaf_pca = vaf_pca.astype(float)
        vaf_pca_list[sub] = vaf_pca

        vaf_ae = np.array(vaf_ae)
        vaf_ae = np.reshape(vaf_ae, (vaf_ae.shape[0],))
        vaf_ae = vaf_ae.astype(float)
        vaf_ae_list[sub] = vaf_ae

    return vaf_pca_list, vaf_ae_list


def compute_vaf_trial(df, names, trial_list, trials_pca, idx):
    """
        compute vaf completed (tc) every 2 mins
        :param df_reach: dataframe with reaching data
        :param names: list with subjects' names
        :param idx: indexslice for handling dataframes
        :return:vaf
    """
    # remove unnecessary rows from df_reach. you don't need out-center, blind and test trials
    df = df.reset_index()
    df = df[df['comeback'] == 0]
    # df = df[df['isBlind'] == 0]
    df = df[~df['block'].isin([1])]

    # need index for cycling through subjects
    df = df.set_index(["Subject"])

    # prepare vaf list for all subjects
    vaf_list = [[] for i in range(len(names))]
    vaf_test_list = [[] for i in range(len(names))]

    # import pca and autoencoder from ae_package
    AE = autoencoder2_0.Autoencoder(1000, 0.02, 2, struc="non_linear", nh1=8)
    PCA = pca.PrincipalComponentAnalysis(2)
    col_imu = ['imu00', 'imu01', 'imu02', 'imu03', 'imu10', 'imu11', 'imu12', 'imu13']

    for sub in range(len(names)):
        vaf = []
        vaf_test = []

        # get imu df and array for current subject
        df_sub = df.loc[idx[sub+1], :]
        df_sub_imu = df.loc[idx[sub + 1], col_imu]
        imu = df_sub_imu.values

        # compute vaf for baseline
        imu = imu[0:3000, :]
        imu_rec_pca = PCA.train_pca(imu)
        # vaf.append(useful_functions.compute_vaf(imu, imu_rec_pca) * 100)

        # remove baseline
        df_sub = df_sub.loc[idx[sub + 1]]
        df_sub = df_sub.iloc[3000:]

        # start from first trial after baseline
        # excluding first 8 test tgts and first of training (added right away in the for loop)
        count_trial = df_sub.loc[idx[sub + 1], 'trial'].values[0] - 9
        start_trial = df_sub.loc[idx[sub + 1], 'trial'].values[0]

        # reset index, take only trial
        df_sub = df_sub.reset_index()
        df_sub = df_sub.set_index(["trial"])

        # append data of x trials and compute pca/vaf on those x trials
        for trial in range(start_trial, trial_list[sub] + 1):
            # exclude test epochs and blind trials
            if [1, 6, 11] not in df_sub.loc[idx[trial], ['block']].values and \
                    1 not in df_sub.loc[idx[trial], ['isBlind']].values:
                count_trial += 1
                imu = df_sub.loc[idx[trial], col_imu].values
                if count_trial == 1 or trial == start_trial:
                    imu_tot = imu
                else:
                    if count_trial != trials_pca:
                        # append IMU data
                        imu_tot = np.append(imu_tot, imu, axis=0)
                    else:
                        imu_rec_pca = PCA.train_pca(imu_tot)
                        # _, _, imu_rec_pca, _ = AE.train_network(imu_tot)
                        vaf.append(useful_functions.compute_vaf(imu_tot, imu_rec_pca) * 100)
                        count_trial = 0

        # append nan for those who miss some trials (+1 is for baseline)
        while len(vaf) != (192 / trials_pca):
            vaf.append(np.nan)

        # create list of lists for all subjects
        vaf = np.array(vaf)
        vaf = np.reshape(vaf, (vaf.shape[0],))
        vaf = vaf.astype(float)
        vaf_list[sub] = vaf

        # vaf_test = np.array(vaf_test)
        # vaf_test = np.reshape(vaf_test, (vaf_test.shape[0],))
        # vaf_test = vaf_test.astype(float)
        # vaf_test_list[sub] = vaf_test

    return vaf_list


def plot_cursor_similarity(names, df_reach, df_w1, df_b1, df_w2, df_b2, idx, s1, s2):
    col_imu = ['imu00', 'imu01', 'imu02', 'imu03', 'imu10', 'imu11', 'imu12', 'imu13']
    col_w1 = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8']
    col_w2 = ['w1', 'w2']
    col_b1 = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8']
    col_b2 = ['b1', 'b2']

    # take only center out reaching task data of final test
    df_end = df_reach.loc[df_reach['block'] == 11]
    df_end = df_end.loc[df_end['comeback'] == 0]

    # then compute neuron activation vectors (2nd layer) for final batch
    cu_s2s1 = [[] for i in range(8)]
    cu_s2 = [[] for i in range(8)]

    # take final forward map for each participant
    df_w1_end = df_w1.groupby(level='Subject').tail(8)
    df_w2_end = df_w2.groupby(level='Subject').tail(8)
    df_b1_end = df_b1.groupby(level='Subject').tail(1)
    df_b2_end = df_b2.groupby(level='Subject').tail(1)

    # compare final cursor trajectories of S1 and S2
    data_end_s1 = df_end.loc[idx[s1], col_imu].values
    data_end_s2 = df_end.loc[idx[s2], :]
    w1_s1 = df_w1_end.loc[idx[s1, :], col_w1].values
    b1_s1 = df_b1_end.loc[idx[s1, :], col_b1].values
    w2_s1 = df_w2_end.loc[idx[s1, :], col_w2].values
    b2_s1 = df_b2_end.loc[idx[s1, :], col_b2].values
    w1_s2 = df_w1_end.loc[idx[s2, :], col_w1].values
    b1_s2 = df_b1_end.loc[idx[s2, :], col_b1].values
    w2_s2 = df_w2_end.loc[idx[s2, :], col_w2].values
    b2_s2 = df_b2_end.loc[idx[s2, :], col_b2].values

    for trial in range(8):
        # cursor coordinates of S2 with its own map
        data_trial_s2 = data_end_s2.loc[data_end_s2['trial'] == 241 + trial]
        cu_s2[trial] = data_trial_s2[['cursor_x', 'cursor_y']].values
        data_trial_s2 = data_trial_s2[col_imu]
        # h1_s2 = np.tanh(np.dot(data_trial_s2, w1_s2) + b1_s2)
        # cu_s2[trial] = np.dot(h1_s2, w2_s2) + b2_s2
        # cu_s2[trial] = cu_s2[trial] * np.array(([722*1.3, 331*2.2]))
        # cu_s2[trial] = cu_s2[trial] + np.array(([100 + data_end_s2['offset_x'].values[0], 400 + data_end_s2['offset_y'].values[0]]))
        # cursor coordinates of S2 with S1 map
        h1_s2s1 = np.tanh(np.dot(data_trial_s2, w1_s1) + b1_s1)
        cu_s2s1[trial] = np.dot(h1_s2s1, w2_s1) + b2_s1
        cu_s2s1[trial] = cu_s2s1[trial] * np.array(([722*1.3, 331*2.2]))
        cu_s2s1[trial] = cu_s2s1[trial] + np.array(([100 + data_end_s2['offset_x'].values[0], 400 + data_end_s2['offset_y'].values[0]]))

    # center cu_s2_s1 into cu_s2
    init = cu_s2s1[3][0, :]
    ref = [1920/2, 1080/2]
    off_s2s1 = ref - init
    for trial in range(8):
        cu_s2s1[trial] = cu_s2s1[trial] + np.array((off_s2s1))

    R = 378
    width = 1920
    height = 1080
    r = 45

    # theta goes from 0 to 2pi
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.figure()
    for trial in range(8):
        plt.plot(cu_s2[trial][:, 0], cu_s2[trial][:, 1], color='black', label='original')
        plt.plot(cu_s2s1[trial][:, 0], cu_s2s1[trial][:, 1], color='red', label='with S' + str(s1) +' map')
        tgt_x_test = (width / 2) + R * np.cos((2 * trial * np.pi / 8) + np.pi / 8)
        tgt_y_test = (height / 2) + R * np.sin((2 * trial * np.pi / 8) + np.pi / 8)
        plt.plot(r*np.cos(theta) + tgt_x_test, r*np.sin(theta) + tgt_y_test, color='b')
        plt.plot(r*np.cos(theta) + 1920/2, r*np.sin(theta) + 1080/2, color='g')
        plt.axis([1, 1920, 1, 1080])
        if trial == 0:
            plt.legend()
        plt.axis('equal')
        plt.title('S' + str(s2) + ' cursor coordinates during final test')
    plt.show()


def plot_ee_training(pdf, names, names_static, metric_rep, metric_rep_static):
    """
        plot endpoint error between coadaptive and static during training blocks
        :param names: list with subjects' names for coadaptive
        :param names_static: list with subjects' names for static
        :param ee_rep: list of lists containing ee values for each subject averaged per repetition - coadaptive
        :param ee_rep_static: list of lists containing ee values for each subject averaged per repetition - static
    """
    metric1 = []
    metric2 = []
    metric3 = []
    metric4 = []
    metric5 = []
    metric6 = []
    metric7 = []
    metric8 = []
    metric1_static = []
    metric2_static = []
    metric3_static = []
    metric4_static = []
    metric5_static = []
    metric6_static = []
    metric7_static = []
    metric8_static = []

    for sub in range(len(names)):
        metric1.append(np.nanmean(metric_rep[sub][0]))
        metric2.append(np.nanmean(metric_rep[sub][1]))
        metric3.append(np.nanmean(metric_rep[sub][2]))
        metric4.append(np.nanmean(metric_rep[sub][3]))
        metric5.append(np.nanmean(metric_rep[sub][4]))
        metric6.append(np.nanmean(metric_rep[sub][5]))
        metric7.append(np.nanmean(metric_rep[sub][6]))
        metric8.append(np.nanmean(metric_rep[sub][7]))
        # if sub != 1:
        #     metric_end.append(np.nanmean(metric_rep[sub][-6:]))
        # # sub 2 does not have block 8. using block 7 instead (rep from 36 to 41)
        # else:
        #     metric_end.append(np.nanmean(metric_rep[sub][36:42]))
    metric = np.array([metric1, metric2, metric3, metric4, metric5, metric6, metric7, metric8])

    for sub in range(len(names_static)):
        metric1_static.append(np.nanmean(metric_rep_static[sub][0]))
        metric2_static.append(np.nanmean(metric_rep_static[sub][1]))
        metric3_static.append(np.nanmean(metric_rep_static[sub][2]))
        metric4_static.append(np.nanmean(metric_rep_static[sub][3]))
        metric5_static.append(np.nanmean(metric_rep_static[sub][4]))
        metric6_static.append(np.nanmean(metric_rep_static[sub][5]))
        metric7_static.append(np.nanmean(metric_rep_static[sub][6]))
        metric8_static.append(np.nanmean(metric_rep_static[sub][7]))

    metric_static = np.array([metric1_static, metric2_static, metric3_static, metric4_static,
                              metric5_static, metric6_static, metric7_static, metric8_static])

    y_ad = np.array([np.nanmean(metric1), np.nanmean(metric2), np.nanmean(metric3), np.nanmean(metric4)
                        , np.nanmean(metric5), np.nanmean(metric6), np.nanmean(metric7), np.nanmean(metric8)])
    e_ad = np.array([1.96 * np.nanstd(metric1) / np.sqrt(len(names)),
                     1.96 * np.nanstd(metric2) / np.sqrt(len(names)),
                     1.96 * np.nanstd(metric3) / np.sqrt(len(names)),
                     1.96 * np.nanstd(metric4) / np.sqrt(len(names)),
                     1.96 * np.nanstd(metric5) / np.sqrt(len(names)),
                     1.96 * np.nanstd(metric6) / np.sqrt(len(names)),
                     1.96 * np.nanstd(metric7) / np.sqrt(len(names)),
                     1.96 * np.nanstd(metric8) / np.sqrt(len(names))])

    y_static = np.array(
        [np.nanmean(metric1_static), np.nanmean(metric2_static), np.nanmean(metric3_static), np.nanmean(metric4_static)
            , np.nanmean(metric5_static), np.nanmean(metric6_static), np.nanmean(metric7_static),
         np.nanmean(metric8_static)])
    e_static = np.array([1.96 * np.nanstd(metric1_static) / np.sqrt(len(names_static)),
                         1.96 * np.nanstd(metric2_static) / np.sqrt(len(names_static)),
                         1.96 * np.nanstd(metric3_static) / np.sqrt(len(names_static)),
                         1.96 * np.nanstd(metric4_static) / np.sqrt(len(names_static)),
                         1.96 * np.nanstd(metric5_static) / np.sqrt(len(names_static)),
                         1.96 * np.nanstd(metric6_static) / np.sqrt(len(names_static)),
                         1.96 * np.nanstd(metric7_static) / np.sqrt(len(names_static)),
                         1.96 * np.nanstd(metric8_static) / np.sqrt(len(names_static))])
    fig_handle = plt.figure()
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    linestyle = {"linestyle": "-", "linewidth": 2, "markeredgewidth": 2, "elinewidth": 2, "capsize": 2}
    plt.errorbar(x, y_ad, yerr=e_ad, color='r', label='adaptive (A)', **linestyle)
    plt.errorbar(x, y_static, yerr=e_static, color='k', label='static (S)', **linestyle)
    plt.legend(loc='upper left')
    plt.legend(fontsize='large')
    plt.xlabel('Training block', fontsize=14)
    # plt.ylabel('Reaching time [s]', fontsize=14)
    plt.ylabel('Endpoint error [cm]', fontsize=14)
    plt.title('Performance trend during Training blocks', fontsize=18)
    ax = plt.gca()
    ax.axis([0, 9, 2, 20])
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    # ax.set_xticklabels(['train 1', 'train 2', 'train 3', 'train 4', 'train 5', 'train 6', 'train 7', 'train 8'],
    #                    fontsize=10)
    ax.set_yticks([2, 8, 14, 20])
    ax.set_yticklabels([2, 8, 14, 20], fontsize=10)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    plt.show()

    return metric, metric_static


def plot_metrics_test(pdf, names, names_static, metric_test_rep, metric_test_rep_static, y_label):
    """
        plot performance between coadaptive and static during test blocks
        :param names: list with subjects' names for coadaptive
        :param names_static: list with subjects' names for static
        :param metric_test_rep: list of lists containing metric values for each subject averaged per repetition - coadaptive
        :param metric_test_rep_static: list of lists containing metric values for each subject averaged per repetition - static
        :param y_label: y label for plot containing name of the metric
    """
    # plot initial vs final test performance for both groups - Test
    metric_init_test = []
    metric_mid_test = []
    metric_end_test = []
    for sub in range(len(names)):
        metric_init_test.append(metric_test_rep[sub][0])
        metric_mid_test.append(metric_test_rep[sub][1])
        # if sub != 1:
        metric_end_test.append(metric_test_rep[sub][2])

    metric_test = np.array([metric_init_test, metric_mid_test, metric_end_test])

    metric_init_static_test = []
    metric_mid_static_test = []
    metric_end_static_test = []
    for sub in range(len(names_static)):
        metric_init_static_test.append(metric_test_rep_static[sub][0])
        metric_mid_static_test.append(metric_test_rep_static[sub][1])
        metric_end_static_test.append(metric_test_rep_static[sub][2])

    metric_static_test = np.array([metric_init_static_test, metric_mid_static_test, metric_end_static_test])

    y_ad_test = np.array([np.nanmean(metric_init_test), np.nanmean(metric_mid_test), np.nanmean(metric_end_test)])
    e_ad_test = np.array(
        [1.96 * np.nanstd(metric_init_test) / np.sqrt(len(names)), 1.96 * np.nanstd(metric_mid_test) / np.sqrt(len(names)),
         1.96 * np.nanstd(metric_end_test) / np.sqrt(len(names))])
    y_static_test = np.array([np.nanmean(metric_init_static_test), np.nanmean(metric_mid_static_test), np.nanmean(metric_end_static_test)])
    e_static_test = np.array([1.96 * np.nanstd(metric_init_static_test) / np.sqrt(len(names_static)),
                            1.96 * np.nanstd(metric_mid_static_test) / np.sqrt(len(names_static)),
                            1.96 * np.nanstd(metric_end_test) / np.sqrt(len(names))])

    fig_handle = plt.figure()
    x = np.array([0.7, 1, 1.3])
    linestyle = {"linestyle": "-", "linewidth": 2, "markeredgewidth": 2, "elinewidth": 2, "capsize": 2}
    plt.errorbar(x, y_ad_test, yerr=e_ad_test, color='r', label='adaptive (A)', **linestyle)
    plt.errorbar(x, y_static_test, yerr=e_static_test, color='k', label='static (S)', **linestyle)
    plt.legend(loc='upper left')
    plt.legend(fontsize='large')
    plt.xlabel('Time', fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title('Performance trend during Test', fontsize=18)
    ax = plt.gca()
    ax.axis([0.2, 1.8, 0, 20])
    ax.set_xticks([0.7, 1, 1.3])
    ax.set_xticklabels(['start', 'mid', 'end'], fontsize=12)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_yticklabels([0, 5, 10, 15, 20], fontsize=12)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    # pdf.savefig(fig_handle, dpi=300, papertype='letter', orientation='landscape', pad_inches=0)
    # plt.close()

    plt.show()

    return metric_test, metric_static_test


def plot_metrics_training(pdf, names, names_static, metric_rep, metric_rep_static, y_label):
    """
        plot performance between coadaptive and static during training blocks
        :param names: list with subjects' names for coadaptive
        :param names_static: list with subjects' names for static
        :param metric_rep: list of lists containing metric values for each subject averaged per repetition - coadaptive
        :param metric_rep_static: list of lists containing metric values for each subject averaged per repetition - static
        :param y_label: y label for plot containing name of the metric
    """
    metric1 = []
    metric2 = []
    metric3 = []
    metric4 = []
    metric5 = []
    metric6 = []
    metric7 = []
    metric8 = []
    metric1_static = []
    metric2_static = []
    metric3_static = []
    metric4_static = []
    metric5_static = []
    metric6_static = []
    metric7_static = []
    metric8_static = []

    for sub in range(len(names)):
        metric1.append(np.nanmean(metric_rep[sub][0:6]))
        metric2.append(np.nanmean(metric_rep[sub][6:12]))
        metric3.append(np.nanmean(metric_rep[sub][12:18]))
        metric4.append(np.nanmean(metric_rep[sub][18:24]))
        metric5.append(np.nanmean(metric_rep[sub][24:30]))
        metric6.append(np.nanmean(metric_rep[sub][30:36]))
        metric7.append(np.nanmean(metric_rep[sub][36:42]))
        metric8.append(np.nanmean(metric_rep[sub][42:48]))
        # if sub != 1:
        #     metric_end.append(np.nanmean(metric_rep[sub][-6:]))
        # # sub 2 does not have block 8. using block 7 instead (rep from 36 to 41)
        # else:
        #     metric_end.append(np.nanmean(metric_rep[sub][36:42]))
    metric = np.array([metric1, metric2, metric3, metric4, metric5, metric6, metric7, metric8])

    for sub in range(len(names_static)):
        metric1_static.append(np.nanmean(metric_rep_static[sub][0:6]))
        metric2_static.append(np.nanmean(metric_rep_static[sub][6:12]))
        metric3_static.append(np.nanmean(metric_rep_static[sub][12:18]))
        metric4_static.append(np.nanmean(metric_rep_static[sub][18:24]))
        metric5_static.append(np.nanmean(metric_rep_static[sub][24:30]))
        metric6_static.append(np.nanmean(metric_rep_static[sub][30:36]))
        metric7_static.append(np.nanmean(metric_rep_static[sub][36:42]))
        metric8_static.append(np.nanmean(metric_rep_static[sub][42:48]))

    metric_static = np.array([metric1_static, metric2_static, metric3_static, metric4_static,
                          metric5_static, metric6_static, metric7_static, metric8_static])

    y_ad = np.array([np.nanmean(metric1), np.nanmean(metric2), np.nanmean(metric3), np.nanmean(metric4)
                     , np.nanmean(metric5), np.nanmean(metric6), np.nanmean(metric7), np.nanmean(metric8)])
    e_ad = np.array([1.96 * np.nanstd(metric1) / np.sqrt(len(names)),
                     1.96 * np.nanstd(metric2) / np.sqrt(len(names)),
                     1.96 * np.nanstd(metric3) / np.sqrt(len(names)),
                     1.96 * np.nanstd(metric4) / np.sqrt(len(names)),
                     1.96 * np.nanstd(metric5) / np.sqrt(len(names)),
                     1.96 * np.nanstd(metric6) / np.sqrt(len(names)),
                     1.96 * np.nanstd(metric7) / np.sqrt(len(names)),
                     1.96 * np.nanstd(metric8) / np.sqrt(len(names))])

    y_static = np.array([np.nanmean(metric1_static), np.nanmean(metric2_static), np.nanmean(metric3_static), np.nanmean(metric4_static)
                     , np.nanmean(metric5_static), np.nanmean(metric6_static), np.nanmean(metric7_static), np.nanmean(metric8_static)])
    e_static = np.array([1.96 * np.nanstd(metric1_static) / np.sqrt(len(names_static)),
                     1.96 * np.nanstd(metric2_static) / np.sqrt(len(names_static)),
                     1.96 * np.nanstd(metric3_static) / np.sqrt(len(names_static)),
                     1.96 * np.nanstd(metric4_static) / np.sqrt(len(names_static)),
                     1.96 * np.nanstd(metric5_static) / np.sqrt(len(names_static)),
                     1.96 * np.nanstd(metric6_static) / np.sqrt(len(names_static)),
                     1.96 * np.nanstd(metric7_static) / np.sqrt(len(names_static)),
                     1.96 * np.nanstd(metric8_static) / np.sqrt(len(names_static))])
    fig_handle = plt.figure()
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    linestyle = {"linestyle": "-", "linewidth": 2, "markeredgewidth": 2, "elinewidth": 2, "capsize": 2}
    plt.errorbar(x, y_ad, yerr=e_ad, color='r', label='adaptive (A)', **linestyle)
    plt.errorbar(x, y_static, yerr=e_static, color='k', label='static (S)', **linestyle)
    plt.legend(loc='upper left')
    plt.legend(fontsize='large')
    plt.xlabel('Training block', fontsize=14)
    # plt.ylabel('Reaching time [s]', fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title('Performance trend during Training blocks', fontsize=18)
    ax = plt.gca()
    ax.axis([0, 9, 0, 15])
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    # ax.set_xticklabels(['train 1', 'train 2', 'train 3', 'train 4', 'train 5', 'train 6', 'train 7', 'train 8'],
    #                    fontsize=10)
    ax.set_yticks([0, 5, 10, 15])
    ax.set_yticklabels([0, 5, 10, 15], fontsize=10)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    # pdf.savefig(fig_handle, dpi=300, papertype='letter', orientation='landscape', pad_inches=0)
    # plt.close()

    plt.show()

    return metric, metric_static


def plot_metrics_trial(typ, pdf, df_b2, names, metric_train, y_label, idx):
    """
    Function to plot metric throughout training trials
    :param typ: string, either adaptive or static to plot in title
    :param df_b2: dataframe containing loss values to plot
    :param names: list with subjects' names
    :param metric_train: list of list containing metric values during training er each subject
    :param y_label: y label for plot containing name of the metric
    :param idx: indexslice for handling dataframes
    """

    f, ax = plt.subplots(2, 5, sharex=True, sharey=True)
    # f, ax = plt.subplots(1, len(names))
    f.suptitle('Performance with single exp fit for ' + typ + ' group', fontsize=14)
    axx = 0  # index to iterate rows of subplot
    axy = 0  # index to iterate cols of subplot

    r_sq = []
    tau = []

    for sub in range(len(names)):
        # handle index of subplot
        if sub == 5:
            axx = 1
            axy = 0

        # prepare loss array
        loss = df_b2.loc[idx[sub+1, :], ['loss']].values
        # resample loss to match metric length. This is needed to compute correlation and plot
        loss = resample_brutal(loss, metric_train[sub].shape[0], 50)

        # exponential fit on metric
        x = np.arange(metric_train[sub].shape[0])
        y = metric_train[sub]
        id = np.isfinite(x) & np.isfinite(y)
        popt, pcov = curve_fit(func, x[id], y[id], p0=[max(y), 1 / 5, min(y)], maxfev=2000)
        tau.append(1 / popt[1])
        est = func(x, popt[0], popt[1], popt[2])
        res = y - est
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_sq.append(1 - (ss_res / ss_tot))

        # plot everything
        ax[axx, axy].plot(loss * (max(metric_train[sub])/max(loss)), color='orange', label='loss J')
        ax[axx, axy].plot(est, color='green', label='fit')
        ax[axx, axy].scatter(x, y, color='black', s=5 * 2 ** 1e-4)
        if sub == 7:
            ax[axx, axy].set_xlabel('trials', fontsize=10)
        if sub == len(names) - 1:
            ax[axx, axy].legend(frameon=False, fontsize=9, loc='upper right')
        elif sub == 0:
            ax[axx, axy].set_ylabel(y_label, fontsize=12)
        ax[axx, axy].set_title('Subject ' + str(sub + 1), fontsize=10)
        ax[axx, axy].spines['right'].set_color('none')
        ax[axx, axy].spines['top'].set_color('none')
        # ax[sub].set_yticks([0, 10, 20, 30, 40])
        ax[axx, axy].set_ylim([0, np.nanmax(np.nanmax(metric_train))])
        ax[axx, axy].set_xlim([-10, 193])
        # set fontsize for xticks and yticks
        for tick in ax[axx, axy].xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        for tick in ax[axx, axy].yaxis.get_major_ticks():
            tick.label.set_fontsize(8)

        axy += 1

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    pdf.savefig(f, dpi=300, papertype='letter', orientation='landscape', pad_inches=0)
    plt.close()

    # plt.show()


# def plot_rt_training(names, names_static, rt_rep, rt_rep_static):
#     """
#         plot performance between coadaptive and static during training blocks
#         :param names: list with subjects' names for coadaptive
#         :param names_static: list with subjects' names for static
#         :param rt_rep: list of lists containing rt values for each subject averaged per repetition - coadaptive
#         :param rt_rep_static: list of lists containing rt values for each subject averaged per repetition - static
#     """
#     rt_base = []
#     rt_init = []
#     rt_end = []
#     for sub in range(len(names)):
#         rt_base.append(np.nanmean(rt_rep[sub][0:3]))
#         rt_init.append(np.nanmean(rt_rep[sub][3:8]))
#         if sub != 1:
#             rt_end.append(np.nanmean(rt_rep[sub][-6:]))
#         # sub 2 does not have block 8. using block 7 instead (rep from 36 to 41)
#         else:
#             rt_end.append(np.nanmean(rt_rep[sub][36:42]))
#
#     rt = np.array([rt_base, rt_init, rt_end])
#
#     rt_base_static = []
#     rt_init_static = []
#     rt_end_static = []
#     for sub in range(len(names_static)):
#         rt_base_static.append(np.nanmean(rt_rep_static[sub][0:3]))
#         rt_init_static.append(np.nanmean(rt_rep_static[sub][3:8]))
#         # rt_end_static.append(np.nanmean(rt_rep_static[sub][-8:-1]))
#         rt_end_static.append(np.nanmean(rt_rep_static[sub][-6:]))
#
#     rt_static = np.array([rt_base_static, rt_init_static, rt_end_static])
#
#     y_ad = np.array([np.nanmean(rt_base), np.nanmean(rt_init), np.nanmean(rt_end)])
#     e_ad = np.array([1.96 * np.nanstd(rt_base) / np.sqrt(len(names)),
#                      1.96 * np.nanstd(rt_init) / np.sqrt(len(names)),
#                      1.96 * np.nanstd(rt_end) / np.sqrt(len(names))])
#     # y_ad = np.array([np.nanmean(rt_init) / np.nanmean(rt_test[0, :]),
#     #                  np.nanmean(rt_end) / np.nanmean(rt_test[0, :])])
#     # e_ad = np.array(
#     #     [1.96 * np.nanstd(np.divide(rt_init, rt_test[0, :])) / np.sqrt(len(names)),
#     #      1.96 * np.nanstd(np.divide(rt_end, rt_test[0, :])) / np.sqrt(len(names))])
#
#     # y_static = np.array([np.nanmean(rt_base_static), np.nanmean(rt_init_static), np.nanmean(rt_end_static)])
#     y_static = np.array([np.nanmean(rt_base_static), np.nanmean(rt_init_static), np.nanmean(rt_end_static)])
#     e_static = np.array([1.96 * np.nanstd(rt_base_static) / np.sqrt(len(names_static)),
#                         1.96 * np.nanstd(rt_init_static) / np.sqrt(len(names_static)),
#                         1.96 * np.nanstd(rt_end_static) / np.sqrt(len(names_static))])
#     # y_static = np.array([np.nanmean(rt_init_static) / np.nanmean(rt_test_static[0, :]),
#     #                      np.nanmean(rt_end_static) / np.nanmean(rt_test_static[0, :])])
#     # e_static = np.array(
#     #     [1.96 * np.nanstd(np.divide(rt_init_static, rt_test_static[0, :])) / np.sqrt(len(names_static)),
#     #      1.96 * np.nanstd(np.divide(rt_end_static, rt_test_static[0, :])) / np.sqrt(len(names_static))])
#
#     plt.figure()
#     x = np.array([0.75, 0.95, 1.3])
#     linestyle = {"linestyle": "-", "linewidth": 2, "markeredgewidth": 2, "elinewidth": 2, "capsize": 2}
#     plt.errorbar(x, y_ad, yerr=e_ad, color='r', label='adaptive (A)', **linestyle)
#     plt.errorbar(x, y_static, yerr=e_static, color='k', label='static (S)', **linestyle)
#     plt.legend(loc='upper left')
#     plt.legend(fontsize='large')
#     plt.xlabel('Time', fontsize=14)
#     # plt.ylabel('Reaching time [s]', fontsize=14)
#     plt.ylabel('Reaching time [s]', fontsize=14)
#     plt.title('Performance trend during Training blocks', fontsize=18)
#     ax = plt.gca()
#     ax.axis([0.15, 1.8, 0, 20])
#     ax.set_xticks([0.65, 0.85, 1.3])
#     ax.set_xticklabels(['baseline', 'start', 'end'], fontsize=12)
#     ax.set_yticks([0, 5, 10, 15, 20])
#     ax.set_yticklabels([0, 5, 10, 15, 20], fontsize=12)
#     ax.spines['right'].set_color('none')
#     ax.spines['top'].set_color('none')
#
#     plt.show()
#
#     t, p = stats.ttest_ind(rt_init, rt_end)
#
#     return rt, rt_static


def plot_pdf(pdf, names, batch_list, df_w1, df_b1, df_b2, encoder_param, idx):
    """
    Function that plots in a PDF all the weights of the encoder for each subjects. Takes a while to run
    """

    for sub in range(len(names)):

        # PLOT WEIGHTS LAYER 0 - 4grids 4x4
        row = [0, 4, 4, 8, 0, 4, 4, 8]
        col = [0, 4, 0, 4, 4, 8, 4, 8]
        tmp_mat = 0
        for m in range(4):
            fig_handle = plt.figure()
            gs = fig_handle.add_gridspec(4, 4)
            axes = []
            tmp = 0
            tmp_row = 0
            for r in range(row[tmp_mat], row[tmp_mat+1]):
                tmp_col = 0
                for c in range(col[tmp_mat], col[tmp_mat+1]):
                    axes.append(fig_handle.add_subplot(gs[tmp_row, tmp_col]))
                    for batch in range(1, batch_list[sub]):
                        axes[tmp].plot(batch, df_w1.loc[idx[sub+1, batch], :].values[r, c], color='b', marker='o', markersize=1)
                    axes[tmp].set_ylabel('w' + str(r) + str(c))
                    axes[tmp].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    tmp += 1
                    tmp_col += 1
                tmp_row += 1
            tmp_mat += 2

            fig_handle.text(0.35, 0.95, 'Subject ' + str(sub+1) + ', 1st layer - weights during coadaptation',
                            fontsize='x-large', ha='center', va='center')
            fig_handle.text(0.3, 0.04, 'batch', fontsize='large', va='center')
            fig_handle.text(0.7, 0.04, 'batch', fontsize='large', va='center')
            fig_handle.tight_layout(rect=[0, 0.03, 1, 0.95])
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            # save figure as new page in pdf:
            pdf.savefig(fig_handle, dpi=300, papertype='letter', orientation='landscape', pad_inches=0)
            plt.close()

        # PLOT WEIGHTS LAYER 0 - grid 2x4 (8 per grid)
        num_col = 8
        cm = plt.get_cmap('gist_rainbow')

        fig_handle = plt.figure()
        gs = fig_handle.add_gridspec(2, 4)
        axes = []
        tmp = 0
        for r in range(2):
            for c in range(4):
                tmp_row = 0
                axes.append(fig_handle.add_subplot(gs[r, c]))
                for iw in range(8):
                    for batch in range(1, batch_list[sub]):
                        axes[tmp].plot(batch, df_w1.loc[idx[sub+1, batch], :].values[tmp_row, tmp],
                                       c=cm(1. * (iw+1) / num_col), marker='o', markersize=1)
                    tmp_row += 1
                axes[tmp].set_ylabel('input ' + str(tmp+1))
                axes[tmp].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                tmp += 1

        fig_handle.text(0.35, 0.95, 'Subject ' + str(sub+1) + ', input weights in 1st layer',
                        fontsize='x-large', ha='center', va='center')
        fig_handle.text(0.3, 0.04, 'batch', fontsize='large', va='center')
        fig_handle.text(0.7, 0.04, 'batch', fontsize='large', va='center')
        fig_handle.tight_layout(rect=[0, 0.03, 1, 0.95])
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        # save figure as new page in pdf:
        pdf.savefig(fig_handle, dpi=300, papertype='letter', orientation='landscape', pad_inches=0)
        plt.close()

        # PLOT WEIGHTS LAYER 1
        fig_handle = plt.figure()
        gs = fig_handle.add_gridspec(2, 8)
        axes = []
        tmp = 0
        for r in range(8):
            for c in range(2):
                axes.append(fig_handle.add_subplot(gs[c, r]))
                for batch in range(1, batch_list[sub]):
                    axes[tmp].plot(batch, df_w1.loc[idx[sub+1, batch], :].values[r, c], color='b', marker='o', markersize=1)
                axes[tmp].tick_params(length=0)
                axes[tmp].set_ylabel('w' + str(r) + str(c))
                axes[tmp].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                tmp += 1

        fig_handle.text(0.35, 0.95, 'Subject ' + str(sub+1) + ', 2nd layer - weights during coadaptation',
                        fontsize='x-large', ha='center', va='center')
        fig_handle.text(0.3, 0.04, 'batch', fontsize='large', va='center')
        fig_handle.text(0.7, 0.04, 'batch', fontsize='large', va='center')
        fig_handle.tight_layout()
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        # save figure as new page in pdf:
        pdf.savefig(fig_handle, dpi=300, papertype='letter', orientation='landscape', pad_inches=0)
        plt.close()

        # PLOT BIASES & LOSS
        fig_handle = plt.figure()
        gs = fig_handle.add_gridspec(4, 4)
        axes = []
        axes.append(fig_handle.add_subplot(gs[0, 0]))
        axes.append(fig_handle.add_subplot(gs[0, 1]))
        axes.append(fig_handle.add_subplot(gs[1, 0]))
        axes.append(fig_handle.add_subplot(gs[1, 1]))
        axes.append(fig_handle.add_subplot(gs[2, 0]))
        axes.append(fig_handle.add_subplot(gs[2, 1]))
        axes.append(fig_handle.add_subplot(gs[3, 0]))
        axes.append(fig_handle.add_subplot(gs[3, 1]))
        axes.append(fig_handle.add_subplot(gs[0:2, 2]))
        axes.append(fig_handle.add_subplot(gs[0:2, -1]))
        axes.append(fig_handle.add_subplot(gs[2:, 2:]))
        for c in range(8):
            axes[c].plot(df_b1.loc[idx[sub+1, :], ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8']].values[:, c])
            axes[c].set_ylabel('b0' + str(c))
            axes[c].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axes[8].plot(df_b2.loc[idx[sub+1, :], ['b1', 'b2']].values[:, 0])
        axes[8].set_ylabel('b10')
        axes[9].plot(df_b2.loc[idx[sub+1, :], ['b1', 'b2']].values[:, 1])
        axes[9].set_ylabel('b11')
        axes[10].plot(df_b2.loc[idx[sub+1, :], ['loss']].values)
        axes[10].set_ylabel('loss')
        fig_handle.text(0.35, 0.95, 'Subject ' + str(sub+1) + ', Biases and Loss during coadaptation',
                        fontsize='x-large', ha='center', va='center')
        fig_handle.text(0.3, 0.04, 'batch', fontsize='large', va='center')
        fig_handle.text(0.7, 0.04, 'batch', fontsize='large', va='center')
        fig_handle.tight_layout(rect=[0, 0.03, 1, 0.95])
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        # save figure as new page in pdf:
        pdf.savefig(fig_handle, dpi=300, papertype='letter', orientation='landscape', pad_inches=0)
        plt.close()

        # for trial in range(1, trial_list):
        #     if trial == 1:
        #         cu_mu = df_reach.loc[idx[trial], col_curs].mean().values
        #     else:
        #         cu_mu = np.vstack((cu_mu, df_reach.loc[idx[trial], col_curs].mean().values))
        #
        # plt.plot(cu_mu[:, 0])
        # plt.plot(cu_mu[:, 1])
        #
        # rt = []
        # # for block in range(1, 12):
        # for trial in range(1, trial_list[sub]+1):
        #     start = df_reach.loc[idx[sub+1, trial, 0], ['time']].values[0]
        #     end = df_reach.loc[idx[sub+1, trial, 0], ['time']].values[-1]
        #     rt.append((end - start)/1000)
        #
        # fig_handle = plt.figure()
        # # axes[10].plot(df_b2.loc[idx[:], ['loss']].values)
        # plt.scatter(np.arange(trial_list[sub]), rt)
        # plt.ylabel('Movement Time (s)')
        # plt.xlabel('Trial')
        # plt.title('Subject ' + str(sub+1))
        # manager = plt.get_current_fig_manager()
        # manager.window.showMaximized()
        # pdf.savefig(fig_handle, dpi=300, papertype='letter', orientation='landscape', pad_inches=0)
        # plt.close()

        # plot IMSHOW of all the parameters
        x_plot = np.diff(encoder_param[sub], axis=0)

        fig_handle = plt.figure()
        plt.imshow(x_plot.T)
        plt.title('Subject ' + str(sub + 1))
        plt.colorbar()
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        pdf.savefig(fig_handle, dpi=300, papertype='letter', orientation='landscape', pad_inches=0)
        plt.close()


def plot_tc_training(pdf, names, names_static, tc_trial, tc_trial_end, tc_trial_static, tc_trial_static_end):
    """
        plot performance (as trials completed - tc) between coadaptive and static during training blocks
        :param names: list with subjects' names for coadaptive
        :param names_static: list with subjects' names for static
        :param tc_trial: list of lists containing tc values for each subject - coadaptive init
        :param tc_trial_end: list of lists containing tc values for each subject - coadaptive end
        :param tc_trial_static: list of lists containing tc values for each subject - static init
        :param tc_trial_static_end: list of lists containing tc values for each subject  - static end
    """
    tc_base = []
    tc_init = []
    tc_mid = []
    tc_mid2 = []
    tc_end = []
    for sub in range(len(names)):
        tc_base.append(tc_trial[sub][0])
        tc_init.append(np.sum(tc_trial[sub][1:3]))
        tc_mid.append(np.sum(tc_trial[sub][3:5]))
        tc_mid2.append(np.sum(tc_trial[sub][5:7]))
        tc_end.append(np.sum(tc_trial_end[sub][0:2]))

    tc = np.array([tc_base, tc_init, tc_mid, tc_mid2, tc_end])

    tc_base_static = []
    tc_init_static = []
    tc_mid_static = []
    tc_mid2_static = []
    tc_end_static = []
    for sub in range(len(names_static)):
        tc_base_static.append(tc_trial_static[sub][0])
        tc_init_static.append(np.sum(tc_trial_static[sub][1:3]))
        tc_mid_static.append(np.sum(tc_trial_static[sub][3:5]))
        tc_mid2_static.append(np.sum(tc_trial_static[sub][5:7]))
        tc_end_static.append(np.sum(tc_trial_static_end[sub][0:2]))

    tc_static = np.array([tc_base_static, tc_init_static, tc_mid_static, tc_mid2_static, tc_end_static])

    y_ad = np.array([np.nanmean(tc_base), np.nanmean(tc_init), np.nanmean(tc_mid), np.nanmean(tc_mid2)])
    e_ad = np.array([1.96 * np.nanstd(tc_base) / np.sqrt(len(names)),
                     1.96 * np.nanstd(tc_init) / np.sqrt(len(names)),
                     1.96 * np.nanstd(tc_mid) / np.sqrt(len(names)),
                     1.96 * np.nanstd(tc_mid2) / np.sqrt(len(names))])

    y_static = np.array([np.nanmean(tc_base_static), np.nanmean(tc_init_static),
                         np.nanmean(tc_mid_static), np.nanmean(tc_mid2_static)])
    e_static = np.array([1.96 * np.nanstd(tc_base_static) / np.sqrt(len(names_static)),
                        1.96 * np.nanstd(tc_init_static) / np.sqrt(len(names_static)),
                        1.96 * np.nanstd(tc_mid_static) / np.sqrt(len(names_static)),
                        1.96 * np.nanstd(tc_mid2_static) / np.sqrt(len(names_static))])

    fig_handle = plt.figure()
    x = np.array([0.65, 0.85, 1.05, 1.25])
    linestyle = {"linestyle": "-", "linewidth": 2, "markeredgewidth": 2, "elinewidth": 2, "capsize": 2}
    plt.errorbar(x, y_ad, yerr=e_ad, color='r', label='adaptive (A)', **linestyle)
    plt.errorbar(1.7, np.nanmean(tc_end), yerr=1.96 * np.nanstd(tc_end) / np.sqrt(len(names)),
                 color='r', **linestyle)
    plt.errorbar(x, y_static, yerr=e_static, color='k', label='static (S)', **linestyle)
    plt.errorbar(1.7, np.nanmean(tc_end_static), yerr=1.96 * np.nanstd(tc_end_static) / np.sqrt(len(names_static)),
                 color='k', **linestyle)
    plt.legend(loc='upper left')
    plt.legend(fontsize='large')
    plt.xlabel('Time', fontsize=14)
    # plt.ylabel('Reaching time [s]', fontsize=14)
    plt.ylabel('Trials completed', fontsize=14)
    plt.title('Performance trend during Training blocks', fontsize=18)
    ax = plt.gca()
    ax.axis([0.15, 1.8, 0, 60])
    ax.set_xticks([0.65, 0.85, 1.05, 1.25, 1.7])
    ax.set_xticklabels(['baseline', '2min', '4min', '6min', 'last 2min'], fontsize=12)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_yticklabels([0, 20, 40, 60], fontsize=12)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    pdf.savefig(fig_handle, dpi=300, papertype='letter', orientation='landscape', pad_inches=0)
    plt.close()

    plt.show()

    t, p = stats.ttest_ind(tc_init, tc_end)

    return tc.T, tc_static.T


def plot_vaf_training(names, names_static, vaf_trial, vaf_trial_static):
    """
        plot performance (as trials completed - vaf) between coadaptive and static during training blocks
        :param names: list with subjects' names for coadaptive
        :param names_static: list with subjects' names for static
        :param vaf_trial: list of lists containing vaf values for each subject - coadaptive init
        :param vaf_trial_static: list of lists containing vaf values for each subject - static init
    """
    vaf_base = []
    vaf_init = []
    vaf_mid = []
    vaf_mid2 = []
    vaf_mid3 = []
    vaf_mid4 = []
    vaf_mid5 = []
    vaf_end = []
    for sub in range(len(names)):
        vaf_base.append(vaf_trial[sub][0])
        vaf_init.append(np.mean(vaf_trial[sub][1]))
        vaf_mid.append(np.mean(vaf_trial[sub][2]))
        vaf_mid2.append(np.mean(vaf_trial[sub][3]))
        vaf_mid3.append(np.mean(vaf_trial[sub][4]))
        vaf_mid4.append(np.mean(vaf_trial[sub][5]))
        vaf_mid5.append(np.mean(vaf_trial[sub][6]))
        vaf_end.append(np.mean(vaf_trial[sub][-1:]))

    vaf = np.array([vaf_base, vaf_init, vaf_mid, vaf_mid2, vaf_mid3, vaf_mid4, vaf_mid5, vaf_end])

    vaf_base_static = []
    vaf_init_static = []
    vaf_mid_static = []
    vaf_mid2_static = []
    vaf_mid3_static = []
    vaf_mid4_static = []
    vaf_mid5_static = []
    vaf_end_static = []

    for sub in range(len(names_static)):
        vaf_base_static.append(vaf_trial_static[sub][0])
        vaf_init_static.append(np.mean(vaf_trial_static[sub][1]))
        vaf_mid_static.append(np.mean(vaf_trial_static[sub][2]))
        vaf_mid2_static.append(np.mean(vaf_trial_static[sub][3]))
        vaf_mid3_static.append(np.mean(vaf_trial_static[sub][4]))
        vaf_mid4_static.append(np.mean(vaf_trial_static[sub][5]))
        vaf_mid5_static.append(np.mean(vaf_trial_static[sub][6]))
        vaf_end_static.append(np.mean(vaf_trial_static[sub][-1:]))

    vaf_static = np.array([vaf_base_static, vaf_init_static, vaf_mid_static, vaf_mid2_static,
                           vaf_mid3_static, vaf_mid4_static, vaf_mid5_static, vaf_end_static])

    y_ad = np.array([np.nanmean(vaf_base), np.nanmean(vaf_init), np.nanmean(vaf_mid), np.nanmean(vaf_mid2),
                     np.nanmean(vaf_mid3), np.nanmean(vaf_mid4), np.nanmean(vaf_mid5)])
    e_ad = np.array([1.96 * np.nanstd(vaf_base) / np.sqrt(len(names)),
                     1.96 * np.nanstd(vaf_init) / np.sqrt(len(names)),
                     1.96 * np.nanstd(vaf_mid) / np.sqrt(len(names)),
                     1.96 * np.nanstd(vaf_mid2) / np.sqrt(len(names)),
                     1.96 * np.nanstd(vaf_mid3) / np.sqrt(len(names)),
                     1.96 * np.nanstd(vaf_mid4) / np.sqrt(len(names)),
                     1.96 * np.nanstd(vaf_mid5) / np.sqrt(len(names))])

    y_static = np.array([np.nanmean(vaf_base_static), np.nanmean(vaf_init_static),
                         np.nanmean(vaf_mid_static), np.nanmean(vaf_mid2_static),
                         np.nanmean(vaf_mid3_static), np.nanmean(vaf_mid4_static), np.nanmean(vaf_mid5_static)])
    e_static = np.array([1.96 * np.nanstd(vaf_base_static) / np.sqrt(len(names_static)),
                        1.96 * np.nanstd(vaf_init_static) / np.sqrt(len(names_static)),
                        1.96 * np.nanstd(vaf_mid_static) / np.sqrt(len(names_static)),
                        1.96 * np.nanstd(vaf_mid2_static) / np.sqrt(len(names_static)),
                        1.96 * np.nanstd(vaf_mid3_static) / np.sqrt(len(names_static)),
                        1.96 * np.nanstd(vaf_mid4_static) / np.sqrt(len(names_static)),
                        1.96 * np.nanstd(vaf_mid5_static) / np.sqrt(len(names_static))])

    plt.figure()
    x = np.array([0.65, 0.85, 1.05, 1.25, 1.45, 1.65, 1.85])
    linestyle = {"linestyle": "-", "linewidth": 2, "markeredgewidth": 2, "elinewidth": 2, "capsize": 2}
    plt.errorbar(x, y_ad, yerr=e_ad, color='r', label='adaptive (A)', **linestyle)
    plt.errorbar(2.3, np.nanmean(vaf_end), yerr=1.96 * np.nanstd(vaf_end) / np.sqrt(len(names)),
                 color='r', **linestyle)
    plt.errorbar(x, y_static, yerr=e_static, color='k', label='static (S)', **linestyle)
    plt.errorbar(2.3, np.nanmean(vaf_end_static), yerr=1.96 * np.nanstd(vaf_end_static) / np.sqrt(len(names_static)),
                 color='k', **linestyle)
    plt.legend(loc='upper left')
    plt.legend(fontsize='large')
    plt.xlabel('Time', fontsize=14)
    # plt.ylabel('Reaching time [s]', fontsize=14)
    plt.ylabel('VAF', fontsize=14)
    plt.title('VAF trend during Training blocks', fontsize=18)
    ax = plt.gca()
    ax.axis([0.15, 2.7, 60, 100])
    ax.set_xticks([0.65, 0.85, 1.05, 1.25, 1.45, 1.65, 1.85, 2.3])
    # ax.set_xticklabels(['baseline', '2min', '4min', '6min', 'last 2min'], fontsize=12)
    ax.set_xticklabels(['b', '0-2', '2-4', '4-6', '6-8', '8-10', '10-12', 'last 2'], fontsize=12)
    ax.set_yticks([60, 70, 80, 90, 100])
    ax.set_yticklabels([60, 70, 80, 90, 100], fontsize=12)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    plt.show()

    t, p = stats.ttest_ind(vaf_init, vaf_end)

    return vaf.T, vaf_static.T


def plot_whole_session(typ, pdf, names, rt_trial, rt_trial_test):
    """
        This function plots rt for the whole session (both training and test)
        :param typ: define whether coAE or static
        :param names: list with subjects' names
        :param rt_trial: list of list with rt for each trial (training) and subject
        :param rt_trial_test: list of list with rt for each trial (test) and subject
        """
    f, ax = plt.subplots(2, 5, sharex=True, sharey=True)
    # f, ax = plt.subplots(1, len(names))
    f.suptitle('Performance during whole session for ' + typ + ' group', fontsize=14)

    axx = 0
    axy = 0

    for sub in range(len(names)):
        # handle index of subplot
        if sub == 5:
            axx = 1
            axy = 0

        # plot everything
        ax[axx, axy].scatter(np.arange(1, 9), rt_trial_test[sub][0:8], color='gold', label='Early test', s=2 * 2 ** 1e-4)
        ax[axx, axy].scatter(np.arange(9, 105), rt_trial[sub][0:96], color='dodgerblue', label='Early train', s=2 * 2 ** 1e-4)
        ax[axx, axy].scatter(np.arange(105, 113), rt_trial_test[sub][8:16], color='darkorange', label='Mid test', s=2 * 2 ** 1e-4)
        # if sub != 1 or typ == "static":
        ax[axx, axy].scatter(np.arange(113, 209), rt_trial[sub][96:], color='blue', label='Late train', s=2 * 2 ** 1e-4)
        ax[axx, axy].scatter(np.arange(209, 217), rt_trial_test[sub][16:], color='red', label='Late test', s=2 * 2 ** 1e-4)
        # elif typ == "coad":
        # ax[sub].scatter(np.arange(113, 185), rt_trial[sub][96:], color='blue', label='Late train', s=25 * 2 ** 1e-4)

        if sub == 7:
            ax[axx, axy].set_xlabel('repetitions', fontsize=10)
        if sub == len(names) - 1:
            ax[axx, axy].legend(frameon=False, fontsize=8)
        if sub == 0:
            ax[axx, axy].set_ylabel('[s]', fontsize=10)
        ax[axx, axy].set_title('Subject ' + str(sub + 1), fontsize=10)
        # get rid of the frame
        ax[axx, axy].spines['right'].set_color('none')
        ax[axx, axy].spines['top'].set_color('none')
        ax[axx, axy].set_yticks([0, 10, 20, 30])
        ax[axx, axy].set_ylim([0, 30])
        ax[axx, axy].set_xlim([-5, 218 + 5])
        # set fontsize for xticks and yticks
        for tick in ax[axx, axy].xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        for tick in ax[axx, axy].yaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        # ax[sub].text(len(x) * 60 / 100, 30 * 90 / 100, r'$R^2$: ' + str(np.ceil(r_sq[sub] * 100) / 100), fontsize=12)

        axy += 1

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    pdf.savefig(f, dpi=300, papertype='letter', orientation='landscape', pad_inches=0)
    plt.close()


def resample(original_time, original_data, new_sampling_freq=50.0):
    """
    resamples a signal given new sampling frequency, can handle 2D or 3D
    :param original_time: numpy array or dataframe series for the original time vector (in ms)
    :param original_data: Nx1 or Nx2 data sampled at times in original_time, a numpy array or dataframe
    :param new_sampling_freq: constant for the new sampling frequency in Hz, default is 100Hz
    :return: interpolated data as an Nx3 (or Nx2) numpy array, resampled time vector as numpy array
    """
    original_time = original_time * 1e3

    num = int(original_time[-1] / (float(1)/new_sampling_freq) * 10**-3)
    resampled_time = np.linspace(original_time[0], original_time[-1], num)

    f1 = interp1d(original_time, original_data)
    interpolated_pos = f1(resampled_time)

    return interpolated_pos, resampled_time


def resample_brutal(vector, n_camp, fc):
    # Function to resample the signal "vector" such that its number of samples
    # is equal to "n_camp". fc is the sampling frequency of vector
    # vector needs to be a numpy array

    vector = np.array(vector)
    length = len(vector)
    a = length / fc
    b = n_camp
    newtime = np.arange(0, a, a / b)
    if len(newtime) > b:
        newtime = np.delete(newtime, -1)
    newtime = newtime * fc
    newtime = np.round(newtime)
    newtime = newtime.astype(int)
    new_vector = vector[newtime]

    return new_vector
