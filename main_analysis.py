import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from ae_package import useful_functions
from import_data import *
import compute_metrics
from scipy import stats
import importlib
import os


# from os import sys, path
# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

importlib.reload(compute_metrics)

# p02-8/13(dalia) and s18-9/24 (Tem) excluded at the moment
# names = ["p02", "p03", "s02", "s03", "s04", "s12", "s13", "s15", "s16", "s18" "s19"]
# sess = ["/8-13/", "/8-14/", "/8-15/", "/8-15/", "/8-15/", "/9-19/", "/9-20/", "/9-20/", "/9-21/", "/9-24/, "/9-24/"]
# names = ["p03", "s02", "s03", "s04", "s12", "s13", "s15", "s16", "s18" "s19"]
# sess = ["/8-14/", "/8-15/", "/8-15/", "/8-15/", "/9-19/", "/9-20/", "/9-20/", "/9-21/", "/9-24/, "/9-24/"]

names = ["S2"]

# pdfPath = 'C:/Users/fabio/Documents/GitKrakenHub/coAE/Data/'

# mainPath = '/media/fabio/ML_Dataset/Hybrid_BMI4_2_6/HybridBMI/HybridBMI/bin/Debug/Data/'
# mainPath = 'C:/Users/fabio/OneDrive/Documents/Hybrid BMI/Hybrid_BMI4_2_6/HybridBMI/HybridBMI/bin/Debug/Data/'
mainPath = 'C:/Users/fabio/Documents/GitKrakenHub/markerless_BoMI/Practice/S2/'

# load reaching file log
df_reach = pd.read_csv(mainPath + 'PracticeLog.txt', sep='\t', header=0)
df_reach["Subject"] = 1

df_tgt = pd.read_csv('C:/Users/fabio/Documents/GitKrakenHub/hybridbmi/HybridBMI/bin/Debug/Targets/circle_coadapt.txt',
                     sep='\t', header=None)
tgt_list = df_tgt[0].values.tolist()

df_reach = df_reach.set_index(["Subject"])
idx = pd.IndexSlice

new_fs = 0.25   # frequency used to downsample rt, loss and distortions

# load number of trials and batches computed during reaching
trial_list = []
for sub in range(len(names)):
    trial_list.append(df_reach.loc[idx[sub+1], 'trial'].values[-1])

# set target position
tgt_x = []
tgt_y = []
R = 378
width = 1600
height = 900
for i in range(4):
    tgt_x.append((width / 2) + R * np.cos((2*i * np.pi / 4) + np.pi / 4))
    tgt_y.append((height / 2) + R * np.sin((2*i * np.pi / 4) + np.pi / 4))
tgt = np.array([tgt_x, tgt_y])

tgt_x_test = []
tgt_y_test = []
for i in range(8):
    tgt_x_test.append((width / 2) + R * np.cos((2 * i * np.pi / 8) + np.pi / 8))
    tgt_y_test.append((height / 2) + R * np.sin((2 * i * np.pi / 8) + np.pi / 8))
tgt_test = np.array([tgt_x_test, tgt_y_test])

#
# ee_trial = compute_metrics.compute_ee(df_reach, names, trial_list, idx, tgt, tgt_list)
#
# # compute performance per repetition (ENDPOINT ERROR)
# ee_rep = compute_metrics.compute_ee_repetition(names, ee_trial)

# compute performance during each trial
rt_trial, rt_trial_test, re_trial, re_trial_test, li_trial, li_trial_test, ms_trial, ms_trial_test, \
    tc_trial, tc_trial_end, tc_li_trial, tc_li_trial_end, tc_ms_trial, tc_ms_trial_end = \
    compute_metrics.compute_metrics_trial(df_reach, names, trial_list, tgt, tgt_test, tgt_list, idx, elapsed=50)

aagagag

# plot performance during each trial with loss and exponential fit
compute_metrics.plot_metrics_trial("adaptive", pdfSUB, df_b2, names, rt_trial, "Reaching time [s]", idx)
compute_metrics.plot_metrics_trial("adaptive", pdfSUB, df_b2, names, li_trial, "Linearity index", idx)
compute_metrics.plot_metrics_trial("adaptive", pdfSUB, df_b2, names, ms_trial, "Movement smoothness", idx)


# compute performance per repetition (REACHING TIME)
rt_rep, rt_rep_base, rt_test_rep, r_sq_rt, tau_rt = \
    compute_metrics.compute_metric_repetition("adaptive", pdfSUB, names, rt_trial, rt_trial_test, tc_trial, "Reaching time [s]", 4)
li_rep, li_rep_base, li_test_rep, r_sq_li, tau_li = \
    compute_metrics.compute_metric_repetition("adaptive", pdfSUB, names, li_trial, li_trial_test, tc_trial, "Linearity index", 4)
ms_rep, ms_rep_base, ms_test_rep, r_sq_ms, tau_ms = \
    compute_metrics.compute_metric_repetition("adaptive", pdfSUB, names, ms_trial, ms_trial_test, tc_trial, "Movement smoothness", 4)

rt_rep_static, rt_rep_base_static, rt_test_rep_static, r_sq_rt_static, tau_rt_static = \
    compute_metrics.compute_metric_repetition("static", pdfSUB, names_static,
                                              rt_trial_static, rt_trial_test_static, tc_trial, "Reaching time [s]", 4)
li_rep_static, li_rep_base_static, li_test_rep_static, r_sq_li_static, tau_li_static = \
    compute_metrics.compute_metric_repetition("static", pdfSUB, names_static,
                                              li_trial_static, li_trial_test_static, tc_trial, "Linearity index", 4)
ms_rep_static, ms_rep_base_static, ms_test_rep_static, r_sq_ms_static, tau_ms_static = \
    compute_metrics.compute_metric_repetition("static", pdfSUB, names_static,
                                              ms_trial_static, ms_trial_test_static, tc_trial, "Movement smoothness", 4)

# compute VAF during training - TIME and save vaf_pca and especially vaf_ae time as a pickle
curr_path = os.path.dirname(os.path.abspath(__file__))

# load vaf_pca and vaf_ae time
vaf_pca_time = useful_functions.load_dictionary(curr_path, '/Data/vaf_pca_time.pkl')
vaf_ae_time = useful_functions.load_dictionary(curr_path, '/Data/vaf_ae_time.pkl')
vaf_pca_time_static = useful_functions.load_dictionary(curr_path, '/Data/vaf_pca_time_static.pkl')
vaf_ae_time_static = useful_functions.load_dictionary(curr_path, '/Data/vaf_ae_time_static.pkl')

# compute and save vaf_pca and vaf_ae time
# vaf_pca_time, vaf_ae_time = compute_metrics.compute_vaf_time(df_reach, names, idx)
# vaf_pca_time_static, vaf_ae_time_static = compute_metrics.compute_vaf_time(df_reach_static, names_static, idx)
# useful_functions.save_dictionary(curr_path + '/Data/', 'vaf_pca_time.pkl', vaf_pca_time)
# useful_functions.save_dictionary(curr_path + '/Data/', 'vaf_ae_time.pkl', vaf_ae_time)
# useful_functions.save_dictionary(curr_path + '/Data/', 'vaf_pca_time_static.pkl', vaf_pca_time_static)
# useful_functions.save_dictionary(curr_path + '/Data/', 'vaf_ae_time_static.pkl', vaf_ae_time_static)

# # plot save vaf_pca and vaf_ae time
# vaf_train, vaf_train_static = compute_metrics.plot_vaf_training(names, names_static, vaf_pca_time, vaf_pca_time_static)

# # compute VAF during training - TRIALS
vaf_ae_trial = useful_functions.load_dictionary(curr_path, '/Data/vaf_ae_trial24.pkl')
vaf_ae_trial_static = useful_functions.load_dictionary(curr_path, '/Data/vaf_ae_trial24_static.pkl')
vaf_pca_trial = compute_metrics.compute_vaf_trial(df_reach, names, trial_list, 24, idx)
# vaf_pca_trial_static = compute_metrics.compute_vaf_trial(df_reach_static, names_static, trial_list, 24, idx)
# useful_functions.save_dictionary(curr_path + '/Data/', 'vaf_ae_trial24.pkl', vaf_pca_trial)
# useful_functions.save_dictionary(curr_path + '/Data/', 'vaf_ae_trial24_static.pkl', vaf_pca_trial_static)

# # # run just if you want to compute loss for static group offline. takes a while
# compute_metrics.compute_loss_static(names, sess, df_reach, idx, mainPath)
#
# plot performance against loss
corr_loss, rt_list = compute_metrics.compute_rt_resample(df_reach, df_b2, names, trial_list, idx, new_fs)
corr_loss_static, rt_list_static = compute_metrics.compute_rt_resample(df_reach_static, df_b2_static, names_static,
                                                                       trial_list_static, idx, new_fs)

# # compute distortions
# dist_list, corr_dist, xcorr_dist = compute_distortions(df_reach, df_w1, df_w2, df_b1, df_b2,
#                                                   names, batch_list, idx, rt_list, df_gain_ae, df_gain_imu, new_fs)
#
# lag = []
# for sub in range(len(names)):
#     lag.append(xcorr_dist[sub].argmax() - (len(dist_list[sub]) - 1))
#     print('Distortion is ' + str(lag[sub]) + ' behind rt')

# plot initial vs final performance for both groups - Training

# plot initial vs final performance for both groups - Test

rt_test, rt_test_static = compute_metrics.plot_metrics_test(pdfSUB, names, names_static, rt_test_rep, rt_test_rep_static,
                                                            "Reaching time [s]")
li_test, li_test_static = compute_metrics.plot_metrics_test(pdfSUB, names, names_static, li_test_rep, li_test_rep_static,
                                                            "Linearity Index")
ms_test, ms_test_static = compute_metrics.plot_metrics_test(pdfSUB, names, names_static, ms_test_rep, ms_test_rep_static,
                                                            "Movement smoothness")

# compute final performances during training (REACHING TIME, TRIALS COMPLETED and VAF)
rt_train, rt_train_static = compute_metrics.plot_metrics_training(pdfSUB, names, names_static, rt_rep_base, rt_rep_static,
                                                                  "Reaching time [s]")
li_train, li_train_static = compute_metrics.plot_metrics_training(pdfSUB, names, names_static, li_rep_base, li_rep_static,
                                                                  "Linearity index")
ms_train, ms_train_static = compute_metrics.plot_metrics_training(pdfSUB, names, names_static, ms_rep_base, ms_rep_static,
                                                                  "Movement smoothness")
tc_train, tc_train_static = compute_metrics.plot_tc_training(pdfSUB, names, names_static, tc_trial, tc_trial_end,
                                                             tc_trial_static, tc_trial_end_static)
tc_li_train, tc_li_train_static = compute_metrics.plot_tc_training(pdfSUB, names, names_static, tc_li_trial, tc_li_trial_end,
                                                             tc_li_trial_static, tc_li_trial_end_static)
tc_ms_train, tc_ms_train_static = compute_metrics.plot_tc_training(pdfSUB, names, names_static, tc_ms_trial, tc_ms_trial_end,
                                                             tc_ms_trial_static, tc_ms_trial_end_static)
vaf_train, vaf_train_static = compute_metrics.plot_vaf_training(names, names_static, vaf_pca_time, vaf_pca_time_static)

# # plot initial vs final performance for both groups - Test
# rt_test, rt_test_static = plot_test(names, names_static, rt_test_rep, rt_test_rep_static)

compute_metrics.plot_whole_session("adaptive", pdfSUB, names, rt_trial, rt_trial_test)

compute_metrics.plot_whole_session("static", pdfSUB, names_static, rt_trial_static, rt_trial_test_static)


# Repeated measures ANOVA stat


# _, p_rt_train_pre_norm = stats.ttest_ind(np.divide(rt_train[0, :], rt_test[0, :]),
#                                          np.divide(rt_train_static[0, :], rt_test_static[0, :]))
# _, p_rt_train_post_norm = stats.ttest_ind(np.divide(rt_train[1, :], rt_test[0, :]),
#                                           np.divide(rt_train_static[1, :], rt_test_static[0, :]),
#                                           nan_policy="omit")

_, p_ee_pre = stats.ttest_ind(ee_train[0, :], ee_train_static[0, :])
_, p_ee_post = stats.ttest_ind(ee_train[1, :], ee_train_static[1, :], nan_policy="omit")

_, p_ee_pre = stats.ttest_ind(li_train[7, :], li_train_static[7, :])

# write file .txt for statistica
np.savetxt('ee_train.txt', ee_train.T, delimiter='\t')
np.savetxt('ee_train_static.txt', ee_train_static.T, delimiter='\t')

# at the end of the loop:
plt.ion()
# pdf.close()
pdfCCA.close()
pdfSUB.close()

x = np.arange(1, 10)
yerr_ad = 1.96 * np.nanstd(vaf_ae_trial, axis=0) / np.sqrt(len(vaf_ae_trial))
yerr_ad_stat = 1.96 * np.nanstd(vaf_ae_trial_static, axis=0) / np.sqrt(len(vaf_ae_trial_static))

plt.figure()
plt.errorbar(x, np.nanmean(vaf_ae_trial, axis=0), yerr=yerr_ad, color='r', label='adaptive')
plt.errorbar(x, np.nanmean(vaf_ae_trial_static, axis=0), yerr=yerr_ad_stat,
             color='k', label='static')
# plt.legend()
plt.ylabel('VAF')
plt.xlabel('Trial')
plt.legend()


plt.figure()
for sub in range(10):
    plt.plot(vaf_ae_trial[sub], 'r', label='adaptive')
    plt.plot(vaf_ae_trial_static[sub], 'k', label='static')
    # plt.legend()
    plt.ylabel('VAF')
    plt.xlabel('Trial')




