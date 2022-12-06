import numpy as np
import warnings
import pandas as pd


# 2022/10/12 Shuxian insert online ave_accuracy todo to double-check
# 2022/11/25 TC inserted other online PF metrics

# silence the warning
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')


def eval_clf_online(result_np, theta_pf=0.99):
    """for pp-report
    :param result_np: (time, y_true, y_pred), created in xxx.jit_sdp_1call()
    :param theta_pf: the para theta_imb to evaluate the online pf
    :return:
    Liyan on 2021-10-12
    """
    # print(info_str) >> the index is 0: time, 1: y_true, 2: y_pred
    actual_labels = result_np[:, 1]
    predict_labels = result_np[:, 2]

    recall0_tt, recall1_tt, gmean_tt = compute_online_PF(actual_labels, predict_labels, theta_pf)
    gmean_tt = gmean_tt.reshape(len(gmean_tt))
    recall0_tt = recall0_tt.reshape(len(recall0_tt))
    recall1_tt = recall1_tt.reshape(len(recall1_tt))
    # pp-report, (3, #steps)
    pf_metrics_tt = np.vstack((gmean_tt, recall0_tt, recall1_tt))

    # ave across time steps
    gmean_ave = np.nanmean(gmean_tt, 0)
    recall0_ave = np.nanmean(recall0_tt, 0)
    recall1_ave = np.nanmean(recall1_tt, 0)
    # pp-report, 1*3
    pf_metrics_ave = (gmean_ave, recall0_ave, recall1_ave)

    metric_names = np.array(("gmean", "recall0", "recall1"))
    return pf_metrics_tt, pf_metrics_ave, metric_names


def Gmean_compute(recall):
    Gmean = 1
    for r in recall:
        Gmean = Gmean * r
    Gmean = pow(Gmean, 1/len(recall))
    return Gmean


def avg_acc_compute(recall):
    avg_acc = np.mean(recall)
    return avg_acc


def f1_compute(TRP, percision):
    f1_score = 2*TRP*percision/(TRP+percision)
    return f1_score


def mcc_compute(S, N, P, t):
    """
    The implementation is based on https://blog.csdn.net/Winnycatty/article/details/82972902
    TC on 2022/11/25
    TC update 2022/11/28
    """
    TP = S[t, 1]
    TN = S[t, 0]
    FP = P[t, 1] - TP
    FN = P[t, 0] - TN
    mcc = (TP*TN-FP*FN)/(pow((TP+FN)*(TP+FP)*(TN+FP)*(TN+FN), 0.5))
    return mcc


def pf_epoch(S, N, P, theta, t, y_t, p_t):
    if t == 0:
        c = int(y_t)  # class 0 or 1
        S[t, c] = (y_t == p_t)
        N[t, c] = 1
        P[t, c] = 1
    else:
        S[t, :] = S[t-1, :]
        N[t, :] = N[t-1, :]
        P[t, :] = P[t-1, :]
        c = int(y_t)  # class 0 or 1
        S[t, c] = (y_t == p_t) + theta * (S[t-1, c])
        N[t, c] = 1 + theta * N[t-1, c]
        p = int(p_t)  # the number of predicted positive data
        P[t, p] = 1 + theta * P[t-1, p]

    recall = S[t, :] / N[t, :]
    tpr = recall[1]
    precision = S[t, 1] / P[t, 1]
    f1_score = f1_compute(tpr, precision)
    mcc = mcc_compute(S, N, P, t)
    gmean = Gmean_compute(recall)
    ave_acc = avg_acc_compute(recall)
    return recall, gmean, mcc, precision, f1_score, ave_acc


def compute_online_PF(y_tru, y_pre, theta_eval=0.99):
    """
    para theta_eval: used in the online PF evaluation, theta_eval=0.99 by default
    reference: 2013_[JML, #, Leandro based] On evaluate stream learning algorithm

    2021/9      Shuxian helps with creating this method
    2021/12/7   Liyan updates this method slightly making it easier to read
    2022/11/25  TC implemented more PF metrics including online mcc
    """
    S = np.zeros([len(y_tru), 2])
    N = np.zeros([len(y_tru), 2])
    P = np.zeros([len(y_tru), 2])
    recalls_tt = np.zeros([len(y_tru), 2])
    Gmean_tt = np.zeros([len(y_tru), ])
    ave_acc_tt = np.zeros([len(y_tru), ])
    precision_tt = np.zeros([len(y_tru), ])
    f1_score_tt = np.zeros([len(y_tru), ])
    mcc_tt = np.zeros([len(y_tru), ])
    # compute at each test step
    for t in range(len(y_tru)):
        y_t = y_tru[t]
        p_t = y_pre[t]
        recalls_tt[t, :], Gmean_tt[t], mcc_tt[t], precision_tt[t], f1_score_tt[t], ave_acc_tt[t] \
            = pf_epoch(S, N, P, theta_eval, t, y_t, p_t)
        recall0_tt = recalls_tt[:, 0]
        recall1_tt = recalls_tt[:, 1]
    return recall0_tt, recall1_tt, Gmean_tt, mcc_tt, precision_tt, f1_score_tt, ave_acc_tt


if __name__ == '__main__':
    theta = 0.99
    mcc_window = []
    for i in range(100):
        file = "unexpect/broadleaf/oob-human/effort0.1/error0/15d/n_trees40-theta_imb0.95/T10000/oob.rslt_test.s"
        dir = file + str(i)
        df = np.loadtxt(
            dir)
        y = df[:, 1]
        p = df[:, 2]
        [recall0, recall1, Gmean, mcc, percision, f1_score, avg_acc] \
            = compute_online_PF(y, p, theta)
        mcc_window.append(np.nanmean(mcc))
    print(mcc_window)
    print(np.nanmean(mcc_window))
    # df = np.loadtxt("unexpect/broadleaf/oob-human/effort0.1/error0/15d/n_trees40-theta_imb0.95/T10000/oob.rslt_test.s0")
    # y = df[:, 1]
    # p = df[:, 2]
    # # y = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
    # # p = [0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    # [recall0, recall1, Gmean, mcc, percision, f1_score, avg_acc] \
    #     = compute_online_PF(y, p, theta)
    # # print
    # print(np.nanmean(Gmean))
    # print(np.nanmean(mcc))
    # print(np.nanmean(recall0))
    # print(np.nanmean(recall1))
    # print(np.nanmean(percision))
    # print(np.nanmean(f1_score))
    # print(np.nanmean(avg_acc))
