import numpy as np
import warnings


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


def f1_compute(tr, precision, positive_class=1):
    assert positive_class == 1 or positive_class == 0, "current version on 20221201 only works for binary class"
    f1_score = 2 * tr * precision / (tr + precision)
    return f1_score[positive_class]


def mcc_compute(tr, fr, positive_class=1):
    """
    The implementation is based on https://blog.csdn.net/Winnycatty/article/details/82972902
    The undefined MCC that a whole row or column of the confusion matrix M is zero is treated the left column of page 5
    of the paper: Davide Chicco and Giuseppe Jurman. "The advantages of the matthews correlation coefficient (mcc) over
        f1 score and accuracy in binary classification evaluation". BMC Genomics, 21, 01, 2020
    The confusion matrix M is
        M = (tp fn
             fp tn)

    TC on 2022/11/25
    TC update 2022/11/28
    Liyan update 2022/12/1
    """
    # todo 12/1 N is not used.
    fenzi = tr[0] * tr[1] - fr[0] * fr[1]
    fenmu = tr * np.flip(fr)
    fenmu = pow(fenmu[0] * fenmu[1], 0.5)
    tp = tr[positive_class]  # defined positive_class is positive
    tn = tr[1-positive_class]
    fn = fr[positive_class]
    fp = fr[1-positive_class]
    if fenmu == 0:
        if (tp or tn) and fn == 0 and fp == 0:  # M has only 1 non-0 entry & all are correctly predicted
            mcc = 1
        elif (fp or fn) and tp == 0 and tn == 0:  # M has only 1 non-0 entry & all are incorrectly predicted
            mcc = -1
        else:  # a row or a column of M are zero
            mcc = 0
    else:
        mcc = fenzi/fenmu
    return mcc


def pf_epoch(S, N, P, theta, t, y_t, p_t, positive_class=1):
    """ Reference:
    Gama, Joao, Raquel Sebastiao, and Pedro Pereira Rodrigues.
    "On evaluating stream learning algorithms." Machine learning 90.3 (2013): 317-346."""
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

    assert positive_class == 1 or positive_class == 0, "current version on 20221201 only works for binary class"
    tr = recall  # positive class is 1, then tpr = tr[1], tnr = tr[0]
    fr = 1 - tr  # positive class is 1, then fnr = fr[1], fpr = fr[0]
    precision = tr / (tr + np.flip(fr))
    precision = precision[positive_class]
    f1_score = f1_compute(tr, precision)
    mcc = mcc_compute(tr, fr)
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
    S = np.empty([len(y_tru), 2])
    N = np.empty([len(y_tru), 2])
    P = np.empty([len(y_tru), 2])
    recalls_tt = np.empty([len(y_tru), 2])
    Gmean_tt = np.empty([len(y_tru), ])
    ave_acc_tt = np.empty([len(y_tru), ])
    precision_tt = np.empty([len(y_tru), ])
    f1_score_tt = np.empty([len(y_tru), ])
    mcc_tt = np.empty([len(y_tru), ])
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
    y = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
    p = [0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    [recall0, recall1, Gmean, mcc, precision, f1_score, avg_acc] \
        = compute_online_PF(y, p, theta)
    # print
    print('Gmean: ', np.nanmean(Gmean))
    print('mcc: ', np.nanmean(mcc))
    print('recall0: ', np.nanmean(recall0))
    print('recall1: ', np.nanmean(recall1))
    print('precision: ', np.nanmean(precision))
    print('f1_score: ', np.nanmean(f1_score))
    print('avg_acc: ', np.nanmean(avg_acc))
