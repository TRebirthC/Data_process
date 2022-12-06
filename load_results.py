import os
import numpy as np
import scipy.stats as st

dir_rslt_save = "result/rslt.save/"
def data_id_2name(project_id):
    """2021-12-19. the below projects suffer issues individually as below.
    homebrew        our method arises an error, no available data
    neutron         ood, error happens at the 10,000 steps
    npm             oob, error n_data < 10,000
    spring-integration, <10,000
    """
    if project_id == 0:
        project_name = "brackets"
    elif project_id == 1:
        project_name = "broadleaf"
    elif project_id == 2:
        project_name = "camel"
    elif project_id == 3:
        project_name = "corefx"
    elif project_id == 4:
        project_name = "django"
    elif project_id == 5:
        project_name = "fabric"
    elif project_id == 6:
        project_name = "jgroups"
    elif project_id == 7:
        project_name = "nova"
    elif project_id == 8:
        project_name = "rails"
    elif project_id == 9:
        project_name = "rust"
    elif project_id == 10:
        project_name = "tensorflow"
    elif project_id == 11:
        project_name = "tomcat"
    elif project_id == 12:
        project_name = "vscode"
    elif project_id == 13:
        project_name = "wp-calypso"
    elif project_id == 14:
        project_name = "npm"
    elif project_id == 15:
        project_name = "spring-integration"
    elif project_id == 16:
        project_name = "neutron"
    elif project_id == 17:
        project_name = "neutron_test"
    elif project_id == 18:
        project_name = "npm_test"
    elif project_id == 19:
        project_name = "spring-integration_test"
    else:
        raise Exception("undefined data id.")
    return project_name

def compute_online_PF(y_tru, y_pre, theta_eval=0.99):
    """
    para theta_eval: used in the online PF evaluation, theta_eval=0.99 by default
    reference: 2013_[JML, #, Leandro based] On evaluate stream learning algorithm

    2021-9      Shuxian helps with creating this method
    2021-12-7   Liyan updates this method slightly making it easier to read
    """
    S = np.zeros([len(y_tru), 2])
    N = np.zeros([len(y_tru), 2])
    recalls_tt = np.zeros([len(y_tru), 2])
    Gmean_tt = np.zeros([len(y_tru), ])  # Shuxian: [:, 1] --> [:, ]
    # TODO 2021-10-21 Liyan: shape is (n, 1) but other metric is (n,)
    for t in range(len(y_tru)):
        y_t = y_tru[t]
        p_t = y_pre[t]
        [recalls_tt[t, :], Gmean_tt[t]] = pf_epoch(S, N, theta_eval, t, y_t, p_t)
        recall0_tt = recalls_tt[:, 0]
        recall1_tt = recalls_tt[:, 1]

    # todo silence the warning of nan 8-2
    return recall0_tt, recall1_tt, Gmean_tt

def pf_epoch(S, N, theta, t, y_t, p_t):
    if t == 0:
        c = int(y_t)  # class 0 or 1
        S[t, c] = (y_t == p_t)
        N[t, c] = 1
    else:
        S[t, :] = S[t-1, :]
        N[t, :] = N[t-1, :]
        c = int(y_t)  # class 0 or 1
        S[t, c] = (y_t == p_t) + theta * (S[t-1, c])
        N[t, c] = 1 + theta * N[t-1, c]

    recall = S[t, :] / N[t, :]
    gmean = Gmean_compute(recall)
    return recall, gmean

def Gmean_compute(recall):
    Gmean = 1
    for r in recall:
        Gmean = Gmean * r
    Gmean = pow(Gmean, 1/len(recall))

    return Gmean

def uti_eval_pfs(test_y_tru, test_y_pre, verbose=False):
    """evaluate PFs in terms of g-mean, recall-1, recall-0..
    2022-6-2    Separate this func.
    2022-8-1
    """
    # ave PFs across test steps
    theta_eval = 0.99
    r0_tt, r1_tt, gmean_tt = compute_online_PF(test_y_tru, test_y_pre, theta_eval)
    gmean_ave_tt, r1_ave_tt, r0_ave_tt = np.nanmean(gmean_tt), np.nanmean(r1_tt), np.nanmean(r0_tt)
    if verbose:
        print("\t ave online gmean=%.4f, r1=%.4f, r0=%.4f" % (gmean_ave_tt, r1_ave_tt, r0_ave_tt))
    return gmean_ave_tt, r1_ave_tt, r0_ave_tt, gmean_tt, r1_tt, r0_tt

def uti_rslt_dir(clf_name="odasc", project_id=1, wait_days=15,
                 n_trees=5, theta_imb=0.9, theta_cl=0.8):
    # 2022-7-30
    clf_name = clf_name.lower()
    pre_to_dir = dir_rslt_save + data_id_2name(project_id) + "/" + clf_name + "/" + str(wait_days) + "d"
    to_dir = pre_to_dir + "/n_trees" + str(n_trees)  # para info, classifier
    if clf_name != "oza":
        to_dir += "-theta_imb" + str(theta_imb)
    if clf_name != "oza" and clf_name != "oob" and clf_name != "oob_filtering" and clf_name != "oob_aio" and clf_name != "oob_addcp_adp":
        to_dir += "-theta_cl" + str(theta_cl)
    # if clf_name != "oza" and clf_name != "oob" and clf_name != "our":
    #     to_dir += "k_refine%d" % k_power_refine  # modified
    return to_dir

def uti_rslt_dir_analyze(to_dir, clf_name, nb_test, seed):
    """
    A method used in jit_sdp_1call().
    Analyse filenames in the directory 'to_dir' and find 'T' that is larger than nb_data,
    so that we can save computational cost to downside load it

    2022-6-2    Adapt from ijcnn
    """
    exist_result = False
    fold_names = next(os.walk(to_dir))[1]
    if len(fold_names) > 0:
        for _, fold_name in enumerate(fold_names):
            nb_test_saved = int(fold_name[fold_name.find("T") + 1:])
            if nb_test_saved >= nb_test:
                to_dir_4save = to_dir
                to_dir += "/T" + str(nb_test_saved) + "/"
                flnm_test = to_dir + clf_name + ".rslt_test.s" + str(seed)
                flnm_train = to_dir + clf_name + ".rslt_train.s" + str(seed)
                exist_result = os.path.exists(flnm_test) and os.path.exists(flnm_train)
                if exist_result:
                    break
                else:
                    """handle empty (e.g.) T5000 folder"""
                    to_dir = to_dir_4save
    return exist_result, to_dir

seed_lst = range(30)
clf_name = "pbsa"
project_id = 17
wait_days = 90
n_tree, theta_imb, theta_cl = 20, 0.99, 0
nb_test = 18046
for ss, seed in enumerate(seed_lst):
    to_dir = uti_rslt_dir(clf_name, project_id, wait_days, n_tree, theta_imb, theta_cl)
    os.makedirs(to_dir, exist_ok=True)
    # analyze filenames in this dir:
    # find T that is larger than nb_data to save computational cost and load the results.
    exist_result, to_dir = uti_rslt_dir_analyze(to_dir, clf_name, nb_test, seed)
    if not exist_result:
        to_dir += "/T" + str(nb_test) + "/"
        os.makedirs(to_dir, exist_ok=True)
    # file_name-s
    flnm_test = "%s%s.rslt_test.s%d" % (to_dir, clf_name, seed)
    flnm_train = "%s%s.rslt_train.s%d" % (to_dir, clf_name, seed)

    """load or compute"""
    if exist_result:
        rslt_test = np.loadtxt(flnm_test)
        rslt_train = np.loadtxt(flnm_train)
    test_y_tru, test_y_pre = rslt_test[:, 1], rslt_test[:, 2]
    _, _, _, gmean_tt, r1_tt, r0_tt = uti_eval_pfs(test_y_tru, test_y_pre)

    # assign
    if ss == 0:  # init
        n_row, n_col = gmean_tt.shape[0], len(seed_lst)
        cl_rmse, gmean_tt_ss = np.empty(n_col), np.empty((n_row, n_col))
        r1_tt_ss, r0_tt_ss = np.copy(gmean_tt_ss), np.copy(gmean_tt_ss)
    gmean_tt_ss[:, ss], r1_tt_ss[:, ss], r0_tt_ss[:, ss] = gmean_tt, r1_tt, r0_tt
gmean = []
for i in range(len(seed_lst)):
    temp = []
    for j in range(len(gmean_tt_ss)):
        temp.append(gmean_tt_ss[j, i])
    gmean.append(np.nanmean(temp))
print(gmean)
b = [0.7988]
t, p = st.ttest_1samp(gmean, b)
print(t)
print(p)