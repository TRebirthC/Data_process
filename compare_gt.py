import os

import pandas as pd
import numpy as np

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
        project_name = "neutron"
    elif project_id == 8:
        project_name = "nova"
    elif project_id == 9:
        project_name = "npm"
    elif project_id == 10:
        project_name = "rails"
    elif project_id == 11:
        project_name = "rust"
    elif project_id == 12:
        project_name = "tensorflow"
    elif project_id == 13:
        project_name = "tomcat"
    elif project_id == 14:
        project_name = "vscode"
    elif project_id == 15:
        project_name = "wp-calypso"
    else:
        raise Exception("undefined data id.")
    return project_name

def data_name_to_id(project_name):
    if project_name == "brackets":
        project_id = 0
    elif project_name == "broadleaf":
        project_id = 1
    elif project_name == "camel":
        project_id = 2
    elif project_name == "corefx":
        project_id = 3
    elif project_name == "django":
        project_id = 4
    elif project_name == "fabric":
        project_id = 5
    elif project_name == "jgroups":
        project_id = 6
    elif project_name == "neutron":
        project_id = 7
    elif project_name == "nova":
        project_id = 8
    elif project_name == "npm":
        project_id = 9
    elif project_name == "rails":
        project_id = 10
    elif project_name == "rust":
        project_id = 11
    elif project_name == "tensorflow":
        project_id = 12
    elif project_name == "tomcat":
        project_id = 13
    elif project_name == "vscode":
        project_id = 14
    elif project_name == "wp-calypso":
        project_id = 15
    return project_id


def load_Ds(dir):
    Ds = np.loadtxt(dir)
    return Ds


def load_gt(dir):
    gt_df = pd.read_csv(dir)
    gt_dataset = []
    for i in range(16):
        gt_method = []
        for j in range(3):
            gt_project_rank = []
            for k in range(16):
                index = k + j * 16 + i * 16 * 3
                dataset = str.split(gt_df["used_project"][index], sep='\'')
                project_id = data_name_to_id(dataset[1])
                gt_project_rank.append(project_id)
            gt_method.append(gt_project_rank)
        gt_dataset.append(gt_method)
    gt = np.array(gt_dataset)
    return gt


def load_window(dir):
    wd_df = pd.read_csv(dir)
    feature = []
    sp = []
    js = []
    for i in range(16):
        feature_method = []
        sp_method = []
        js_method = []
        for j in range(3):
            feature_across_cp = []
            sp_across_cp = []
            js_across_cp = []
            for k in range(16):
                index = k + j * 16 + i * 16 * 3
                feature_across_cp.append(wd_df["metrics_dis"][index])
                sp_across_cp.append(wd_df["spearman_cor"][index])
                js_across_cp.append(wd_df["js_div"][index])
            feature_method.append(feature_across_cp)
            sp_method.append(sp_across_cp)
            js_method.append(js_across_cp)
        feature.append(feature_method)
        sp.append(sp_method)
        js.append(js_method)
    feature = np.array(feature)
    sp = np.array(sp)
    js = np.array(js)
    return feature, sp, js


def get_rank_from_similaroty(Ds, feature, sp, js):
    Ds_rank = np.zeros(feature.shape)
    feature_rank = np.zeros(feature.shape)
    sp_rank = np.zeros(sp.shape)
    js_rank = np.zeros(js.shape)
    for i in range(16):
        for j in range(3):
            Ds_rank[i][j] = np.argsort(-Ds[i])
            feature_rank[i][j] = np.argsort(-feature[i][j])
            sp_rank[i][j] = np.argsort(-sp[i][j])
            js_rank[i][j] = np.argsort(-js[i][j])
    return Ds_rank, feature_rank, sp_rank, js_rank


def delete_invalid_rank(gt, Ds_rank, feature_rank, sp_rank, js_rank, sp):
    gt_valid, Ds_rank_valid, feature_rank_valid, sp_rank_valid, js_rank_valid = gt, Ds_rank, feature_rank, sp_rank, js_rank
    gt_valid = gt_valid.astype(float)
    Ds_rank_valid = Ds_rank_valid.astype(float)
    feature_rank_valid = feature_rank_valid.astype(float)
    sp_rank_valid = sp_rank_valid.astype(float)
    js_rank_valid = js_rank_valid.astype(float)
    for i in range(16):
        for j in range(3):
            for k in range(16):
                if np.isnan(sp[i][j][k]):
                    for z in range(16):
                        if gt_valid[i][j][z] == k:
                            gt_valid[i][j][z] = np.nan
                        if Ds_rank_valid[i][j][z] == k:
                            Ds_rank_valid[i][j][z] = np.nan
                        if feature_rank_valid[i][j][z] == k:
                            feature_rank_valid[i][j][z] = np.nan
                        if sp_rank_valid[i][j][z] == k:
                            sp_rank_valid[i][j][z] = np.nan
                        if js_rank_valid[i][j][z] == k:
                            js_rank_valid[i][j][z] = np.nan
    return gt_valid, Ds_rank_valid, feature_rank_valid, sp_rank_valid, js_rank_valid


def calculate_order_dif(temp_gt, temp_target):
    opt = 1
    dif = 0
    if opt == 0:
        for i in range(len(temp_gt)):
            for j in range(len(temp_target)):
                if temp_gt[i] == temp_target[j]:
                    dif += np.absolute(i-j)
    elif opt == 1:
        for i in range(1):
            for j in range(len(temp_target)):
                if temp_gt[i] == temp_target[j]:
                    dif += np.absolute(i-j)
    return dif


def calculate_rank_dif(gt_valid, target_rank_valid):
    len_valid = []
    rank_dif = np.zeros([16, 3])
    for i in range(16):
        valid_length = 0
        for j in range(3):
            temp_gt = []
            temp_target = []
            for k in range(16):
                if not np.isnan(gt_valid[i][j][k]):
                    temp_gt.append(gt_valid[i][j][k])
                if not np.isnan(target_rank_valid[i][j][k]):
                    temp_target.append(target_rank_valid[i][j][k])
            valid_length = len(temp_gt)
            dif = calculate_order_dif(temp_gt, temp_target)
            rank_dif[i][j] = dif
        len_valid.append(valid_length)
    return np.array(len_valid), rank_dif


def save_result(length_valid, Ds_dif, feature_dif, sp_dif, js_dif, opt):
    if opt == 1:
        to_dir_csv = "ground_truth_compare/"
        os.makedirs(to_dir_csv, exist_ok=True)
        to_flnm_csv = to_dir_csv + "compare_result.csv"
        with open(to_flnm_csv, "a+") as fh2:
            if not os.path.getsize(to_flnm_csv):  # header
                print("%s,%s,%s,%s,%s,%s,%s" % (
                    "target_project", "method", "valid_length", "pre-set_dif", "feature_dif",
                    "sp_dif", "js_dif"), file=fh2)
            for i in range(16):
                project_name = data_id_2name(i)
                print("%s,%s,%d,%d,%d,%d,%d" % (
                    project_name, "odasc", length_valid[i], Ds_dif[i][0], feature_dif[i][0], sp_dif[i][0], js_dif[i][0]), file=fh2)
                print("%s,%s,%d,%d,%d,%d,%d" % (
                    project_name, "oob", length_valid[i], Ds_dif[i][1], feature_dif[i][1], sp_dif[i][1], js_dif[i][1]), file=fh2)
                print("%s,%s,%d,%d,%d,%d,%d" % (
                    project_name, "pbsa", length_valid[i], Ds_dif[i][2], feature_dif[i][2], sp_dif[i][2], js_dif[i][2]), file=fh2)



if __name__ == "__main__":
    Ds_dir = "ground_truth_compare/20221201_D_similarity.txt"
    Ds = load_Ds(Ds_dir)
    gt_dir = "ground_truth_compare/20221130_ground_truth.csv"
    gt = load_gt(gt_dir)
    window_dir = "ground_truth_compare/20221202_window.csv"
    feature, sp, js = load_window(window_dir)
    Ds_rank, feature_rank, sp_rank, js_rank = get_rank_from_similaroty(Ds, feature, sp, js)
    gt_valid, Ds_rank_valid, feature_rank_valid, sp_rank_valid, js_rank_valid = \
        delete_invalid_rank(gt, Ds_rank, feature_rank, sp_rank, js_rank, sp)
    length_valid, Ds_dif =calculate_rank_dif(gt_valid, Ds_rank_valid)
    length_valid, feature_dif = calculate_rank_dif(gt_valid, feature_rank_valid)
    length_valid, sp_dif = calculate_rank_dif(gt_valid, sp_rank_valid)
    length_valid, js_dif = calculate_rank_dif(gt_valid, js_rank_valid)
    opt = 1
    save_result(length_valid, Ds_dif, feature_dif, sp_dif, js_dif, opt)
    # print(Ds_rank_valid[15][0])
    # print(gt_valid[15][0])
    # print(feature_rank_valid[15][0])
    # print(sp_rank_valid[15][0])
