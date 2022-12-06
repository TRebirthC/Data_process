import os

import pandas as pd
import numpy as np
import scipy.stats


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
    wd_df = pd.read_csv(dir)
    start, core, license, language, domain, company, user_interface, use_database, localized, single_pl = \
        [], [], [], [], [], [], [], [], [], []
    for i in range(16):
        start_across_cp, core_across_cp, license_across_cp, language_across_cp, domain_across_cp, company_across_cp,\
        user_interface_across_cp, use_database_across_cp, localized_across_cp, single_pl_across_cp = \
            [], [], [], [], [], [], [], [], [], []
        for j in range(16):
            index = j + i * 16
            start_across_cp.append(wd_df["0"][index])
            core_across_cp.append(wd_df["1"][index])
            license_across_cp.append(wd_df["2"][index])
            language_across_cp.append(wd_df["3"][index])
            domain_across_cp.append(wd_df["4"][index])
            company_across_cp.append(wd_df["5"][index])
            user_interface_across_cp.append(wd_df["6"][index])
            use_database_across_cp.append(wd_df["7"][index])
            localized_across_cp.append(wd_df["8"][index])
            single_pl_across_cp.append(wd_df["9"][index])
        start.append(start_across_cp)
        core.append(core_across_cp)
        license.append(license_across_cp)
        language.append(language_across_cp)
        domain.append(domain_across_cp)
        company.append(company_across_cp)
        user_interface.append(user_interface_across_cp)
        use_database.append(use_database_across_cp)
        localized.append(localized_across_cp)
        single_pl.append(single_pl_across_cp)

    start = np.array(start)
    core = np.array(core)
    license = np.array(license)
    language = np.array(language)
    domain = np.array(domain)
    company = np.array(company)
    user_interface = np.array(user_interface)
    use_database = np.array(use_database)
    localized = np.array(localized)
    single_pl = np.array(single_pl)
    return start, core, license, language, domain, company, user_interface, use_database, localized, single_pl


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
    defect = []
    commit = []
    median = []
    max = []
    std = []
    sp = []
    js = []
    for i in range(16):
        defect_method = []
        commit_method = []
        median_method = []
        max_method = []
        std_method = []
        sp_method = []
        js_method = []
        for j in range(3):
            defect_across_cp = []
            commit_across_cp = []
            median_across_cp = []
            max_across_cp = []
            std_across_cp = []
            sp_across_cp = []
            js_across_cp = []
            for k in range(16):
                index = k + j * 16 + i * 16 * 3
                defect_across_cp.append(wd_df["defect_ratio"][index])
                commit_across_cp.append(wd_df["n_commit"][index])
                median_across_cp.append(wd_df["median_feature"][index])
                max_across_cp.append(wd_df["maximum_feature"][index])
                std_across_cp.append(wd_df["std_feature"][index])
                sp_across_cp.append(wd_df["spearman_cor"][index])
                js_across_cp.append(wd_df["js_div"][index])
            defect_method.append(defect_across_cp)
            commit_method.append(commit_across_cp)
            median_method.append(median_across_cp)
            max_method.append(max_across_cp)
            std_method.append(std_across_cp)
            sp_method.append(sp_across_cp)
            js_method.append(js_across_cp)
        defect.append(defect_method)
        commit.append(commit_method)
        median.append(median_method)
        max.append(max_method)
        std.append(std_method)
        sp.append(sp_method)
        js.append(js_method)
    defect = np.array(defect)
    commit = np.array(commit)
    median = np.array(median)
    max = np.array(max)
    std = np.array(std)
    sp = np.array(sp)
    js = np.array(js)
    return defect, commit, median, max, std, sp, js


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


def calculate_rank(temp_features, used_feature):
    similarity = []
    for i in range(16):
        temp_sim = 0
        for j in range(len(used_feature)):
            if j in temp_features:
                temp_sim = temp_sim + used_feature[j][i]
        similarity.append(temp_sim)
    similarity = np.array(similarity)
    rank = np.argsort(-similarity)
    return rank


def calculate_performance(temp_rank, used_gt_rank, js_dev):
    temp_rank_valid, gt_valid = temp_rank, used_gt_rank
    temp_rank_valid = temp_rank_valid.astype(float)
    gt_valid = gt_valid.astype(float)
    for i in range(16):
        if np.isnan(js_dev[i]):
            for j in range(16):
                if gt_valid[j] == i:
                    gt_valid[j] = np.nan
                if temp_rank_valid[j] == i:
                    temp_rank_valid[j] = np.nan
    cor, p = scipy.stats.kendalltau(temp_rank_valid, gt_valid)
    return cor


def fs_run(used_feature, used_gt_rank):
    selected_features = []
    performance = -1
    is_better = 1
    while is_better > 0:
        is_better = -1
        for i in range(len(used_feature)):
            if i not in selected_features:
                temp_features = selected_features.copy()
                temp_features.append(i)
                temp_rank = calculate_rank(temp_features, used_feature)
                temp_performance = calculate_performance(temp_rank, used_gt_rank, used_feature[-1])
                if temp_performance > performance:
                    performance = temp_performance
                    selected_features.append(i)
                    is_better = 1

        if len(selected_features) > 1:
            for i in range(len(selected_features)):
                temp_features = selected_features.copy()
                removed_feature = temp_features.pop(i)
                temp_rank = calculate_rank(temp_features, used_feature)
                temp_performance = calculate_performance(temp_rank, used_gt_rank, used_feature[-1])
                if temp_performance > performance:
                    performance = temp_performance
                    selected_features.remove(removed_feature)
                    is_better = 1
    selected_features = np.array(selected_features)
    return selected_features


def feature_selection(start, core, license, language, domain, company, user_interface, use_database, localized,
                      single_pl, defect, commit, median, max, std, sp, js, gt, base_method):
    used_method = 0
    if base_method == "odasc":
        used_method = 0
    elif base_method == "oob":
        used_method = 1
    elif base_method == "pbsa":
        used_method = 2
    feature_selected_across_tp = []
    for i in range(16):
        used_feature = [start[i], core[i], license[i], language[i], domain[i], company[i], user_interface[i],
                        use_database[i], localized[i], single_pl[i],
                        defect[i][used_method], commit[i][used_method], median[i][used_method],
                        max[i][used_method], std[i][used_method], sp[i][used_method], js[i][used_method]]
        used_feature = np.array(used_feature)
        used_gt_rank = gt[i][used_method]
        feature_selected_across_tp.append(fs_run(used_feature, used_gt_rank))
    feature_selected_across_tp = np.array(feature_selected_across_tp)
    np.savetxt("ground_truth_compare/selected_features_"+base_method+".txt", feature_selected_across_tp)
    return feature_selected_across_tp





if __name__ == "__main__":
    Ds_dir = "ground_truth_compare/20221206_D_similarity.csv"
    start, core, license, language, domain, company, user_interface, use_database, localized, single_pl = load_Ds(Ds_dir)
    gt_dir = "ground_truth_compare/20221130_ground_truth.csv"
    gt = load_gt(gt_dir)
    window_dir = "ground_truth_compare/20221206_window.csv"
    defect, commit, median, max, std, sp, js = load_window(window_dir)
    base_method = "odasc"
    result = feature_selection(start, core, license, language, domain, company, user_interface, use_database,
                               localized, single_pl, defect, commit, median, max, std, sp, js, gt, base_method)
    # Ds_rank, feature_rank, sp_rank, js_rank = get_rank_from_similaroty(Ds, feature, sp, js)
    # gt_valid, Ds_rank_valid, feature_rank_valid, sp_rank_valid, js_rank_valid = \
    #     delete_invalid_rank(gt, Ds_rank, feature_rank, sp_rank, js_rank, sp)
    # length_valid, Ds_dif =calculate_rank_dif(gt_valid, Ds_rank_valid)
    # length_valid, feature_dif = calculate_rank_dif(gt_valid, feature_rank_valid)
    # length_valid, sp_dif = calculate_rank_dif(gt_valid, sp_rank_valid)
    # length_valid, js_dif = calculate_rank_dif(gt_valid, js_rank_valid)
    # opt = 1
    # save_result(length_valid, Ds_dif, feature_dif, sp_dif, js_dif, opt)
    # print(Ds_rank_valid[15][0])
    # print(gt_valid[15][0])
    # print(feature_rank_valid[15][0])
    # print(sp_rank_valid[15][0])
