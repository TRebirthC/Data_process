import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def draw_a_method(method, method_range, dataset_used, periods, gmeans, save):
    this_method = []

    for i in method_range:
        temp = []
        for j in dataset_used:
            temp.append(gmeans[i * len(dataset_used) + j])
        this_method.append(temp)

    x = np.arange(len(dataset_used))
    total_width, n = 0.8, 3
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.bar(x, this_method[0], width=width, label=method)
    plt.bar(x + width, this_method[1], width=width, label=method+"_aio")
    plt.bar(x + 2 * width, this_method[2], width=width, label=method+"_filtering")

    plt.ylabel("gmean")
    plt.xlabel("dataset index")

    my_x_ticks = np.arange(0, len(dataset_used), 1)
    plt.xticks(my_x_ticks)
    my_y_ticks = np.arange(0, 1, 0.1)
    plt.yticks(my_y_ticks)
    plt.title("performance in " + periods + " periods about " + method + " group")

    plt.legend()
    if save:
        plt.savefig("plot/"+method+"_"+periods)
    else:
        plt.show()
    plt.close()

def draw_compare_method(periods, gmeans, save):
    this_method = []

    for i in [1,4,7]:
        temp = []
        for j in dataset_used:
            temp.append(gmeans[i * len(dataset_used) + j])
        this_method.append(temp)

    x = np.arange(len(dataset_used))
    total_width, n = 0.8, 3
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.bar(x, this_method[0], width=width, label="odasc_aio")
    plt.bar(x + width, this_method[1], width=width, label="oob_aio")
    plt.bar(x + 2 * width, this_method[2], width=width, label="pbsa_aio")

    plt.ylabel("gmean")
    plt.xlabel("dataset index")

    my_x_ticks = np.arange(0, len(dataset_used), 1)
    plt.xticks(my_x_ticks)
    my_y_ticks = np.arange(0, 1, 0.1)
    plt.yticks(my_y_ticks)
    plt.title("performance in " + periods + " periods")

    plt.legend()
    if save:
        plt.savefig("plot/compare_"+periods)
    else:
        plt.show()
    plt.close()

def draw_all_compare_method(periods, gmeans, save):
    this_method = []

    for i in range(9):
        temp = []
        for j in dataset_used:
            temp.append(gmeans[i * len(dataset_used) + j])
        this_method.append(temp)

    x = np.arange(len(dataset_used))
    total_width, n = 0.8, 9
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.bar(x, this_method[0], width=width, label="odasc_")
    plt.bar(x + width, this_method[1], width=width, label="odasc_aio")
    plt.bar(x + 2 * width, this_method[2], width=width, label="odasc_filtering")
    plt.bar(x + 3 * width, this_method[3], width=width, label="oob")
    plt.bar(x + 4 * width, this_method[4], width=width, label="oob_aio")
    plt.bar(x + 5 * width, this_method[5], width=width, label="oob_filtering")
    plt.bar(x + 6 * width, this_method[6], width=width, label="pbsa")
    plt.bar(x + 7 * width, this_method[7], width=width, label="pbsa_aio")
    plt.bar(x + 8 * width, this_method[8], width=width, label="pbsa_filtering")

    plt.ylabel("gmean")
    plt.xlabel("dataset index")

    my_x_ticks = np.arange(0, len(dataset_used), 1)
    plt.xticks(my_x_ticks)
    my_y_ticks = np.arange(0, 1, 0.1)
    plt.yticks(my_y_ticks)
    plt.title("all performance in " + periods + " periods")

    plt.legend(ncol=3)
    if save:
        plt.savefig("plot/all_compare_"+periods)
    else:
        plt.show()
    plt.close()

dir = "result_20221124_sorted.csv"
df = pd.read_csv(dir)
gmean_all = []
gmean_ini = []
gmean_dro = []
gmean_sta = []
# 0 for total, 1 for initial, 2 for drop, 3 for stable
periods_used = [0,1,2,3]
dataset_used = range(14)
# 0 for odasc, 1 for oob, 2 for pbsa
method_used = [0, 1, 2]
# 0 for draw a method, 1 for draw compare methods, 2 for draw all compare methods
a_method = 2

for i in range(len(df)):
    gmean_all.append(df["gmean_bst"][i])
    gmean_ini.append(df["initial_pf"][i])
    gmean_dro.append(df["drop_pf"][i])
    gmean_sta.append(df["stable_pf"][i])

if a_method == 0:
    for each_method in method_used:
        if each_method == 0:
            method = "odasc"
            method_range = range(3)
        elif each_method == 1:
            method = "oob"
            method_range = range(3, 6)
        elif each_method == 2:
            method = "pbsa"
            method_range = range(6, 9)
        for each_period in periods_used:
            if each_period == 0:
                periods = "total"
                gmeans = gmean_all
            elif each_period == 1:
                periods = "initial"
                gmeans = gmean_ini
            elif each_period == 2:
                periods = "drop"
                gmeans = gmean_dro
            elif each_period == 3:
                periods = "stable"
                gmeans = gmean_sta
            draw_a_method(method, method_range, dataset_used, periods, gmeans, True)
elif a_method == 1:
    for each_period in periods_used:
        if each_period == 0:
            periods = "total"
            gmeans = gmean_all
        elif each_period == 1:
            periods = "initial"
            gmeans = gmean_ini
        elif each_period == 2:
            periods = "drop"
            gmeans = gmean_dro
        elif each_period == 3:
            periods = "stable"
            gmeans = gmean_sta
        draw_compare_method(periods, gmeans, True)
elif a_method == 2:
    for each_period in periods_used:
        if each_period == 0:
            periods = "total"
            gmeans = gmean_all
        elif each_period == 1:
            periods = "initial"
            gmeans = gmean_ini
        elif each_period == 2:
            periods = "drop"
            gmeans = gmean_dro
        elif each_period == 3:
            periods = "stable"
            gmeans = gmean_sta
        draw_all_compare_method(periods, gmeans, True)


# for i in range(len(df)):
#     gmean_all.append(df["gmean_bst"][i])
#     gmean_ini.append(df["initial_pf"][i])
#     gmean_dro.append(df["drop_pf"][i])
#     gmean_sta.append(df["stable_pf"][i])
# odasc = []
# oob = []
# pbsa = []
# for i in range(3):
#     temp = []
#     for j in range(14):
#         temp.append(gmean_sta[i*14+j])
#     odasc.append(temp)
#
# x = np.arange(14)
# total_width, n = 0.8, 3
# width = total_width/n
# x = x - (total_width-width)/2
#
# plt.bar(x, odasc[0], width=width, label="odasc")
# plt.bar(x + width, odasc[1], width=width, label="odasc_aio")
# plt.bar(x + 2*width, odasc[2], width=width, label="odasc_filtering")
#
# plt.ylabel("gmean")
# plt.xlabel("dataset index")
#
# my_x_ticks=np.arange(0, 14, 1)
# plt.xticks(my_x_ticks)
# my_y_ticks=np.arange(0, 1, 0.1)
# plt.yticks(my_y_ticks)
# plt.title("performance in stable periods about odasc group")
#
# plt.legend()
# plt.show()
#
# for i in range(3, 6):
#     temp = []
#     for j in range(14):
#         temp.append(gmean_sta[i*14+j])
#     oob.append(temp)
#
# x = np.arange(14)
# total_width, n = 0.8, 3
# width = total_width/n
# x = x - (total_width-width)/2
#
# plt.bar(x, oob[0], width=width, label="oob")
# plt.bar(x + width, oob[1], width=width, label="oob_aio")
# plt.bar(x + 2*width, oob[2], width=width, label="oob_filtering")
#
# plt.ylabel("gmean")
# plt.xlabel("dataset index")
#
# my_x_ticks=np.arange(0, 14, 1)
# plt.xticks(my_x_ticks)
# my_y_ticks=np.arange(0, 1, 0.1)
# plt.yticks(my_y_ticks)
# plt.title("performance in stable periods about oob group")
#
# plt.legend()
# plt.show()
#
# for i in range(6, 9):
#     temp = []
#     for j in range(14):
#         temp.append(gmean_sta[i*14+j])
#     pbsa.append(temp)
#
# x = np.arange(14)
# total_width, n = 0.8, 3
# width = total_width/n
# x = x - (total_width-width)/2
#
# plt.bar(x, pbsa[0], width=width, label="pbsa")
# plt.bar(x + width, pbsa[1], width=width, label="pbsa_aio")
# plt.bar(x + 2*width, pbsa[2], width=width, label="pbsa_filtering")
#
# plt.ylabel("gmean")
# plt.xlabel("dataset index")
#
# my_x_ticks=np.arange(0, 14, 1)
# plt.xticks(my_x_ticks)
# my_y_ticks=np.arange(0, 1, 0.1)
# plt.yticks(my_y_ticks)
# plt.title("performance in stable periods about pbsa group")
#
# plt.legend()
# plt.show()