import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dir = "result_20221115_sorted.csv"
df = pd.read_csv(dir)
gmean_all = []
gmean_ini = []
gmean_dro = []
gmean_sta = []
for i in range(len(df)):
    gmean_all.append(df["gmean_bst"][i])
    gmean_ini.append(df["initial_pf"][i])
    gmean_dro.append(df["drop_pf"][i])
    gmean_sta.append(df["stable_pf"][i])
odasc = []
oob = []
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

for i in range(3,6):
    temp = []
    for j in range(14):
        temp.append(gmean_ini[i*14+j])
    odasc.append(temp)

x = np.arange(14)
total_width, n = 0.8, 4
width = total_width/n
x = x - (total_width-width)/2

# odasc_addcp_adp_total = [0.4180,0.5011,0.5416,0,0,0.4765,0.5984,0.4908,0,0,0,0.5041,0,0]
# odasc_addcp_adp_initial = [0.1054, 0.4961, 0.4286, 0,0,0.3395,0.1022,0.3306,0,0,0,0.5118,0,0]
# odasc_addcp_adp_drop = [0.3851, 0, 0.5774, 0,0, 0.5360,0,0.5202,0,0,0,0.2865,0,0,]
# odasc_addcp_adp_stable = [0.4223,0.5012,0.5478,0,0,0.4723,0.6214,0.4867,0,0,0,0.5072,0,0]
#
# oob_addcp_adp_total = [0.4292,0.5100, 0.5114, 0,0,0.5035,0.5420,0.5112,0,0,0,0.4940,0,0]
# oob_addcp_adp_initial = [0.0848,0.4762,0.4448,0,0,0.1995,0.0828,0.2164,0,0,0,0.4652,0,0]
# oob_addcp_adp_drop = [0.3191,0.5410,0.4936,0,0,0.5541,0.5991,0.5280,0,0,0,0.4864,0,0]
# oob_addcp_adp_stable = [0.4327,0.5032,0.5136,0,0,0.5008,0.5621,0.5045,0,0,0,0.4947,0,0]

plt.bar(x, odasc[0], width=width, label="oob")
plt.bar(x + width, odasc[1], width=width, label="oob_aio")
plt.bar(x + 2*width, odasc[2], width=width, label="oob_filtering")
plt.bar(x + 3*width, oob_addcp_adp_initial, width=width, label="oob_addcp_adp")

plt.ylabel("gmean")
plt.xlabel("dataset index")

my_x_ticks=np.arange(0, 14, 1)
plt.xticks(my_x_ticks)
my_y_ticks=np.arange(0, 1, 0.1)
plt.yticks(my_y_ticks)
plt.title("performance in initial periods about oob group")

plt.legend()
plt.show()
