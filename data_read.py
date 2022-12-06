import pandas as pd
import numpy as np
import time
def strftime(timestamp, format_string='%Y-%m-%d %H:%M:%S'):
    return time.strftime(format_string, time.localtime(timestamp))


def strptime(string, format_string='%Y-%m-%d %H:%M:%S'):
    return time.mktime(time.strptime(string, format_string))

df = pd.DataFrame(pd.read_csv('dataset/openstack.csv'))
df.sort_values("author_date", inplace=True)
df = df.reset_index(drop=True)
print(df.shape)
print(df.info)
print(df.columns)

df2 = pd.DataFrame(pd.read_csv('dataset/nova_vld_st.csv'))

print(df2.columns)

#
# s = 0
# e = 0
# a = 0
# b = 0
# for i in range(len(df["author_date"])):
#     if df["author_date"][i]>1309449540 and s == 0:
#         s = 1
#         a = i
#     if df["author_date"][i] > 1393603200 and e == 0:
#         e = 1
#         b = i
# # df = pd.DataFrame(pd.read_csv('dataset/qt.csv'))
# # print(df.shape)
# # print(df.info)
# # print(df.columns)
# print(b-a)
# c = 0
# bug = df["bugcount"]
# for i in range(a,b):
#     if bug[i] >0:
#         c = c + 1
# print(c)
#
# print(strptime('2011-06-30 23:59:00'))
# print(strptime('2014-03-01 00:00:00'))