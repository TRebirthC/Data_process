import numpy as np
import scipy
import pandas as pd

a = []
a.append(np.asarray([1,2,3]))
a.append(np.asarray([2, 2, 3]))
# a.append([3, 2, 3])
for i in range(len(a)):
    if (a[i] == np.asarray([1,2,3])).all():
        print("yes")
        a.pop(i)
        break

# index = np.where(a==np.asarray([1,2,3]))
# print(index)
print(a)
# print(a)
# print(a.__contains__(np.asarray([1,2,3])))
# a.remove(np.asarray([1,2,3]))
# print(a)
# print(a.__contains__([2,2,3]))

t = np.ones(10)*5
print(t)


target = t == 5
print(target)

target2 = target[:2]
print(target2)

a = [0,0,1]
b = [0,1,0]
print(np.linalg.norm(np.array(a)-np.array(b)))

c = np.array(range(40))
print(c[1:13])
print(c[13:25])
print(c[25:37])

q = [1, 2, 3, 4, 5]
w = [5, 4, 3, 2, 1]
e = [5, 1, 2, 3, 4]
r = [1, 2, 4, 3, 5]

# r1, r2 = scipy.stats.kendalltau(q, w)
# print(r1)
# print(r2)
# r1, r2 = scipy.stats.kendalltau(q, e)
# print(r1)
# print(r2)
# r1, r2 = scipy.stats.kendalltau(q, r)
# print(r1)
# print(r2)
#
# gt_df = pd.read_csv("ground_truth_compare/20221206_D_similarity.csv")
# print(gt_df["1"])
r.remove(3)
print(r)
a = r.pop(1)
print(r)
print(a)