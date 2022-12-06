import numpy as np

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
