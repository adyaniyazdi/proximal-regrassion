import random
import numpy as np
import math
# j = 7
# n = 50
#
# x = numpy.zeros((n,j))
# y = numpy.zeros((n,1))
# print(x)
# print(y)

num_groups = 5
group_size = 10
group_overlap = 3
groups=[]

for i in range(num_groups):
    group_start_index = i * (group_size - group_overlap)
    groups.append(list(range(group_start_index, group_start_index + group_size)))

print("groups: ", groups)

num_features = group_overlap + (num_groups * (group_size - group_overlap)) #J

b = np.zeros(num_features)
# print(list(range(num_features)))
# print(b)
# for i in range(num_features):
#     b[i]= i
# print(b)

for group in groups:
    # r = random.random()
    r = random.gauss(0,1)
    for member in group:
        b[member] +=r
for i in range(num_features//2, num_features):
    b[i] = 0.0
print("b", b)














