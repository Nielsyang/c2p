# from bloom_filter import BloomFilter
# import gc
# import os
# import numpy as np

"""
obtain all compounds(~55w) and all proteins(~6000)

"""

# pos_fp = open('E:\\pos.csv', 'r')
# compounds = set()
# proteins = set()

# length_compound = 2200
# length_protein = 1400
# pos = pos_fp.readlines()
# idx = 0
# len_pos = len(pos)
# for line in pos:
#     d = line.split(',')[2:-1]   # drop label\
#     assert len(d) == length_compound+length_protein
#     c = ','.join(d[:length_compound])
#     p = ','.join(d[length_compound:])
#     compounds.add(c)
#     proteins.add(p)
#     print('*******{}/{}*******'.format(idx+1, len_pos))
#     idx += 1

# print(len(compounds), len(proteins))
# c_fp = open('E:\\compounds.txt', 'w')
# p_fp = open('E:\\proteins.txt', 'w')

# c_lines = list(map(lambda x:x+'\n', compounds))
# p_lines = list(map(lambda x:x+'\n', proteins))
# c_fp.writelines(c_lines)
# p_fp.writelines(p_lines)


"""
mapping pos to id_pos

"""

# import gc
# import numpy as np
# from bloom_filter import BloomFilter
# import time

# pos_fp = open('E:\\pos.csv', 'r')
# c_fp = open('E:\\compounds.txt', 'r')
# p_fp = open('E:\\proteins.txt', 'r')

# id_pos_fp = open('E:\\id_pos.txt', 'w')
# # pos = pos_fp.readlines()
# c = c_fp.readlines()
# p = p_fp.readlines()
# len_c = len(c)
# len_p = len(p)
# # pos_fp.close()
# c_fp.close()
# p_fp.close()
# id_pos = []

# c_dict = dict()
# p_dict = dict()

# for i in range(len_c):
#     c_dict[c[i][:-1]] = i
# for i in range(len_p):
#     p_dict[p[i][:-1]] = i
# print('construct dict success')

# del c
# del p 
# gc.collect()

# pos = pos_fp.readlines()
# pos_fp.close()

# len_pos = len(pos)
# len_compound = 2200
# len_protein = 1400
# idx = 1
# for line in pos:
#     line_c = ','.join(line.split(',')[2:len_compound+2])
#     line_p = ','.join(line.split(',')[len_compound+2:-1])
#     c_idx = c_dict[line_c]
#     p_idx = p_dict[line_p]
#     id_pos.append(','.join([str(c_idx), str(p_idx)])+'\n')
#     print('{}/{}'.format(idx, len_pos))
#     idx += 1

# del pos
# gc.collect()

# id_pos_fp.writelines(id_pos)
# id_pos_fp.close()


"""
add label after id_pos and id_neg_nooverlap

"""

# id_pos_fp = open('E:\\id_pos.txt', 'r')
# id_neg_fp = open('E:\\id_neg_nooverlap.txt', 'r')
# labeled_id_pos_fp = open('E:\\labeled_id_pos.txt', 'w')
# labeled_id_neg_fp = open('E:\\labeled_id_neg_nooverlap.txt', 'w')
# id_neg = id_neg_fp.readlines()
# id_pos = id_pos_fp.readlines()

# labeled_id_neg = []
# labeled_id_pos = []

# id_neg_fp.close()
# id_pos_fp.close()

# for line in id_pos:
#     labeled_id_pos.append(','.join((line[:-1], '1\n')))
# for line in id_neg:
#     labeled_id_neg.append(','.join((line[:-1], '0\n')))

# labeled_id_neg_fp.writelines(labeled_id_neg)
# labeled_id_pos_fp.writelines(labeled_id_pos)


"""
split id_pos to id_pos1, id_pos2..., id_pos5
split id_neg_nooverlap to id_neg1,..., id_neg15
and pairing them

"""

# import numpy as np
# import gc

# id_pos_fp = open('E:\\labeled_id_pos.txt', 'r')
# id_neg_fp = open('E:\\labeled_id_neg_nooverlap.txt', 'r')

# # train1 = [(1,1), (2,2), (3,3), (4,4), (5,5)]
# # train2 = [(1,6), (2,7), (3,8), (4,9), (5,10)]
# # train3 = [(1,11), (2,12), (3,13), (4,14), (5,15)]

# id_pos = id_pos_fp.readlines()
# id_neg = id_neg_fp.readlines()

# len_pos = len(id_pos)
# len_neg = len(id_neg)

# id_pos_fp.close()
# id_neg_fp.close()

# np.random.shuffle(id_neg)
# np.random.shuffle(id_pos)

# slice_pos_cnt = 5
# slice_neg_cnt = 15
# per_pos = len_pos//slice_pos_cnt
# per_neg = len_neg//slice_neg_cnt

# for i in range(15):
#     train_fp = open('E:\\id_train{}_{}.txt'.format(i//5+1, i%5), 'w')
#     train = id_pos[(i%5)*per_pos:(i%5+1)*per_pos] + id_neg[i*per_neg:(i+1)*per_neg]
#     train_fp.writelines(train)
#     train_fp.close()
#     del train
#     gc.collect()
#     print('{}/15'.format(i))

"""
generate id_neg no detecting overlap
(use SQL to detect overlap)

"""
# import numpy as np

# c_fp = open('E:\\compounds.txt', 'r')
# p_fp = open('E:\\proteins.txt', 'r')
# c = c_fp.readlines()
# p = p_fp.readlines()
# len_c = len(c)
# len_p = len(p)

# len_pos = 1224408

# id_neg = []
# id_neg_fp = open('E:\\id_neg.txt', 'w')
# for i in range(len_pos*3):
#     cur_id_neg = ','.join((str(np.random.randint(len_c)), str(np.random.randint(len_p)))) + '\n'
#     id_neg.append(cur_id_neg)
#     print('{}/{}'.format(i, len_pos*3))

# id_neg_fp.writelines(id_neg)



"""
from id_neg_nooverlap generate neg

"""

# import numpy as np
# len_pos = 1224408
# id_neg_nooverlap_fp = open('E:\\id_neg_nooverlap.txt', 'r')
# id_neg_nooverlap = id_neg_nooverlap_fp.readlines()
# np.random.shuffle(id_neg_nooverlap)

# id_neg = id_neg_nooverlap[:len_pos]

# c_fp = open('E:\\compounds.txt', 'r')
# p_fp = open('E:\\proteins.txt', 'r')
# c = c_fp.readlines()
# p = p_fp.readlines()

# c_fp.close()
# p_fp.close()

# inv_c_dict = {}
# inv_p_dict = {}

# for i in range(len(c)):
#     inv_c_dict[i] = c[i][:-1]

# for i in range(len(p)):
#     inv_p_dict[i] = p[i][:-1]

# neg = []

# for line in id_neg:
#     cid = int(line[:-1].split(',')[0])
#     pid = int(line[:-1].split(',')[1])
#     neg.append(','.join([inv_c_dict[cid], inv_p_dict[pid]]) + '\n')
# neg_fp = open('E:\\neg.txt', 'w')
# neg_fp.writelines(neg)


"""
from train{}_{}(id) generate train

"""
# import gc

# c_fp = open('E:\\compounds.txt', 'r')
# p_fp = open('E:\\proteins.txt', 'r')
# c = c_fp.readlines()
# p = p_fp.readlines()

# c_fp.close()
# p_fp.close()

# inv_c_dict = {}
# inv_p_dict = {}

# for i in range(len(c)):
#     inv_c_dict[i] = c[i][:-1]

# for i in range(len(p)):
#     inv_p_dict[i] = p[i][:-1]

# for i in range(1,4):
#     for j in range(5):
#         id_train_fp = open('E:\\id_train{}_{}.txt'.format(i,j), 'r')
#         train_fp = open('E:\\train{}_{}.txt'.format(i,j), 'w')
#         train = []
#         id_train = id_train_fp.readlines()
#         for line in id_train:
#             cid = int(line.split(',')[0])
#             pid = int(line.split(',')[1])
#             label = line[-2]
#             assert label in ['0', '1']
#             train.append(','.join([inv_c_dict[cid], inv_p_dict[pid]]) + ',' + label + '\n')
        
#         np.random.shuffle(train)
#         train_fp.writelines(train)
#         print('{}/15'.format(i*j))
#         id_train_fp.close()
#         train_fp.close()
#         del id_train
#         del train
#         gc.collect()

"""
drop pos data 1st and 2nd id

"""

# pos_fp = open('E:\\pos.csv', 'r')
# new_pos_fp = open('E:\\new_pos.txt', 'w')
# pos = pos_fp.readlines()
# new_pos_fp.writelines(list(map(lambda x:','.join(x.split(',')[2:]), pos)))

"""
test random function

"""
# import numpy as np
# import matplotlib.pyplot as plt

# for i in range(10):
#     X1 = np.random.randint(-275000, 275000, size=1000000)
#     Y1 = np.random.randint(-3000, 3000, size=1000000)

#     X = np.random.randint(-275000, 275000, size=1000000)
#     Y = np.random.randint(-3000, 3000, size=1000000)

#     s1 = set(zip(X1, Y1))
#     s2 = set(zip(X, Y))
#     print(len(s1 & s2))
# first = 0
# second = 0
# third = 0
# fourth = 0
# for x,y in zip(X, Y):
#         if x >0:
#             if y > 0:
#                 first += 1
#             else:
#                 fourth += 1
#         else:
#             if y > 0:
#                 second += 1
#             else:
#                 third += 1
# print(first, second, third, fourth)
# plt.scatter(X,Y)
# plt.show()

"""
shuffle train data

"""
# import gc
# import numpy as np

# for i in range(1,4):
#     for j in range(5):
#         train_fp = open('E:\\train{}_{}.txt'.format(i, j), 'r')
#         train = train_fp.readlines()
#         np.random.shuffle(train)
#         train_fp.close()
#         train_fp = open('E:\\train{}_{}.txt'.format(i, j), 'w')
#         train_fp.writelines(train)
#         train_fp.close()
#         del train
#         gc.collect()


"""
gain small data (1000) to test NN and data is or not correct

"""
# import gc

# train_fp = open('E:\\c2p_newdata\\test_train\\train.txt', 'w')
# eval_fp = open('E:\\c2p_newdata\\test_eval\\eval.txt', 'w')

# train = []
# eval = []

# for i in range(5):
#     train_large_fp = open('E:\\train3_{}.txt'.format(i), 'r')
#     train_large = train_large_fp.readlines()
#     train_large_fp.close()
#     train.extend(train_large[:200])
#     eval.extend(train_large[200:300])
#     del train_large
#     gc.collect()

# train_fp.writelines(train)
# eval_fp.writelines(eval)