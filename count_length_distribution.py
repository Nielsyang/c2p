from collections import defaultdict
import pandas as pd

"""
count atoms length distribution
"""

# atom = open('D:\\atoms.txt', 'r')
# atom_data_dict = defaultdict(int)
# atom_lines = atom.readlines()

# for line in atom_lines:
# 	idx = line.split(',')[0]
# 	atom_data_dict[idx] += 1

# all_possible_length = atom_data_dict.values()
# max_length = max(all_possible_length)
# length_cnt_dict = defaultdict(int)
# start = 1

# while start <= max_length:
# 	for length in all_possible_length:
# 		if length == start:
# 			length_cnt_dict[start] += 1
# 	start += 1
# 	print('Deal with {}/{}'.format(start, max_length))

# out_file = pd.DataFrame()
# length_cnt = length_cnt_dict.items()
# out_file['length'] = list(map(lambda x:x[0], length_cnt))
# out_file['cnt'] = list(map(lambda x:x[1], length_cnt))
# out_file.to_csv('D:\\atoms_distribution.csv', index=False)


"""
count relations length distribution
"""

# relations = open('D:\\relations.txt', 'r')
# relation_data_dict = defaultdict(int)
# relation_lines = relations.readlines()

# for line in relation_lines:
# 	idx = line.split(',')[0]
# 	relation_data_dict[idx] += 1

# all_possible_length = relation_data_dict.values()
# max_length = max(all_possible_length)
# length_cnt_dict = defaultdict(int)
# start = 1

# while start <= max_length:
# 	for length in all_possible_length:
# 		if length == start:
# 			length_cnt_dict[start] += 1
# 	start += 1
# 	print('Deal with {}/{}'.format(start, max_length))

# out_file = pd.DataFrame()
# length_cnt = length_cnt_dict.items()
# out_file['length'] = list(map(lambda x:x[0], length_cnt))
# out_file['cnt'] = list(map(lambda x:x[1], length_cnt))
# out_file.to_csv('D:\\relations_distribution.csv', index=False)

"""
count protein sequence length distribution
"""

# protein = pd.read_csv('D:\\protein.txt')
# sequence = protein[protein.columns[1]]
# length_cnt_dict = defaultdict(int)
# data_len = len(sequence)
# start = 0

# for seq in sequence:
# 	length_cnt_dict[len(seq)] += 1
# 	print('Deal with {}/{}'.format(start, data_len))
# 	start += 1

# out_file = pd.DataFrame()
# length_cnt = length_cnt_dict.items()
# out_file['length'] = list(map(lambda x:x[0], length_cnt))
# out_file['cnt'] = list(map(lambda x:x[1], length_cnt))
# out_file.to_csv('D:\\protein_distribution.csv', index=False)

"""
count protein sequence length distribution
"""
# proteins = open('D:\\proteins.txt', 'r', encoding='utf-16')
# length_cnt_dict = defaultdict(int)
# lines = proteins.readlines()
# length_lines = len(lines)

# length_protein = 0
# for i in range(length_lines-1, -1, -1):
# 	if lines[i] == '\n':
# 		if length_protein > 0:
# 			length_cnt_dict[length_protein] += 1
# 			length_protein = 0
# 	else:
# 		length_protein += len(lines[i].split(',')[-1])-1
# 	# print('deal with {}/{}'.format(i,length_lines))

# out_file = pd.DataFrame()
# length_cnt = length_cnt_dict.items()

# print('number of all proteins: {}'.format(sum(length_cnt_dict.values())))

# length_cnt_sorted = sorted(length_cnt, key=lambda x:x[0])
# out_file['length'] = list(map(lambda x:x[0], length_cnt_sorted))
# out_file['cnt'] = list(map(lambda x:x[1], length_cnt_sorted))
# out_file.to_csv('D:\\protein_distribution.csv', index=False)

"""
plot distribution using matplotlib
"""
# import matplotlib.pyplot as plt

# df_protein = pd.read_csv('D:\\protein_distribution.csv')
# X = df_protein['length'].values
# Y = df_protein['cnt'].values
# plt.scatter(X, Y)
# plt.xlabel('length')
# plt.ylabel('count')
# plt.show()

"""
split pos.txt to 5 small files prevent memoryoverflow
"""
# pos = open('D:\\train.csv', 'r')
# lines = pos.readlines()
# length = len(lines)
# per = length//2

# pos1_fp = open('E:\\pos1.txt', 'w')
# pos2_fp = open('E:\\pos2.txt', 'w')
# # pos3_fp = open('D:\\pos3.txt', 'w')
# # pos4_fp = open('D:\\pos4.txt', 'w')
# # pos5_fp = open('D:\\pos5.txt', 'w')

# pos1_fp.writelines(lines[:per])
# pos2_fp.writelines(lines[per:])
# pos3_fp.writelines(lines[2*per:3*per])
# pos4_fp.writelines(lines[3*per:4*per])
# pos5_fp.writelines(lines[4*per:])

"""
count all overlap of generated negative samples
"""
# import gc
# train1 = open('D:\\train_data1.txt', 'r')
# train2 = open('D:\\train_data2.txt', 'r')
# train3 = open('D:\\train_data3.txt', 'r')
# train4 = open('D:\\train_data4.txt', 'r')
# train5 = open('D:\\train_data5.txt', 'r')

# negative = 0

# lines = train1.readlines()
# for line in lines:
# 	if line[-2] == '0':
# 		negative += 1
# del lines
# gc.collect()
# train1.close()

# lines = train2.readlines()
# for line in lines:
# 	if line[-2] == '0':
# 		negative += 1
# del lines
# gc.collect()
# train2.close()
# lines = train3.readlines()
# for line in lines:
# 	if line[-2] == '0':
# 		negative += 1
# del lines
# gc.collect()
# train3.close()
# lines = train4.readlines()
# for line in lines:
# 	if line[-2] == '0':
# 		negative += 1
# del lines
# gc.collect()
# train4.close()
# lines = train5.readlines()
# for line in lines:
# 	if line[-2] == '0':
# 		negative += 1
# del lines
# gc.collect()

# print(negative)
# import gc
# from bloom_filter import BloomFilter
# bloom = BloomFilter(max_elements=1000000, error_rate=0.0000001)
# del bloom
# gc.collect()
# print('ok')

# import gc
# import numpy as np

# pos = open('D:\\train.csv', 'r')

# train1 = open('E:\\train1.txt', 'w')
# train2 = open('E:\\train2.txt', 'w')
# train3 = open('E:\\train3.txt', 'w')
# train4 = open('E:\\train4.txt', 'w')
# train5 = open('E:\\train5.txt', 'w')

# pos_data = pos.readlines()
# length_pos = len(pos_data)
# print('number of pos data: {}'.format(length_pos))

# compound_length = 2200
# protein_length = 1400
# proteins = list(map(lambda x:','.join(x.split(',')[compound_length:-1]), pos_data))
# compounds = list(map(lambda x:','.join(x.split(',')[:compound_length]), pos_data))

# assert len(proteins) == len(compounds)
# assert protein_length == len(proteins[0].split(','))
# assert compound_length == len(compounds[0].split(','))

# neg_data = []
# np.random.shuffle(compounds)
# np.random.shuffle(proteins)
# neg_data.extend(zip(compounds, proteins))

# assert len(neg_data) == length_pos
# print('shuffle success')

# del compounds
# del proteins
# gc.collect()

# overlap = 0
# print('start detecting.....')

# last_comma = -1                         # last comma's index
# while pos_data[0][last_comma] != ',':
#     last_comma -= 1

# negs = []
# idx = 0
# for c,p in neg_data:
# 	neg = ','.join((c,p))
# 	for positive in pos_data:
# 		if neg == positive[:last_comma]:
# 			overlap += 1
# 		else:
# 			negs.append(','.join((neg, '0', '\n')))
# 	print('detected neg: {}/{}'.format(idx+1, length_pos))
# 	print('overlap: {}'.format(overlap))
# 	idx += 1

# print('detect over, overlap: {}'.format(overlap))

# length_neg = len(negs)
# per_neg = length_neg//5
# per_pos = length_pos//5

# train1_data = pos_data[:per_pos] + negs[:per_neg]
# train2_data = pos_data[per_pos:2*per_pos] + negs[per_neg:2*per_neg]
# train3_data = pos_data[2*per_pos:3*per_pos] + negs[2*per_neg:3*per_neg]
# train4_data = pos_data[3*per_pos:4*per_pos] + negs[3*per_neg:4*per_neg]
# train5_data = pos_data[4*per_pos:] + negs[4*per_neg:]

# np.random.shuffle(train1_data)
# np.random.shuffle(train2_data)
# np.random.shuffle(train3_data)
# np.random.shuffle(train4_data)
# np.random.shuffle(train5_data)

# train1.writelines(train1_data)
# train2.writelines(train2_data)
# train3.writelines(train3_data)
# train4.writelines(train4_data)
# train5.writelines(train5_data)
import time
a = [str(i) for i in range(1000000)]
b = [str(i) for i in range(1000000)]
c = ','.join(a)
d = ','.join(b)

stime = time.time()
e = (a==b)
print(time.time()-stime)
stime = time.time()
f = (c==d)
print(time.time()-stime)
