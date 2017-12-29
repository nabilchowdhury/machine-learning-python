import math
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import random
import warnings
from matplotlib import style
from collections import Counter

# Euclidean dist = sqrt(sum from 1 to n of (q_i - p_i)^2)
def euclidean_dist(p, q):
    return math.sqrt( sum( [ ( q[i] - p[i] )**2 for i in range(len(p)) ] ) )


'''
KNN 1 (Not used)
'''
def knn(examples, label, new, k):
    distances = [ ( euclidean_dist(examples[i], new), label[i] ) for i in range(len(examples)) ]
    distances.sort(key=lambda x: x[0])

    class_dict = dict()
    for i in range(k):
        class_dict[distances[i][1]] = class_dict.get(distances[i][1], 0) + 1

    print(class_dict)
    return max(class_dict, key=class_dict.get)


'''
KNN 2 (Used)
'''
# dataset = {'k': [[1, 2],[2, 3],[3, 1]], 'r': [[6, 5],[7, 7],[8, 6]]}
# new_features = [5, 7]

# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1], s=100)
# plt.show()

def knn2(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups.')
    distances = []
    for group in data:
        for feature in data[group]:
            euclidean_distance = np.linalg.norm(np.array(feature) - np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k

    return vote_result, confidence

# Read in the data and process it for KNN
df = pd.read_csv('../../datasets/breast-cancer-wisconsin.data.txt')
df.drop(['id'], 1, inplace=True)
df.replace('?', -99999, inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

# Do KNN
for group in test_set:
    for data in test_set[group]:
        vote, confidence = knn2(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1

print('Accuracy:', correct/total)