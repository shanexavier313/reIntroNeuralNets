import numpy as np
import matplotlib.pyplot as plt 

np.random.seed(45)

class1 = np.random.randn(700, 2) + np.array([0, -3])
class2 = np.random.randn(700, 2) + np.array([3, 3])
class3 = np.random.randn(700, 2) + np.array([-3, 3])

feature_set = np.vstack([class1, class2, class3])

labels = np.array([0]*700 + [1]*700 + [2]*700)

one_hot_labels = np.zeros((2100, 3))

for i in range (2100):
    one_hot_labels[i, labels[i]] = 1


plt.figure(figsize=(10, 7))
plt.scatter(feature_set[:, 0], feature_set[:, 1], c=labels, cmap='plasma', s=100, alpha=0.5)
plt.show()

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1-sigmoid(x))


def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


instances = feature_set.shape[0]
attributes = feature_set.shape[1]
hidden_nodes = 4
output_labels = 3

wh = np.random.rand(attributes, hidden_nodes)
bh = np.random.randn(hidden_nodes)

wo = np.random.rand(hidden_nodes, output_labels)
bo = np.random.randn(output_labels)
lr = 10e-4

error_cost = []

for epoch in range(50000):
######## Feed-forwwards

    # phase 1
    zh = np.dot(feature_set, wh) + bh
    ah = sigmoid(zh)

    # Phase2
    zo = np.dot(ah, wo) + bo
    ao = softmax(zo)

######## Back Propogation
    dcost_dzo = ao - one_hot_labels
    dzo_dwo = ah

    dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

    dcost_bo = dcost_dzo

    # Phase 2

    dzo_dah = wo
    dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
    dah_dzh = sigmoid_der(zh)
    dzh_dwh = feature_set
    dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

    dcost_bh = dcost_dah * dah_dzh

    # update weights
    wh -= lr * dcost_wh
    bh -= lr * dcost_bh.sum(axis=0)

    wo -= lr * dcost_wo
    bo -= lr * dcost_bo.sum(axis=0)

    if epoch % 200 == 0:
        loss = np.sum(-one_hot_labels * np.log(ao))
        print('Loss function value: ', loss)
        error_cost.append(loss)


