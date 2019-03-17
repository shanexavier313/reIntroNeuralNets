from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
feature_set, labels = datasets.make_moons(100, noise=0.10)
plt.figure(figsize=(10, 7))
plt.scatter(feature_set[:, 0], feature_set[:, 1], c=labels, cmap=plt.cm.winter)

labels = labels.reshape(100, 1)


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1-sigmoid(x))


np.random.seed(32)
wh = np.random.rand(len(feature_set[0]), 4)
wo = np.random.rand(4, 1)
error_out = 0

## optimizing learning rate
# bases = np.repeat(10, 3)
# exponents_1 = -(np.random.rand(3) + 3)
# exponents_2 = -(np.random.rand(3) + 2)

# first test of learning rate
# lr = [1, 0.5, 0.1, 0.01, 0.001, 0.0001]

# second test of learning rate
# lr = np.power(bases, exponents_1).tolist() + np.power(bases, exponents_2).tolist()

# third test of learning rate
# lr = [.005, .006, .007, .008, .009, .01]

# fourth test of learning rate
# lr = [.01, .03, .05, .07, .09, .11]

# best learning rate:
lr = .03
results = []

print(lr)


# for x in range(0, 6):
#     lr_x = lr[x]

for epoch in range(200000):
    # feedforward
    zh = np.dot(feature_set, wh)
    ah = sigmoid(zh)

    zo = np.dot(ah, wo)
    ao = sigmoid(zo)

    # Phase1 ==========================
    error_out = ((1/2) * (np.power((ao - labels), 2)))

    dcost_dao = ao - labels
    dao_dzo = sigmoid_der(zo)
    dzo_dwo = ah

    dcost_wo = np.dot(dzo_dwo.T, dcost_dao * dao_dzo)

    # Phase 2 ===========================
    dcost_dzo = dcost_dao * dao_dzo
    dzo_dah = wo
    dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
    dah_dzh = sigmoid_der(zh)
    dzh_dwh = feature_set
    dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

    # Update Weights ====================
    wh -= lr * dcost_wh
    wo -= lr * dcost_wo

print(error_out.sum())

#print(results[x])

# Plots to identify the best learning rate
# plt.plot(lr, results)
# plt.scatter(lr, results)
# plt.xlabel("Learning Rate")
# plt.ylabel("Mean Squared Error")
# plt.show()
