from sklearn import neighbors, datasets
import numpy as np
import matplotlib.pyplot as plt

n_neighbours = 100

#load data
img = np.genfromtxt('data\\images', delimiter=',')
test = img[6000:7000]
img = img[:6000]
labels = np.genfromtxt('data\\labels.csv', delimiter=',')
test_labels = labels[6000:7000]
labels = labels[:6000]

#run classifier
knn = neighbors.KNeighborsClassifier(n_neighbours, weights='distance')
knn.fit(img,labels)
predicted = knn.predict(test)

#plot results
plt.scatter(test_labels,predicted)
plt.ylabel('True labels')
plt.xlabel('Predicted labels')
plt.show()






