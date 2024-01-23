import numpy as np
import sklearn 
import matplotlib.pyplot as plt

data = np.loadtxt('cricketers.cls.csv', delimiter=',')
plt.scatter(data[:,0], data[:,1], s=4, alpha=0.3, c=data[:,2], cmap='RdYlBu_r');

#

x = data[:,0:2]
y = data[:,2]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

from sklearn.svm import SVC

linear = SVC(kernel='linear')
linear.fit(x_train, y_train)

nw_data = np.asarray([[1896, 20],[1922, 20]])
linear.predict(nw_data)

from sklearn.metrics import accuracy_score

y_predicted = linear.predict(x_test)
print(accuracy_score(y_test, y_predicted))

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)

y_predicted = tree.predict(x_test)
accuracy_score(y_test, y_predicted)

from sklearn.metrics import roc_curve, auc

# The linear classifier doesn't produce class probabilities by default. We'll retrain it for probabilities.
linear = SVC(kernel='linear', probability=True)
linear.fit(x_train, y_train)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(15) # We set the number of neighbors to 15
knn.fit(x_train, y_train)

y_predicted = knn.predict(x_test)

# We'll need class probabilities from each of the classifiers
y_linear = linear.predict_proba(x_test)
y_tree  = tree.predict_proba(x_test)
y_knn   = knn.predict_proba(x_test)

# Compute the points on the curve
# We pass the probability of the second class (KIA) as the y_score
curve_linear = sklearn.metrics.roc_curve(y_test, y_linear[:, 1])
curve_tree   = sklearn.metrics.roc_curve(y_test, y_tree[:, 1])
curve_knn    = sklearn.metrics.roc_curve(y_test, y_knn[:, 1])

# Compute Area Under the Curve
auc_linear = auc(curve_linear[0], curve_linear[1])
auc_tree   = auc(curve_tree[0], curve_tree[1])
auc_knn    = auc(curve_knn[0], curve_knn[1])

plt.plot(curve_linear[0], curve_linear[1], label='linear (area = %0.2f)' % auc_linear)
plt.plot(curve_tree[0], curve_tree[1], label='tree (area = %0.2f)' % auc_tree)
plt.plot(curve_knn[0], curve_knn[1], label='knn (area = %0.2f)'% auc_knn)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')

plt.legend()

plt.show()