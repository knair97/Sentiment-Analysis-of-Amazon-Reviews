import numpy as np 
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Parse through the training data
print('Reading training data')
data = np.loadtxt('training_data.txt', skiprows = 1)
y_train = data[:, 0]
x_train = data[:, 1:]

# Parse through the testing data
print('Reading testing data')
x_test = np.loadtxt('test_data.txt', skiprows = 1)

# Create a validation set
x_val = x_train[:1000]
y_val = y_train[:1000]
x_train = x_train[1000:]
y_train = y_train[1000:]

val_err = []
train_err = []
depth = []
# Create the model for various maximum depths
for d in range(10, 35):
    print('Creating the model for maximum depth = %i' % d)
    clf = RandomForestClassifier(n_estimators = 200, max_depth=d, n_jobs=-1)
    print('Fitting the model')
    clf.fit(x_train, y_train)
    print('Max depth: ' + str(d))
    print('Training error: ')
    train_err.append(clf.score(x_train, y_train))
    print('Validation error: ')
    val_err.append(clf.score(x_val, y_val))
    depth.append(d)

# Plot the training and test error versus the maximum depth
plt.figure()
plt.plot(depth, train_err, label = 'Training Accuracy')
plt.plot(depth, val_err, label = 'Testing Accuracy')
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('forest_depth_vs_error.png')

val_err = []
train_err = []
leaf_nodes = []
# Create the model for various minimum number of samples for leaf nodes
for l in range(1, 10):
    print('Creating the model for minimum samples for leaf node = %i' % l)
    clf = RandomForestClassifier(n_estimators = 200, 
        min_samples_leaf=l, n_jobs=-1)
    print('Fitting the model')
    clf.fit(x_train, y_train)
    print('Training error: ')
    train_err.append(clf.score(x_train, y_train))
    print('Validation error: ')
    val_err.append(clf.score(x_val, y_val))
    leaf_nodes.append(l)

# Plot the training and testing error versus minimum number of samples for leaf
plt.figure()
plt.plot(leaf_nodes, train_err, label = 'Training Error')
plt.plot(leaf_nodes, val_err, label = 'Testing Error')
plt.xlabel('Minimum samples per Leaf Node')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('forest_leaf_nodes_vs_error.png')

# Write final predictions to a file
print('Writing predictions')
# Write the predictions to a file
with open('forest_submissions.txt', 'w') as f:
    f.write('Id,Prediction\n')
    it = 1
    for i in (x_test):
        pred = int(clf.predict(i.reshape(1, -1))[0])
        f.write('%d,%d\n' % (it, pred))
        it += 1
f.close()