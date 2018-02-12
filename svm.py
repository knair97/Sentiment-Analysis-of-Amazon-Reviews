import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

print('Reading training data')
data = np.loadtxt('training_data.txt', skiprows = 1)
y_train = data[:, 0]
x_train = data[:, 1:]

# Parse through the testing data
print('Reading testing data')
test = np.loadtxt('test_data.txt', skiprows = 1)

# Create a validation set
x_test = x_train[:1000]
y_test = y_train[:1000]

x_train = x_train[1000:]
y_train = y_train[1000:]

# Transform the training data
ext = TfidfTransformer()
ext.fit(x_train)
x_train = ext.transform(x_train)

# Transform test data
x_test = ext.transform(x_test)
test = ext.transform(test)

test_err = []
train_err = []
c_val = []
# Running SVM for various values of C
for c in [0.001, 0.01, 0.1, 1, 10, 100]:
    print('Running SVM for C = %g' % c)
    clf = SVC(C = c, gamma = 1, kernel='rbf')
    clf.fit(x_train, y_train)
    print('Evaluating scores')
    train_err.append(clf.score(x_train, y_train))
    test_err.append(clf.score(x_test, y_test))
    c_val.append(c)

# Plot the data
plt.plot(c_val, train_err, label = 'Training Accuracy')
plt.plot(c_val, test_err, label = 'Testing Accuracy')
plt.legend()
plt.xlabel('C value')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.savefig('svm_c.png')


# Write the predictions to a file
print('Writing predictions')
preds = clf.predict(test)
# Write the predictions to a file
with open('svm_submissions.txt', 'w') as f:
    f.write('Id,Prediction\n')
    it = 1
    for i in preds:
        pred = int(i)
        f.write('%d,%d\n' % (it, pred))
        it += 1
f.close()
