import numpy as np 
import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, \
    BatchNormalization, LSTM, Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers
import matplotlib.pyplot as plt

# Parse through the training data
print('Reading training data')
data = np.loadtxt('training_data.txt', skiprows = 1)
y_train = data[:, 0]
x_train = data[:, 1:]

# Parse through the testing data
print('Reading testing data')
test = np.loadtxt('test_data.txt', skiprows = 1)

# Create a validation set
x_test = x_train[-1000:]
y_test = y_train[-1000:]

x_train = x_train[:-1000]
y_train = y_train[:-1000]

val_err = []
train_err = []
drop = []

# Find best amount of dropout regularization
for i in np.arange(0, 1, 0.1):
    # Create the model
    model = Sequential()
    # Add the layers
    model.add(Dense(1000, activation='relu', input_dim=1000))
    model.add(Dropout(i))
    model.add(Dense(800, activation='relu'))
    model.add(Dropout(i))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(i))
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, 
        metrics=['accuracy'])
    # Fit the data
    model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=20, 
        batch_size=128)
    # Evaluate the model
    score = model.evaluate(x_test, y_test, batch_size=128)
    score_train = model.evaluate(x_train, y_train, batch_size=128)

    print('Val accuracy:', score[1])

    val_err.append(score[1])
    train_err.append(score_train[1])
    drop.append(i)

# Create the plot of training error and testing error versus the dropout rate
plt.plot(drop, val_err, label = 'Testing Error')
plt.plot(drop, train_err, label = 'Training Error')
plt.xlabel('Dropout percentage')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('neural_net_dropout.png')
print('Test accuracy = %g maximized at dropout percentage = %g' \
    % (val_err[np.argmax(val_err)], drop[np.argmax(val_err)]))

# Write final results to text file
preds = model.predict_classes(test)
print('Writing predictions')
# Write the predictions to a file
with open('neural_net_pred.txt', 'w') as f:
    f.write('Id,Prediction\n')
    it = 1
    for i in preds:
        pred = int(i[0])
        f.write('%d,%d\n' % (it, pred))
        it += 1
f.close()