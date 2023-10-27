import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.decomposition import PCA
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model, Sequential
import tensorflow as tf

from prettytable import PrettyTable
import time


### Collect Data
mnist_images = np.concatenate((mnist.train_images(), mnist.test_images()))
mnist_labels = np.concatenate((mnist.train_labels(), mnist.test_labels()))

### Preprocessing
def train_test_split(input, output, split_ratio=0.2, val_split=False, val_ratio=None):
    # randomly shuffle data
    data_indices = np.arange(np.shape(input)[0])
    np.random.shuffle(data_indices)
    input = input[data_indices, :, :]
    output = output[data_indices]

    number_of_observations = np.shape(input)[0]
    # train-test split
    if val_split:
        # define windows of split
        split_window_1 = int((1 - split_ratio - val_ratio) * number_of_observations)
        split_window_2 = split_window_1 + int(val_ratio * number_of_observations)

        # train data
        X_train = input[:split_window_1, :, :]
        Y_train = output[:split_window_1, ]

        # test data
        X_test = input[split_window_1:split_window_2, :, :]
        Y_test = output[split_window_1:split_window_2, ]

        # validation data
        X_val = input[split_window_2:, :, :]
        Y_val = output[split_window_2:, ]

        return X_train, Y_train, X_test, Y_test, X_val, Y_val

    else:
        # define windows of split
        split_window = int((1 - split_ratio) * number_of_observations)

        # train data
        X_train = input[:split_window, :, :]
        Y_train = output[:split_window, ]

        # test data
        X_test = input[split_window:, :, :]
        Y_test = output[split_window:, ]

        X_val = NoneY_val = None

        return X_train, Y_train, X_val, Y_val, X_test, Y_test


def MinMaxNormalize(input, output=None):
    min_val = np.min(input)
    max_val = np.max(input)
    input = (input - min_val) / (max_val - min_val)
    return input

def reduce_dimensions(x_train):

    # reduce dimensions
    pca = PCA(n_components=100)
    x_train = pca.fit_transform(x_train)

    return x_train

def preprocess(X, pca=False):
    # Apply PCA
    input_shape = np.shape(X)
    X = np.reshape(X, (input_shape[0], input_shape[1] * input_shape[2]))
    if pca:
        X = reduce_dimensions(X)
    else:
        pass
    X = MinMaxNormalize(X)
    return X

### Learning Methods
def RandomForestLearner(X_train, Y_train):
    # Build Model
    Tree_model = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    # Train
    Tree_model.fit(X_train, Y_train)

    return Tree_model


def SVMLearner(x_train, y_train):

    # Build Model
    SVM_model = LinearSVC(dual=False)

    # Train
    SVM_model.fit(x_train, y_train)

    return SVM_model


def MLP_learner(x_train, y_train):

    # Convert to Categorical Features
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    # Build Model
    num_features = np.shape(x_train)[1]
    In = Input(shape=(num_features,))
    Hidden1 = Dense(num_features * 2, activation='softmax')(In)
    Hidden2 = Dense(num_features, activation='softmax')(Hidden1)
    Out = Dense(10, activation='softmax')(Hidden2)
    MLP_model = Model(inputs=[In], outputs=[Out])
    MLP_model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])

    # print('Model Summary: \n')
    # print(MLP_model.summary())

    # Train
    history = MLP_model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=0,
                            validation_split=0.05)

    return MLP_model, history


def CNN_learner(x_train, y_train):
    # Convert to Categorical Features
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    # Build Model
    CNN_model = Sequential()
    CNN_model.add(Conv2D(64, kernel_size=3, activation="softmax", input_shape=(28, 28, 1)))
    CNN_model.add(Conv2D(32, kernel_size=3, activation="softmax"))
    CNN_model.add(Flatten())
    CNN_model.add(Dense(10, activation="softmax"))
    CNN_model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
    # print('Model Summary: \n')
    # print(CNN_model.summary())

    # Train
    history = CNN_model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=0,
                            validation_split=0.05)

    return CNN_model, history


def multiClassPostProcess(y):
    y_shape = np.shape(y)
    y_new = np.zeros(np.shape(y)[0])

    for i in range(np.shape(y)[0]):
        y_new[i] = np.argmax(y[i, :])

    return y_new


### ====================================================== Training
# Split Data
[X_train, Y_train, X_val, Y_val,  X_test, Y_test] = train_test_split(mnist_images, mnist_labels, 0.2, val_split=True, val_ratio=0.1)


# Plot a sample image
# since train_test_split randomly shuffles the observations
# every time this cell is executed a different sample will be shown
sample_image = X_train[0, :, :]
print("Digit")
plt.figure()
plt.imshow(sample_image)
plt.title("Sample Digit")
plt.show()

# plot class distribution
f, axs = plt.subplots(1, 3, tight_layout = True)
ax1, ax2, ax3 = axs
ax1.hist(Y_train, edgecolor='black')
ax1.set_xlabel("Digits")
ax1.set_ylabel("Count")
ax1.set_title("Train")
ax2.hist(Y_val, edgecolor='black')
ax2.set_xlabel("Digits")
ax2.set_ylabel("Count")
ax2.set_title("Validation")
ax3.hist(Y_test, edgecolor='black')
ax3.set_xlabel("Digits")
ax3.set_ylabel("Count")
ax3.set_title("Test")

# Preprocess
X_train_ = preprocess(X_train)
X_val_ = preprocess(X_val)
X_test_ = preprocess(X_test)


# Train models
print("Training...\n")
runtime_table = PrettyTable(["Model", "Training Time (seconds)"])

start = time.time()
tree_model = RandomForestLearner(X_train_, Y_train)
end = time.time()
runtime_table.add_row(["Random Forest", round(end-start, 2)])
start = time.time()
svm_model = SVMLearner(X_train_, Y_train)
end = time.time()
runtime_table.add_row(["SVM", round(end-start, 2)])
start = time.time()
mlp_model, mlp_history = MLP_learner(X_train_, Y_train)
end = time.time()
runtime_table.add_row(["MLP", round(end-start, 2)])
start = time.time()
cnn_model, cnn_history = CNN_learner(X_train, Y_train)
end = time.time()
runtime_table.add_row(["CNN", round(end-start, 2)])

print(runtime_table)

# plot deep learning models
tf.keras.utils.plot_model(
    cnn_model,
    to_file='model_cnn.png',
    show_shapes=False,
    show_dtype=False,
    show_layer_names=True,
    rankdir='TB',
    expand_nested=False,
    dpi=96,
    layer_range=None,
    show_layer_activations=False,
)

tf.keras.utils.plot_model(
    mlp_model,
    to_file='model_mlp.png',
    show_shapes=False,
    show_dtype=False,
    show_layer_names=True,
    rankdir='TB',
    expand_nested=False,
    dpi=96,
    layer_range=None,
    show_layer_activations=False,
)

### ====================================================== Training History for MLP and CNN
f, (ax1, ax2) = plt.subplots(1, 2, tight_layout = True)
ax1.plot(mlp_history.epoch, mlp_history.history['loss'], label=' Training Loss')
ax1.plot(mlp_history.epoch, mlp_history.history['accuracy'], label='Training Accuracy')
ax1.plot(mlp_history.epoch, mlp_history.history['val_loss'], label='Validation Loss')
ax1.plot(mlp_history.epoch, mlp_history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_ylabel("Loss/Acc")
ax1.set_xlabel("Epoch")
ax1.legend()
ax1.set_title("MLP")

ax2.plot(cnn_history.epoch, cnn_history.history['loss'], label=' Training Loss')
ax2.plot(cnn_history.epoch, cnn_history.history['accuracy'], label='Training Accuracy')
ax2.plot(cnn_history.epoch, cnn_history.history['val_loss'], label='Validation Loss')
ax2.plot(cnn_history.epoch, cnn_history.history['val_accuracy'], label='Validation Accuracy')
ax2.set_ylabel("Loss/Acc")
ax2.set_xlabel("Epoch")
ax2.legend()
ax2.set_title("CNN")

### ====================================================== Performance Evaluation
accuracy_metric = metrics.accuracy_score

# performance of random forest
tree_performance = np.round([tree_model.score(X_train_, Y_train), tree_model.score(X_val_, Y_val), tree_model.score(X_test_, Y_test)], 2)

# performance of SVM
SVM_performance = np.round([svm_model.score(X_train_, Y_train), svm_model.score(X_val_, Y_val), svm_model.score(X_test_, Y_test)], 2)

# MLP performance
y_pred = mlp_model.predict(X_train_)
y_pred_train = multiClassPostProcess(y_pred)

y_pred = mlp_model.predict(X_val_)
y_pred_val = multiClassPostProcess(y_pred)

y_pred = mlp_model.predict(X_test_)
y_pred_test = multiClassPostProcess(y_pred)

MLP_performance = np.round([accuracy_metric(y_pred_train, Y_train), accuracy_metric(y_pred_val, Y_val), accuracy_metric(y_pred_test, Y_test)], 2)

# CNN performance
y_pred = cnn_model.predict(X_train)
y_pred_train = multiClassPostProcess(y_pred)

y_pred = cnn_model.predict(X_val)
y_pred_val = multiClassPostProcess(y_pred)

y_pred = cnn_model.predict(X_test)
y_pred_test = multiClassPostProcess(y_pred)

CNN_performance = np.round([accuracy_metric(y_pred_train, Y_train), accuracy_metric(y_pred_val, Y_val), accuracy_metric(y_pred_test, Y_test)], 2)

performance_table = PrettyTable([ "Model", "Training Accuracy", "Validation Accuracy", "Test Accuracy"])
performance_table.add_row(["Random Forest", tree_performance[0], tree_performance[1], tree_performance[2]])
performance_table.add_row(["SVM", SVM_performance[0], SVM_performance[1], SVM_performance[2]])
performance_table.add_row(["MLP", MLP_performance[0], MLP_performance[1], MLP_performance[2]])
performance_table.add_row(["CNN", CNN_performance[0], CNN_performance[1], CNN_performance[2]])

print(performance_table)

### ====================================================== Confusion Matrix on Test Data
confusion_matrix = metrics.confusion_matrix(Y_test, tree_model.predict(X_test_))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix)
disp.plot()
plt.title("Random Forest")

confusion_matrix = metrics.confusion_matrix(Y_test, svm_model.predict(X_test_))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix)
disp.plot()
plt.title("SVM")

y_pred_test_mlp = multiClassPostProcess(mlp_model.predict(X_test_))
confusion_matrix = metrics.confusion_matrix(Y_test, y_pred_test_mlp)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix)
disp.plot()
plt.title("MLP")

y_pred_test_cnn = multiClassPostProcess(cnn_model.predict(X_test))
confusion_matrix = metrics.confusion_matrix(Y_test, y_pred_test_cnn)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix)
disp.plot()
plt.title("CNN")
