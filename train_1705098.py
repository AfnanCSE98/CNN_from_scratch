import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report , f1_score

class Convolution():
    def __init__(self, n_output_channels, filter_width, stride=1, padding=0):
        self.n_output_channels = n_output_channels
        self.filter_width = filter_width
        self.stride = stride
        self.padding = padding
        self.weights = None
        self.biases = None

    def init_weights_and_biases(self, input_shape):
        # input_shape = (batch, n_input_channels, input_height, input_width)
        batch, n_input_channels, input_height, input_width = input_shape
        self.weights = np.random.randn(self.n_output_channels, n_input_channels, self.filter_width, self.filter_width)
        self.biases = np.random.randn(self.n_output_channels)


    # use np.einsum and np.lib.stride_tricks.as_strided to implement convolution
    def forward(self, input):
        # if None, init weights and biases
        if self.weights is None:
            batch, n_input_channels, input_height, input_width = input.shape
            self.weights = np.random.randn(self.n_output_channels, n_input_channels, self.filter_width, self.filter_width)*np.sqrt(2/(self.filter_width*self.filter_width*n_input_channels))
            self.biases = np.random.randn(self.n_output_channels)

        self.input = input
        n_batch, n_input_channels, input_height, input_width = input.shape
        output_height = (input_height - self.filter_width + 2 * self.padding) // self.stride + 1
        output_width = (input_width - self.filter_width + 2 * self.padding) // self.stride + 1

        output = self.convolve(input, n_batch, output_height, output_width)

        return output


    def convolve(self , input , n_batch , output_height , output_width):
        output = np.zeros((n_batch, self.n_output_channels, output_height, output_width))

        # pad input
        input_padded = np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        for i in range(n_batch):
            for j in range(self.n_output_channels):
                for k in range(output_height):
                    for l in range(output_width):
                        output[i, j, k, l] = np.sum(input_padded[i, :, k * self.stride:k * self.stride + self.filter_width, l * self.stride:l * self.stride + self.filter_width] * self.weights[j]) + self.biases[j]

        return output

    def backward(self, dout, alpha):
        n_batch, n_input_channels, input_height, input_width = self.input.shape
        _, _, output_height, output_width = dout.shape

        input_gradient = self.calculate_gradient_and_update_weights(dout , n_batch , output_height , output_width , input_height , input_width , alpha)

        return input_gradient

    def calculate_gradient_and_update_weights(self , dout , n_batch , output_height , output_width , input_height , input_width , alpha):
        input_gradient = np.zeros(self.input.shape)
        input_padded = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        input_gradient_padded = np.pad(input_gradient, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        for i in range(n_batch):
            for j in range(self.n_output_channels):
                for k in range(output_height):
                    for l in range(output_width):
                        input_gradient_padded[i, :, k * self.stride:k * self.stride + self.filter_width, l * self.stride:l * self.stride + self.filter_width] += dout[i, j, k, l] * self.weights[j]
                        self.weights[j] -= alpha * dout[i, j, k, l] * input_padded[i, :, k * self.stride:k * self.stride + self.filter_width, l * self.stride:l * self.stride + self.filter_width]
                        self.biases[j] -= alpha * dout[i, j, k, l]


        input_gradient = input_gradient_padded[:, :, self.padding:self.padding + input_height, self.padding:self.padding + input_width]
        return input_gradient

    def clean(self):
        self.input = None
        self.output = None


# input_shape = (batch, n_input_channels, input_height, input_width)
class MaxPool2D():
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        self.input = input
        n_batch, n_input_channels, input_height, input_width = input.shape
        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1

        output = self.calculate_output(input, n_batch, n_input_channels, output_height, output_width)

        return output

    def calculate_output(self , input , n_batch , n_input_channels ,output_height , output_width):
        output = np.zeros((n_batch, n_input_channels, output_height, output_width))

        for i in range(n_batch):
            for j in range(n_input_channels):
                for k in range(output_height):
                    for l in range(output_width):
                        output[i, j, k, l] = np.max(input[i, j, k * self.stride:k * self.stride + self.pool_size, l * self.stride:l * self.stride + self.pool_size])
        self.output = output
        return output

    def backward(self, dout, alpha):
        n_batch, n_input_channels, input_height, input_width = self.input.shape
        _, _, output_height, output_width = dout.shape

        input_gradient = self.calculate_gradient_and_update_weights(dout , n_batch , n_input_channels ,output_height , output_width)

        return input_gradient



    def calculate_gradient_and_update_weights(self , dout , n_batch , n_input_channels ,output_height , output_width):
        input_gradient = np.zeros(self.input.shape)
        for i in range(n_batch):
            for j in range(n_input_channels):
                for k in range(output_height):
                    for l in range(output_width):
                        max_index = np.argmax(self.input[i, j, k * self.stride:k * self.stride + self.pool_size, l * self.stride:l * self.stride + self.pool_size])
                        max_index = np.unravel_index(max_index, (self.pool_size, self.pool_size))
                        input_gradient[i, j, k * self.stride + max_index[0], l * self.stride + max_index[1]] = dout[i, j, k, l]
        return input_gradient

    def clean(self):
        self.input = None
        self.output = None


# input_shape = (batch, n_input_channels, input_height, input_width)
class ReLU():
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, dout, alpha):
        input_gradient = np.zeros(self.input.shape)
        input_gradient[self.input > 0] = dout[self.input > 0]
        return input_gradient

    def clean(self):
        self.input = None


class Softmax():
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        exps = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def backward(self, dout, alpha):
        return dout

    def clean(self):
        self.input = None

class Flattener():
    def __init__(self):
        self.input_shape = None

    def forward(self, input):
        self.input_shape = input.shape
        return input.reshape(input.shape[0], -1)

    def backward(self, dout, alpha):
        return dout.reshape(self.input_shape)

    def clean(self):
        self.input_shape = None

class FullyConnLayer():
    def __init__(self, output_dimension):
        self.output_dimension = output_dimension
        self.weights = None
        self.bias = None
        self.input = None

    def forward(self, input):
        self.input = input
        if self.weights is None:
            self.weights = np.random.randn(input.shape[1], self.output_dimension)*np.sqrt(2/input.shape[1])
        if self.bias is None:
            self.bias = np.random.randn(self.output_dimension)
        return np.dot(input, self.weights) + self.bias

    def backward(self, dout, alpha):
        input_gradient = np.dot(dout, self.weights.T)
        self.weights -= alpha * np.dot(self.input.T, dout)
        self.bias -= alpha * np.sum(dout, axis=0)
        return input_gradient

    def clean(self):
        self.input = None


class CLE():
    def __init__(self):
        pass

    def clean(self):
        self.input = None

    def loss_f(self, input, true_labels):
        self.input = input
        self.true_labels = true_labels
        self.batch_size = input.shape[0]
        self.loss = -np.sum(true_labels * np.log(input + 1e-8)) / self.batch_size
        return self.loss

    def lossPrime(self):
        dout_ = (self.input - self.true_labels) / self.batch_size
        return dout_


class Model:
    def __init__(self, layers):
        self.layers = layers
        self.error = CLE()

    def clean(self):
        for layer in self.layers:
            layer.clean()

    def backward(self, dout, alpha):
        for layer in reversed(self.layers):
            dout = layer.backward(dout, alpha)
        return dout

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, train_data, train_labels, batch_size, epochs, alpha):
        # create validation set from training set.Take 20% of training set as validation set
        validation_data = train_data[:int(train_data.shape[0] * 0.2)]
        validation_labels = train_labels[:int(train_labels.shape[0] * 0.2)]

        losses = []
        accuracy = []
        f1s = []

        self.batch_size = batch_size
        for epoch in tqdm(range(epochs)):
            for i in tqdm(range(0, train_data.shape[0], batch_size)):
                batch_data = train_data[i:i + batch_size]
                batch_labels = train_labels[i:i + batch_size]
                output = self.forward(batch_data)
                loss = self.error.loss_f(output, batch_labels)
                dout = self.error.lossPrime()
                self.backward(dout, alpha)

            # evaluate on validation set
            losses.append(loss)
            acc , f1 = self.evaluate(validation_data, validation_labels)
            accuracy.append(acc)
            f1s.append(f1)
            print("Epoch: {}, : Loss: {} Accuracy: {}, F1: {}".format(epoch, loss , acc, f1))

            model_path = 'model-ep-'+str(epoch)+'.pkl'
            self.clean()
            with open(model_path, 'wb') as model_file:
                pickle.dump(model.layers , model_file)

        epoch_list = list(range(epochs))
        # create csv file containing epochs , loss , accuracy and f1 score
        df = pd.DataFrame({'epoch': epoch_list, 'loss': losses, 'accuracy': accuracy, 'f1': f1s})
        df.to_csv('loss_accuracy_f1.csv', index=False)
        self.generate_plots(df)
        print("Training complete")

    def generate_plots(self , df):
        # plot loss vs epochs
        plt.plot(df['epoch'], df['loss'])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('loss vs epochs')
        plt.savefig('loss_vs_epochs.png')
        # plt.show()

        # plot accuracy vs epochs
        plt.plot(df['epoch'], df['accuracy'])
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title('accuracy vs epochs')
        plt.savefig('accuracy_vs_epochs.png')
        # plt.show()

        # plot f1 score vs epochs
        plt.plot(df['epoch'], df['f1'])
        plt.xlabel('epochs')
        plt.ylabel('f1')
        plt.title('f1 vs epochs')
        plt.savefig('f1_vs_epochs.png')
        # plt.show()



    def evaluate(self, test_data, test_labels):
        # TODO: implement test function
        output = self.forward(test_data)
        output = np.argmax(output, axis=1)
        test_labels = np.argmax(test_labels, axis=1)
        acc =  accuracy_score(test_labels, output)
        f1 = f1_score(test_labels, output, average='macro')

        return acc , f1


def getData(label_path, image_path):
    df = pd.read_csv(label_path)
    y = df['digit'].values
    y = np.eye(10)[np.array(y)]

    image_names = df['filename'].values
    image_paths = [image_path + name for name in image_names]

    X = []
    for path in tqdm(image_paths):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (32, 32))
        X.append(img)
    X = np.array(X).transpose(0, 3, 1, 2)
    X = X / 255.0
    print(X.shape, y.shape) # X.shape is (batch, dim, dim, channels)
    return X, y


Xc, yc = getData(label_path="training-c.csv", image_path="training-c/")
Xb , yb = getData(label_path="training-b.csv", image_path="training-b/")
Xa , ya = getData(label_path="training-a.csv", image_path="training-a/")
X , y = np.concatenate((Xc, Xb, Xa)) , np.concatenate((yc, yb, ya))

print("Loaded data successfully " , X.shape , y.shape)
X_train, y_train = X , y

# create a model
model = Model([
    Convolution(6, 5, 1),
    ReLU(),
    MaxPool2D(2, 2),
    Convolution(16, 5, 1),
    ReLU(),
    MaxPool2D(2, 2),
    Flattener(),
    FullyConnLayer(120),
    FullyConnLayer(84),
    FullyConnLayer(10),
    Softmax()
])

# train the model
model.train(X_train, y_train, batch_size=32, epochs=20, alpha=0.01)
