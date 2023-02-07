import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn import metrics
import pickle
import sys 
from sklearn.metrics import accuracy_score

class Module:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, dout, alpha):
        pass


class Convolution(Module):
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
            self.weights = np.random.randn(self.n_output_channels, n_input_channels, self.filter_width, self.filter_width)
            self.biases = np.random.randn(self.n_output_channels)            
        
        self.input = input
        # Stride the input to obtain windows of size (filter_width x filter_width)
        input_stride = np.lib.stride_tricks.as_strided(
            input, 
            shape=(input.shape[0], 
                input.shape[1], 
                input.shape[2] - self.filter_width + 1, 
                input.shape[3] - self.filter_width + 1, 
                self.filter_width, 
                self.filter_width), 
            strides=input.strides + input.strides[2:4]
        )
        
        # Reshape the strided input to have windows as samples and n_input_channels x filter_width x filter_width as features
        input_stride = input_stride.reshape(-1, input.shape[1], self.filter_width, self.filter_width)
        
        # Calculate dot product between weights and strided input
        self.output = np.einsum("ijkl, mjkl->mi", self.weights, input_stride) + self.biases
        self.output = self.output.reshape(input.shape[0], self.n_output_channels, input.shape[2] - self.filter_width + 1, input.shape[3] - self.filter_width + 1)
        
        return self.output
    
    def backward(self, dout, alpha):
        input_gradient = np.zeros(self.input.shape)
        n_batch, n_input_channels, input_height, input_width = self.input.shape
        _, _, output_height, output_width = dout.shape

        # pad input
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


# input_shape = (batch, n_input_channels, input_height, input_width)
class MaxPool2D(Module):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        self.input = input
        n_batch, n_input_channels, input_height, input_width = input.shape
        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1
        output = np.zeros((n_batch, n_input_channels, output_height, output_width))

        for i in range(n_batch):
            for j in range(n_input_channels):
                for k in range(output_height):
                    for l in range(output_width):
                        output[i, j, k, l] = np.max(input[i, j, k * self.stride:k * self.stride + self.pool_size, l * self.stride:l * self.stride + self.pool_size])
        self.output = output
        return output

    def backward(self, dout, alpha):
        input_gradient = np.zeros(self.input.shape)
        n_batch, n_input_channels, input_height, input_width = self.input.shape
        _, _, output_height, output_width = dout.shape

        for i in range(n_batch):
            for j in range(n_input_channels):
                for k in range(output_height):
                    for l in range(output_width):
                        max_index = np.argmax(self.input[i, j, k * self.stride:k * self.stride + self.pool_size, l * self.stride:l * self.stride + self.pool_size])
                        max_index = np.unravel_index(max_index, (self.pool_size, self.pool_size))
                        input_gradient[i, j, k * self.stride + max_index[0], l * self.stride + max_index[1]] = dout[i, j, k, l]
        return input_gradient


# input_shape = (batch, n_input_channels, input_height, input_width)
class ReLU(Module):
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, dout, alpha):
        input_gradient = np.zeros(self.input.shape)
        input_gradient[self.input > 0] = dout[self.input > 0]
        return input_gradient


class Softmax(Module):
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        exps = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def backward(self, dout, alpha):
        return dout


class Flattener(Module):
    def __init__(self):
        self.input_shape = None

    def forward(self, input): 
        self.input_shape = input.shape
        return input.reshape(input.shape[0], -1)

    def backward(self, dout, alpha):
        return dout.reshape(self.input_shape)
    
class Print(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input):
        print("layer forward's input shape", input.shape)
        return input

    def backward(self, dout, alpha):
        print("layer backward's input shape", dout.shape)
        return dout

class FullyConnLayer(Module):
    def __init__(self, output_dimension):
        self.output_dimension = output_dimension
        self.weights = None
        self.bias = None
        self.input = None
    
    def forward(self, input):
        self.input = input
        if self.weights is None:
            self.weights = np.random.randn(input.shape[1], self.output_dimension)
        if self.bias is None:
            self.bias = np.random.randn(self.output_dimension)
        return np.dot(input, self.weights) + self.bias

    def backward(self, dout, alpha):
        input_gradient = np.dot(dout, self.weights.T)
        self.weights -= alpha * np.dot(self.input.T, dout)
        self.bias -= alpha * np.sum(dout, axis=0)
        return input_gradient


class CLE():
    def __init__(self):
        pass
    
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
        self.batch_size = batch_size
        for epoch in tqdm(range(epochs)):
            for i in tqdm(range(0, train_data.shape[0], batch_size)):
                batch_data = train_data[i:i + batch_size]
                batch_labels = train_labels[i:i + batch_size]
                output = self.forward(batch_data)
                loss = self.error.loss_f(output, batch_labels)
                dout = self.error.lossPrime()
                self.backward(dout, alpha)

            print("Epoch: {}, Loss: {}".format(epoch, loss))

    def evaluate(self, data):
        # TODO: implement test function
        test_data = data["X_test"]
        test_labels = data["y_test"]
        filenames = data["filenames"]

        output = self.forward(test_data)
        output = np.argmax(output, axis=1)
        test_labels = np.argmax(test_labels, axis=1)

        # create a csv file with filename and predicted label
        df = pd.DataFrame({"filename": filenames, "digit": output})
        df.to_csv("1705098_predictions.csv", index=False)
        
        print("Accuracy: ", accuracy_score(test_labels, output))
        pass


def getData(label_path, image_path):
    print("loading data from " , image_path , label_path)
    df = pd.read_csv(label_path)
    y = df['digit'].values
    y = np.eye(10)[np.array(y)] 

    image_names = df['filename'].values
    image_paths = ["./" + image_path + "/" + name for name in image_names]
    
    X = []
    filenames = []
    for path in tqdm(image_paths):
        filenames.append(path.split('/')[-1])
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (32, 32))
        X.append(img)
    X = np.array(X).transpose(0, 3, 1, 2) 
    X = X / 255.0
    print(X.shape, y.shape) # X.shape is (batch, dim, dim, channels)
    return X, y , filenames


# create a model
model = Model([
    Convolution(6, 5, 1),
    ReLU(),
    MaxPool2D(2, 2),
    Convolution(16, 5, 1),
    ReLU(),
    MaxPool2D(2, 2),    
    Flattener(),
    FullyConnLayer(10),
    Softmax()
])

# load model
model_path = 'model.pkl'
with open(model_path, 'rb') as model_file:
    model.layers = pickle.load(model_file)

test_path = sys.argv[1]
X_test, y_test , filenames = getData(label_path=test_path+".csv", image_path=test_path)
data = {
    "X_test": X_test,
    "y_test": y_test,
    "filenames": filenames
}
model.evaluate(data)