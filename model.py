import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle 
from convolution import ConvolutionLayer
from relu import ActivationLayer
from maxpooling import MaxPoolingLayer
from flatten import FlatteningLayer
from fullyconnectedlayer import FullyConnectedLayer
from softmax import SoftmaxLayer


class Model:
    def __init__(self, model_path):
        with open(model_path, 'r') as model_file:
            model_specs = [model_spec.split() for model_spec in model_file.read().split('\n') if model_spec != '']
        
        self.model_components = []
        
        for model_spec in model_specs:
            if model_spec[0] == 'Conv':
                self.model_components.append(ConvolutionLayer(num_filters=int(model_spec[1]), kernel_size=int(model_spec[2]), stride=int(model_spec[3]), padding=int(model_spec[4])))
            elif model_spec[0] == 'ReLU':
                self.model_components.append(ActivationLayer())
            elif model_spec[0] == 'Pool':
                self.model_components.append(MaxPoolingLayer(kernel_size=int(model_spec[1]), stride=int(model_spec[2])))
            elif model_spec[0] == 'Flatten':
                self.model_components.append(FlatteningLayer())
            elif model_spec[0] == 'FC':
                self.model_components.append(FullyConnectedLayer(output_dim=int(model_spec[1])))
            elif model_spec[0] == 'Softmax':
                self.model_components.append(SoftmaxLayer())
    
    def __str__(self):
        return '\n'.join(map(str, self.model_components))
    
    def train(self, u, y_true, lr):
        for i in range(len(self.model_components)):
            u = self.model_components[i].forward(u)
        
        del_v = u - y_true  # denoting y_predicted by u
        # calculate training loss
        loss = -np.sum(y_true * np.log(u))
        print('Training loss: ', loss)
        print(" ")
        
        for i in range(len(self.model_components) - 1, -1, -1):
            del_v = self.model_components[i].backward(del_v, lr)

        return loss
    
    def predict(self, u):
        for i in range(len(self.model_components)):
            u = self.model_components[i].forward(u)
        
        return u  # denoting y_predicted by u
    
    def save_model(self):
        for i in range(len(self.model_components)):
            self.model_components[i].save_learnable_parameters()
    
    def set_model(self):
        for i in range(len(self.model_components)):
            self.model_components[i].set_learnable_parameters()

    def print_model(self):
        for i in range(len(self.model_components)):
            print(self.model_components[i])

    #save model using pickle
    def save_model_pickle(self, model_path):
        with open(model_path, 'wb') as model_file:
            pickle.dump(self.model_components, model_file)

    def load_model_pickle(self, model_path):
        with open(model_path, 'rb') as model_file:
            self.model_components = pickle.load(model_file)