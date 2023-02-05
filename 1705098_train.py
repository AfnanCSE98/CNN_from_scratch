# import model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math 
import os 
import csv 
import time 

from model import Model
from utils import load_train_data , calculate_f1_scores , calculate_cross_entropy_loss

model = Model('model.txt')
# model.load_model_pickle('outputdir/model-a-0-1000.pkl') 
model.print_model()

num_classes = 10
num_samples = 100
num_epochs = 1
lr = 0.001

def load_data(n_samples , path):
    images = []
    for i in range( n_samples+1):
        filename = path + '/a' + str(i).zfill(5) + '.png'
        # read png file using pillow 
        image = plt.imread(filename)
        # convert image to numpy array
        image = np.array(image)
        # append image to images list
        images.append(image)

        # print progress
        if i % 500 == 0:
            print('loaded {} images'.format(i))
    

    #read training-a.csv as pandas dataframe and load two columns : Filename & Digit
    labels = pd.read_csv(path + '.csv')
    # get Digit column of dataframe.Take first n_samples rows
    labels = labels['digit'].values[:n_samples]
    # convert labels to numpy array
    train_labels = np.array(labels)

    images = np.array(images)

    x_train = images
    y_train = train_labels

    x_train = np.mean(x_train, axis=3)
        
    x_validation , y_validation = x_train[int(n_samples*0.8):] , y_train[int(n_samples*0.8):]

    x_train = np.reshape(x_train, (*x_train.shape, 1)).astype(np.float32)
    y_train = np.reshape(y_train, (*y_train.shape, 1))
    x_validation = np.reshape(x_validation, (*x_validation.shape, 1)).astype(np.float32)
    y_validation = np.reshape(y_validation, (*y_validation.shape, 1))

    return x_train, y_train, x_validation, y_validation

n_samples = 500
x_train , y_train , x_validation , y_validation  = load_data(n_samples , 'training-a')

num_batches = math.ceil(y_train.shape[0] / num_samples)
min_f1_score = math.inf
validation_stats = []
training_stats = []

prev_loss = 500
for epoch in range(num_epochs):
    for batch in range(num_batches):
        print(f'(Training) Epoch: {epoch + 1} -> {batch + 1}/{num_batches} Batches Trained.')

        if (batch + 1) * num_samples > y_train.shape[0]:
            n_samples = y_train.shape[0] - batch * num_samples 
        else:
            num_samples

        y_true = np.zeros((num_classes, n_samples))
        print("--")
        for i in range(y_true.shape[1]):
            y_true[y_train[batch * num_samples + i, 0], i] = 1  # generating one-hot encoding of y_train
        
        # perform normalisation
        # print("**")
        x_train[batch * num_samples: batch * num_samples + n_samples] = (x_train[batch * num_samples: batch * num_samples + n_samples] - np.mean(x_train[batch * num_samples: batch * num_samples + n_samples])) / np.std(x_train[batch * num_samples: batch * num_samples + n_samples])
        # print("++")
        loss = model.train(x_train[batch * num_samples: batch * num_samples + n_samples], y_true, lr)
        # print("__")

        if loss > prev_loss:
            lr /= 2.0
        else:
            lr *= 1.05
        
        prev_loss = loss
    
    print()
    training_stats.append([epoch + 1, loss])
    print(f'(Training) Epoch: {epoch + 1} -> Training loss {loss}')

    current_time = time.time()
    model.save_model_pickle('outputdir/model-' +  str(current_time) +'.pkl')
    
    y_true = np.zeros((num_classes, y_validation.shape[0]))
    y_predicted = model.predict(x_validation)
    
    for i in range(y_true.shape[1]):
        y_true[y_validation[i, 0], i] = 1  # generating one-hot encoding of y_validation
    
    cross_entropy_loss = calculate_cross_entropy_loss(y_true, y_predicted)
    accuracy, f1_score = calculate_f1_scores(num_classes, y_validation, np.reshape(np.argmax(y_predicted, axis=0), y_validation.shape))
    
    if f1_score < min_f1_score:
        min_f1_score = f1_score
        model.save_model()
    
    validation_stats.append([epoch + 1, cross_entropy_loss, accuracy, f1_score])
    print(f'\n(Validation) Epoch: {epoch + 1} -> CE Loss: {cross_entropy_loss:.4f}\tAccuracy: {accuracy:.4f}\tF1 Score: {f1_score:.4f}\n')

if not os.path.exists('outputdir/'):
    os.makedirs('outputdir/')

with open('outputdir/validation_stats.csv', 'w') as csv_file:
    csv_writer = csv.writer(csv_file) 
    csv_writer.writerow(['Epoch', 'CE Loss', 'Accuracy', 'F1 Score']) 
    csv_writer.writerows(validation_stats)


model.set_model()


with open('outputdir/training_stats.csv', 'w') as csv_file:
    csv_writer = csv.writer(csv_file) 
    csv_writer.writerow(['Epoch', 'Training_Loss']) 
    csv_writer.writerows(training_stats)
