# import model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math 
import os 
import csv 

from model import Model
from utils import load_train_data , calculate_f1_scores , calculate_cross_entropy_loss

model = Model('model.txt')
model.print_model()
num_classes = 10
num_samples = 43
num_epochs = 2
lr = 0.001

x_train , y_train , x_validation , y_validation  = load_train_data(n_samples=43 , path = 'training-a')


num_batches = math.ceil(y_train.shape[0] / num_samples)
min_f1_score = math.inf
validation_stats = []
training_stats = []

for epoch in range(num_epochs):
    for batch in range(num_batches):
        print(f'(Training) Epoch: {epoch + 1} -> {batch + 1}/{num_batches} Batches Trained.')
        n_samples = y_train.shape[0] - batch * num_samples if (batch + 1) * num_samples > y_train.shape[0] else num_samples
        y_true = np.zeros((num_classes, n_samples))
        
        for i in range(y_true.shape[1]):
            y_true[y_train[batch * num_samples + i, 0], i] = 1  # generating one-hot encoding of y_train
        
        loss = model.train(x_train[batch * num_samples: batch * num_samples + n_samples], y_true, lr)
    print()
    training_stats.append([epoch + 1, loss])
    print(f'(Training) Epoch: {epoch + 1} -> Training loss {loss}')
    
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

model.save_model_pickle('outputdir/model.pkl')

with open('outputdir/training_stats.csv', 'w') as csv_file:
    csv_writer = csv.writer(csv_file) 
    csv_writer.writerow(['Epoch', 'Training_Loss']) 
    csv_writer.writerows(training_stats)