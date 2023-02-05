
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math 
import os 
import csv 
import sys 

from model import Model
from utils import load_test_data , calculate_f1_scores , calculate_cross_entropy_loss
from sklearn.metrics import classification_report

num_classes = 10

model = Model('model.txt')
# model.load_model_pickle('outputdir/model.pkl')
model.load_model_pickle('outputdir/model-a-0-1000.pkl') 
model.print_model()

test_path = sys.argv[1]
n_sample = 500
x_test , y_test , filenames = load_test_data(n_samples=n_sample , path=test_path)

y_true = np.zeros((num_classes, y_test.shape[0]))
y_predicted = model.predict(x_test)

for i in range(y_true.shape[1]):
    y_true[y_test[i, 0], i] = 1  # generating one-hot encoding of y_test

# print y_true and y_predicted side by side for each image
# print(y_true.shape) # ((10  , num_samples))
# print(y_predicted.shape) # ((10  , num_samples))

predictions = []
for i in range(y_predicted.shape[1]):
    max_index = np.argmax(y_predicted[:,i])
    predictions.append(max_index)
    # filenames.append(x_test[i].split('/')[-1])

#create a dataframe using the filenames and predictions and write to csv file
df = pd.DataFrame({'Filename': filenames, 'Digit': predictions})
df.to_csv('outputdir/1705098_predictions.csv', index=False)

cross_entropy_loss = calculate_cross_entropy_loss(y_true, y_predicted)
accuracy, f1_score = calculate_f1_scores(num_classes, y_test, np.reshape(np.argmax(y_predicted, axis=0), y_test.shape))


test_stats = [[cross_entropy_loss, accuracy, f1_score]]
print(f'(Testing) -> CE Loss: {cross_entropy_loss:.4f}\tAccuracy: {accuracy:.4f}\tF1 Score: {f1_score:.4f}')

print(classification_report(y_test, np.reshape(np.argmax(y_predicted, axis=0), y_test.shape)))

if not os.path.exists('outputdir/'):
    os.makedirs('outputdir/')

with open('outputdir/test_stats.csv', 'w') as csv_file:
    csv_writer = csv.writer(csv_file) 
    csv_writer.writerow(['CE Loss', 'Accuracy', 'F1 Score']) 
    csv_writer.writerows(test_stats)
