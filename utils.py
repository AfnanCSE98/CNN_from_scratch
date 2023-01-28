import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_data(n_samples):

    images = []
    for i in range(n_samples):
        filename = 'training-a/a' + str(i).zfill(5) + '.png'
        # read png file using pillow 
        image = plt.imread(filename)
        # convert image to numpy array
        image = np.array(image)
        # append image to images list
        images.append(image)

        # print progress
        if i % 100 == 0:
            print('loaded {} images'.format(i))
    


    #read training-a.csv as pandas dataframe and load two columns : Filename & Digit
    labels = pd.read_csv('training-a.csv')
    # get Digit column of dataframe.Take first n_samples rows
    labels = labels['digit'].values[:n_samples]
    # convert labels to numpy array
    train_labels = np.array(labels)

    images = np.array(images)

    x_train = images
    y_train = train_labels

    # print(x_train[0])
    #normalize data
    # x_train = x_train / 255.0

    x_train = np.mean(x_train, axis=3)
        
    x_validation , y_validation = x_train[int(n_samples*0.7):] , y_train[int(n_samples*0.7):]

    x_train = np.reshape(x_train, (*x_train.shape, 1)).astype(np.float32)
    y_train = np.reshape(y_train, (*y_train.shape, 1))
    x_validation = np.reshape(x_validation, (*x_validation.shape, 1)).astype(np.float32)
    y_validation = np.reshape(y_validation, (*y_validation.shape, 1))

    _ = x_validation.shape[0]
    x_test = x_validation[:int(_*0.5)]
    y_test = y_validation[:int(_*0.5)]

    # print("x_train " , x_train.shape)
    # print("y_train " , y_train.shape)
    # print("x_validation " , x_validation.shape)
    # print("y_validation " , y_validation.shape)
    # print("x_test " , x_test.shape)
    # print("y_test " , y_test.shape)

    return x_train, y_train, x_validation, y_validation, x_test, y_test

def subsample_dataset(num_classes, num_samples_per_class, x, y):
    indices = []
    
    for n_class in range(num_classes):
        indices.append(np.where(y == n_class)[0][: num_samples_per_class])
    
    indices = np.concatenate(indices, axis=0)
    np.random.shuffle(indices)
    
    x = np.take(x, indices, axis=0)
    y = np.take(y, indices, axis=0)
    
    return x, y

def calculate_cross_entropy_loss(y_true, y_predicted):
    return np.sum(-1 * np.sum(y_true * np.log(y_predicted), axis=0))

def calculate_f1_scores(num_classes, y_true, y_predicted):
    true_positives = np.zeros(num_classes)
    false_positives = np.zeros(num_classes)
    false_negatives = np.zeros(num_classes)
    
    for i in range(y_true.shape[0]):
        if y_true[i, 0] == y_predicted[i, 0]:
            true_positives[y_true[i, 0]] = true_positives[y_true[i, 0]] + 1
        else:
            false_positives[y_predicted[i, 0]] = false_positives[y_predicted[i, 0]] + 1
            false_negatives[y_true[i, 0]] = false_negatives[y_true[i, 0]] + 1
    
    # ref: https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f
    accuracy = np.sum(true_positives) / (np.sum(true_positives) + 0.5 * (np.sum(false_positives) + np.sum(false_negatives)))  # micro/global average f1 score
    f1_score = np.mean(true_positives / (true_positives + 0.5 * (false_positives + false_negatives)))  # macro average f1 score
    return accuracy, f1_score


