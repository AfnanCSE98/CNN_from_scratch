# convolutional-neural-network-from-scratch  

This repository contains all the programs coded for the assignment on *Convolutional Neural Network (CNN) Implementation from Scratch* **(Offline-4)**. The following model components of a convolutional neural network are implemented from scratch for image classification tasks in this assignment.  

- Convolution Layer  
- Activation Layer  
- Max Pooling Layer  
- Flattening Layer  
- Fully Connected Layer  
- Softmax Layer  

The forward and backward functions are implemented for each of these components so that any model architecture containing these components can be trained with backpropagation algorithm. **[numtaDB](https://www.kaggle.com/datasets/BengaliAI/numta)** dataset from kaggle were used in this assignment for image classification of bangla handwritten digits.  



## Train
- Make sure numpy,pandas,scikit-learn,matplotlib and seaborn packages are installed.
- Extract the dataset into the same directory where the training script resides.
- run the `train_1705098.py` file.
- a pickle file named `1705098_model.pkl` will be generated after the training is completed.

## Inference
- If you want generate predictions for the images from directory `testing-a` , run `python test_1705098.py 'testing-a'`. A csv file containing the predictions will be generated inside `testing-a` directory.
- If you want to generate predictions as well as check the correctness of your model's performance , you must have a `testing-a.csv` file.Run the test script with `python test_1705098.py 'testing-a' True

## Architecture
Lenet-5
![](https://images.app.goo.gl/aCbjaSh2k1DcXUFc7)

## model's performance
