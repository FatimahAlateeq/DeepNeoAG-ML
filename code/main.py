#!/usr/bin/env python
# coding: utf-8
import h5py
import os
import pickle
from tqdm import tqdm
from time import gmtime, strftime
import numpy as np
import math
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import roc_curve
import tensorflow as tf
from tensorflow.keras import layers,Model
from sklearn.model_selection import KFold
import gc
import argparse

import loading_data as load_data

"----------------------------------------------------------------------------------------------------"

parser = argparse.ArgumentParser(description="Parse arguments for the training model")
parser.add_argument("-ms", "--max_seq", type=int, default=35, 
                    help="The setting of sequence length.")
parser.add_argument("-n_fil", "--num_filter", type=int, default=64, 
                    help="The number of filters in the convolutional layer.")
parser.add_argument("-n_hid", "--num_hidden", type=int, default=256, 
                    help="The number of hidden units in the dense layer.")
parser.add_argument("-bs", "--batch_size", type=int, default=256, 
                    help="The batch size for training the model.")
parser.add_argument("-ws", "--window_sizes", nargs="+", type=int, default=[4, 8, 16], 
                    help="The window sizes for convolutional filters.")
parser.add_argument("-vm", "--validation_mode", type=str, default="cross", 
                    help='The validation mode. Options are "cross", "independent"')
parser.add_argument("-d", "--data_type", type=str, default="ProtTrans", 
                    help='The type of data. Options are "ProtTrans", "tape", "esm2".')
parser.add_argument("-n_feat", "--num_feature", type=int, default=1024, 
                    help="The number of data feature dimensions. 1024 for ProtTrans, 768 for tape, 1028 for esm2.")

args = parser.parse_args()
MAXSEQ = args.max_seq
NUM_FILTER = args.num_filter
NUM_HIDDEN = args.num_hidden
BATCH_SIZE  = args.batch_size
WINDOW_SIZES = args.window_sizes
VALIDATION_MODE=args.validation_mode
DATA_TYPE = args.data_type
NUM_FEATURE  = args.num_feature

NUM_CLASSES = 2
CLASS_NAMES = ['Negative','Positive']
EPOCHS      = 20
K_Fold = 5

print("\nMCNN_MC\n")
print("The setting of sequence length: ",MAXSEQ)
print("The number of filters in the convolutional layer: ",NUM_FILTER)
print("The number of hidden units in the dense layer: ",NUM_HIDDEN)
print("The batch size for training the model: ",BATCH_SIZE)
print("The window sizes for convolutional filters: ",WINDOW_SIZES)
print("The validation mode: ",VALIDATION_MODE)
print("The type of data: ",DATA_TYPE)
print("The number of data feature dimensions: ",NUM_FEATURE)
print("\n")

"----------------------------------------------------------------------------------------------------"
# model fit batch funtion
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.data))

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = [self.data[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]
        return np.array(batch_data), np.array(batch_labels)
    
"----------------------------------------------------------------------------------------------------"
class DeepScan(Model):
    def __init__(self,
                 input_shape=(1, MAXSEQ, NUM_FEATURE),
                 window_sizes=[32],
                 num_filters=256,
                 num_hidden=1000):
        # Initialize the parent class
        super(DeepScan, self).__init__()
        
        # Initialize the input layer
        self.input_layer = tf.keras.Input(input_shape)
        
        # Initialize convolution window sizes
        self.window_sizes = window_sizes
        
        # Initialize lists to store convolution, pooling, and flatten layers
        self.conv2d = []
        self.maxpool = []
        self.flatten = []
        
        # Create corresponding convolution, pooling, and flatten layers for each window size
        for window_size in self.window_sizes:
            self.conv2d.append(
                layers.Conv2D(filters=num_filters,
                              kernel_size=(1, window_size),
                              activation=tf.nn.relu,
                              padding='valid',
                              bias_initializer=tf.constant_initializer(0.1),
                              kernel_initializer=tf.keras.initializers.GlorotUniform())
            )
            self.maxpool.append(
                layers.MaxPooling2D(pool_size=(1, MAXSEQ - window_size + 1),
                                    strides=(1, MAXSEQ),
                                    padding='valid')
            )
            self.flatten.append(
                layers.Flatten()
            )
        
        # Initialize Dropout layer to prevent overfitting
        self.dropout = layers.Dropout(rate=0.7)
        
        # Initialize the first fully connected layer
        self.fc1 = layers.Dense(num_hidden,
                                activation=tf.nn.relu,
                                bias_initializer=tf.constant_initializer(0.1),
                                kernel_initializer=tf.keras.initializers.GlorotUniform()
        )
        
        # Initialize the output layer with softmax activation
        self.fc2 = layers.Dense(NUM_CLASSES,
                                activation='softmax',
                                kernel_regularizer=tf.keras.regularizers.l2(1e-3)
        )
        
        # Get the output layer by calling the call method
        self.out = self.call(self.input_layer)

    def call(self, x, training=False):
        # List to store outputs of convolution, pooling, and flatten layers
        _x = []
        
        # Perform convolution, pooling, and flatten operations for each window size
        for i in range(len(self.window_sizes)):
            x_conv = self.conv2d[i](x)
            x_maxp = self.maxpool[i](x_conv)
            x_flat = self.flatten[i](x_maxp)
            _x.append(x_flat)
        
        # Concatenate the outputs of all flatten layers
        x = tf.concat(_x, 1)
        
        # Apply Dropout layer
        x = self.dropout(x, training=training)
        
        # Apply the first fully connected layer
        x = self.fc1(x)
        
        # Apply the output layer
        x = self.fc2(x)
        
        return x

        
"----------------------------------------------------------------------------------------------------"

x_train,y_train,x_test,y_test= load_data.MCNN_data_load() #Load dataset from loading_data.py

print("The shape of training dataset :",x_train.shape)
print("The data type of training dataset :",x_train.dtype)
print("The shape of training label :",y_train.shape)
print("The shape of validation dataset :",x_test.shape)
print("The data type of validation dataset :",x_test.dtype)
print("The shape of validation label :",y_test.shape)
print("\n")
"----------------------------------------------------------------------------------------------------"
def model_test(model, x_test, y_test):
    
    # Generate predictions for the test data
    pred_test = model.predict(x_test)
    
    # Calculate the false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(y_test[:, 1], pred_test[:, 1])
    # Calculate the Area Under the Curve (AUC) for the ROC curve
    AUC = metrics.auc(fpr, tpr)
    # Display the ROC curve
    if (VALIDATION_MODE!="cross"):
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=AUC, estimator_name='mCNN')
        display.plot()
    
    # Calculate the geometric mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))
    # Locate the index of the largest geometric mean
    ix = np.argmax(gmeans)
    print(f'\nBest Threshold={thresholds[ix]}, G-Mean={gmeans[ix]}')
    # Set the threshold to the one with the highest geometric mean
    threshold = thresholds[ix]
    # Generate binary predictions based on the threshold
    y_pred = (pred_test[:, 1] >= threshold).astype(int)
    
    # Calculate confusion matrix values: TN, FP, FN, TP
    TN, FP, FN, TP = metrics.confusion_matrix(y_test[:, 1], y_pred).ravel()
    # Calculate Sensitivity (Recall)
    Sens = TP / (TP + FN) if TP + FN > 0 else 0.0
    # Calculate Specificity
    Spec = TN / (FP + TN) if FP + TN > 0 else 0.0
    # Calculate Accuracy
    Acc = (TP + TN) / (TP + FP + TN + FN)
    # Calculate Matthews Correlation Coefficient (MCC)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) if TP + FP > 0 and FP + TN > 0 and TP + FN and TN + FN else 0.0
    # Calculate F1 Score
    F1 = 2 * TP / (2 * TP + FP + FN)
    # Calculate Precision
    Prec = TP / (TP + FP)
    # Calculate Recall
    Recall = TP / (TP + FN)
    
    # Print the performance metrics
    print(f'TP={TP}, FP={FP}, TN={TN}, FN={FN}, Sens={Sens:.4f}, Spec={Spec:.4f}, Acc={Acc:.4f}, MCC={MCC:.4f}, AUC={AUC:.4f}, F1={F1:.4f}, Prec={Prec:.4f}, Recall={Recall:.4f}\n')
    
    # Return the performance metrics
    return TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, F1, Prec, Recall
    
"----------------------------------------------------------------------------------------------------"

if(VALIDATION_MODE == "cross"):
    # Initialize K-Fold cross-validation
    kfold = KFold(n_splits=K_Fold, shuffle=True, random_state=2)
    
    results = []  # List to store results of each fold
    i = 1  # Counter for fold number
    
    # Iterate over each split of the dataset
    for train_index, test_index in kfold.split(x_train):
        print(f"{i} / {K_Fold}\n")
        
        # Split the data into training and testing sets for the current fold
        X_train, X_test = x_train[train_index], x_train[test_index]
        Y_train, Y_test = y_train[train_index], y_train[test_index]
        
        # Print the shapes of the training and testing datasets
        print("The shape of training dataset of cross validation:", X_train.shape)
        print("The shape of training label of cross validation:", Y_train.shape)
        print("The shape of validation dataset of cross validation:", X_test.shape)
        print("The shape of validation label of cross validation:", Y_test.shape)
        print("\n")
        
        # Create a data generator for the training data
        generator = DataGenerator(X_train, Y_train, batch_size=BATCH_SIZE)
        
        # Initialize the DeepScan model
        model = DeepScan(
            num_filters=NUM_FILTER,
            num_hidden=NUM_HIDDEN,
            window_sizes=WINDOW_SIZES
        )
        
        # Compile the model with Adam optimizer and binary cross-entropy loss
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Build the model with the input shape of the training data
        model.build(input_shape=X_train.shape)
        
        # Print the model summary
        model.summary()
        
        # Train the model
        history = model.fit(
            generator,
            epochs=EPOCHS,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)],
            verbose=1,
            shuffle=True
        )
        
        # Test the model on the validation set and get performance metrics
        TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, F1, Prec, Recall = model_test(model, X_test, Y_test)
        
        # Append the results to the list
        results.append([TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, F1, Prec, Recall])
        
        # Increment the fold counter
        i += 1
        
        # Clear the training and testing data from memory
        del X_train
        del X_test
        del Y_train
        del Y_test
        gc.collect()
    
    # Calculate the mean results across all folds
    mean_results = np.mean(results, axis=0)
    
    # Print the mean results of the cross-validation
    print(f"The mean of {K_Fold}-Fold cross-validation results:")
    print(f'TP={mean_results[0]:.4}, FP={mean_results[1]:.4}, TN={mean_results[2]:.4}, FN={mean_results[3]:.4}, '
          f'Sens={mean_results[4]:.4}, Spec={mean_results[5]:.4}, Acc={mean_results[6]:.4}, MCC={mean_results[7]:.4}, AUC={mean_results[8]:.4}, F1={mean_results[9]:.4}, Prec={mean_results[10]:.4}, Recall={mean_results[10]:.4}\n')

"----------------------------------------------------------------------------------------------------"

if(VALIDATION_MODE == "independent"):
    # Create a data generator for the training data
    generator = DataGenerator(x_train, y_train, batch_size=BATCH_SIZE)
    
    # Initialize the DeepScan model
    model = DeepScan(
        num_filters=NUM_FILTER,
        num_hidden=NUM_HIDDEN,
        window_sizes=WINDOW_SIZES
    )
    
    # Compile the model with Adam optimizer and binary cross-entropy loss
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Build the model with the input shape of the training data
    model.build(input_shape=x_train.shape)
    
    # Print the model summary
    model.summary()
    
    # Train the model
    model.fit(
        generator,
        epochs=EPOCHS,
        shuffle=True,
    )
    
    # Test the model on the independent test set and get performance metrics
    TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, F1, Prec, Recall = model_test(model, x_test, y_test)
    
    # Print the performance metrics
    print(f'TP={TP}, FP={FP}, TN={TN}, FN={FN}, Sens={Sens:.4f}, Spec={Spec:.4f}, Acc={Acc:.4f}, MCC={MCC:.4f}, AUC={AUC:.4f}, F1={F1:.4f}, Prec={Prec:.4f}, Recall={Recall:.4f}\n')

    
"----------------------------------------------------------------------------------------------------"





