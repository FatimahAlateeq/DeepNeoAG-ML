from sklearn.utils import shuffle
import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import gc



def MCNN_data_load():
    """
    Load training and testing data for the MCNN model.
    
    Returns:
        tuple: Tuple containing training data (x_train, y_train) and testing data (x_test, y_test).
    """
    # Define the paths to the training and testing data files
    path_train_pos = "../dataset/Train/pos.npy"
    path_train_neg = "../dataset/Train/neg.npy"
    path_test_pos = "../dataset/Test/pos.npy"
    path_test_neg = "../dataset/Test/neg.npy"
    
    # Load the training and testing data using the data_load function
    x_train,y_train=data_load(path_train_pos,path_train_neg)
    x_test,y_test=data_load(path_test_pos,path_test_neg)
    
    return(x_train,y_train,x_test,y_test)

def data_load(folder1,folder2):
    """
    Load data from two folders and create labels.
    
    Args:
        folder1 (str): Path to the first data folder.
        folder2 (str): Path to the second data folder.
    
    Returns:
        tuple: Tuple containing concatenated data (x) and one-hot encoded labels (y).
    """
    # Load data from the specified folders
    f1=np.load(folder1)
    f2=np.load(folder2)
    
    # Create labels for the data
    label1 = np.ones(f1.shape[0])
    label2 = np.zeros(f2.shape[0])
    
     # Concatenate the data from both folders
    x=np.concatenate([f1,f2], axis=0)
    
    # Concatenate the labels
    y=np.concatenate([label1,label2], axis=0)
    
    # Convert the labels to one-hot encoding
    y= tf.keras.utils.to_categorical(y,2)
    
    # Collect garbage to free up memory
    gc.collect()
    
    return x ,y