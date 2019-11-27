import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import glob

# Defining Directory Paths
training_images_directories = ['./images/Abdullah/training/', 
                               './images/Mustafa/training/',
                               './images/Saleh/training/',
                               './images/Adham/training/']
testing_images_directories = ['./images/Abdullah/testing/',
                              './images/Mustafa/testing/',
                              './images/Saleh/testing/',
                              './images/Adham/testing/']

# Load the cascade for facial detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



class ImageManager:
    def __init__(self):
        # Defining Directory Paths
        self.training_images_directories = ['./images/Abdullah/training/', 
                                             './images/Mustafa/training/',
                                             './images/Saleh/training/',
                                             './images/Adham/training/']
        self.testing_images_directories = ['./images/Abdullah/testing/',
                                           './images/Mustafa/testing/',
                                           './images/Saleh/testing/',
                                           './images/Adham/testing/']
        # Load the cascade for facial detection
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        
    def get_training_images(self, color=cv2.COLOR_BGR2RGB):
        # Populate Training Image list
        for directory in self.training_images_directories:
            images = [cv2.imread(file) for file in glob.glob("{directory}*.jpg".format(directory=directory))]
            for img in images:
                training_images.append(cv2.cvtColor(img, color))
        return training_images
    
    def get_training_images(self, color=cv2.COLOR_BGR2RGB):
        # Populate Training Image list
        for directory in self.testing_images_directories:
            images = [cv2.imread(file) for file in glob.glob("{directory}*.jpg".format(directory=directory))]
            for img in images:
                training_images.append(cv2.cvtColor(img, color))
        return training_images