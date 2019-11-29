import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

class ImageManager:
    def __init__(self):
        # Defining Directory Paths
        self.training_images_directories = ['./images/Abdullah/training/', 
                                             './images/Mustafa/training/',
                                             './images/Saleh/training/',
                                             './images/Adham/training/',
                                             './images/Anees/training/']
        self.testing_images_directories = ['./images/Abdullah/testing/',
                                           './images/Mustafa/testing/',
                                           './images/Saleh/testing/',
                                           './images/Adham/testing/',
                                           './images/Anees/testing/']
        
        self.training_bounds = []
        self.testing_bounds = []
        self.training_faces = []
        self.testing_faces = []
        self.training_images = []
        self.testing_images = []
    
    def extract_face(self, img, bounds):
        """
        Function takes in an image and bounds dict and returns a face
        """
        min_x =  bounds["min_x"]
        max_x =  bounds["max_x"]
        min_y =  bounds["min_y"]
        max_y =  bounds["max_y"]
        return img[min_y:max_y, min_x:max_x]
        
    def get_training_images(self, color=cv2.COLOR_BGR2RGB):
        training_images = []
        # Populate Training Image list
        for directory in self.training_images_directories:
            images = [cv2.imread(file) for file in glob.glob("{directory}*.jpg".format(directory=directory))]
            for img in images:
                training_images.append(cv2.cvtColor(img, color))
        self.training_images = training_images
        return training_images
    
    def get_testing_images(self, color=cv2.COLOR_BGR2RGB):
        testing_images = []
        # Populate Testing Image list
        for directory in self.testing_images_directories:
            images = [cv2.imread(file) for file in glob.glob("{directory}*.jpg".format(directory=directory))]
            for img in images:
                testing_images.append(cv2.cvtColor(img, color))
        self.testing_images = testing_images
        return testing_images
    
    def update_training_bounds(self):
        training_bounds = []
        result_images = []
        for i, img in enumerate(self.training_images):     
            img_disp = img.copy()
            
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                # Detect harris corners
                corners = cv2.cornerHarris(img_gray, 2, 3, 0.04)
            else:
                # Detect harris corners
                corners = cv2.cornerHarris(img, 2, 3, 0.04)

            # Normalize corner map [0,1]
            cv2.normalize(corners, corners, 0, 1, cv2.NORM_MINMAX)

            # Define threshold for an optimal value
            thres = 0.5

            # List all points higher than threshold
            loc = np.where(corners >= thres)
            
            pts = []
            # Loop though points
            for pt in zip(*loc[::-1]):
                pts.append(pt)
                # draw filled circle on each point
                cv2.circle(img_disp, pt, 4, (255,0,0), -1)
           
            xs, ys = np.array(list(zip(*pts)))
            bounds_dict = {"min_x" : np.amin(xs),
                           "min_y" : np.amin(ys),
                           "max_x" : np.amax(xs),
                           "max_y" : np.amax(ys)}
            
            result_images.append(img)
            training_bounds.append(bounds_dict)
            face = self.extract_face(img, bounds_dict)
            face = cv2.resize(face, (128,128))
            self.training_faces.append(face)  
            
        self.training_bounds = training_bounds
        
        return result_images, training_bounds
    
    def update_testing_bounds(self):
        testing_bounds = []
        result_images = []
        for i, img in enumerate(self.testing_images):
            img_disp = img.copy()
            
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                # Detect harris corners
                corners = cv2.cornerHarris(img_gray, 2, 3, 0.04)
            else:
                # Detect harris corners
                corners = cv2.cornerHarris(img, 2, 3, 0.04)
                
            # Normalize corner map [0,1]
            cv2.normalize(corners, corners, 0, 1, cv2.NORM_MINMAX)

            # Define threshold for an optimal value
            thres = 0.5

            # List all points higher than threshold
            loc = np.where(corners >= thres)
            
            pts = []
            # Loop though points
            for pt in zip(*loc[::-1]):
                pts.append(pt)
                # draw filled circle on each point
                cv2.circle(img_disp, pt, 4, (255,0,0), -1)
           
            xs, ys = np.array(list(zip(*pts)))
            bounds_dict = {"min_x" : np.amin(xs),
                           "min_y" : np.amin(ys),
                           "max_x" : np.amax(xs),
                           "max_y" : np.amax(ys)}
            
            result_images.append(img_disp)
            testing_bounds.append(bounds_dict)
            face = self.extract_face(img, bounds_dict)
            face = cv2.resize(face, (128,128))
            self.testing_faces.append(face)  

        self.testing_bounds = testing_bounds
        
        return result_images, testing_bounds
