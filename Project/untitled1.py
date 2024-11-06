import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# paths to the datasets
train_path = r'D:\IntroToAI\Project\archive (sport Balls)\train'
test_path = r'D:\IntroToAI\Project\archive (sport Balls)\test'

# preparing the dataset by loading the images and their labels from the dataset folder structure.
def load_images_from_folder(folder_path):
    #Initialise empty lists
    images = []
    labels = []
    #Loop through folders
    for label in os.listdir(folder_path):
        #create the path to the current subfolder
        label_path = os.path.join(folder_path, label)
        #Check if it’s a folder
        if os.path.isdir(label_path):  # Skip non-directory files if any
            #Loop through each image in the subfolder
            for filename in os.listdir(label_path):                
                img_path = os.path.join(label_path, filename) #For each file in the subfolder
                image = cv2.imread(img_path) # loads the image into memory using OpenCV.
                #Store valid images
                if image is not None: #checks if the image was loaded successfully.
                    images.append(image)
                    labels.append(label)
                    
    return images, labels

# Load train and test data
train_images, train_labels = load_images_from_folder(train_path)
test_images, test_labels = load_images_from_folder(test_path)

#Feature Extraction Using HOG
def extract_hog_features(images):
    hog_features = [] # creates a list to store the HOG feature vector for each image.
    for image in images:
        # Resize image to a fixed size for consistent feature extraction
        image_resized = cv2.resize(image, (64, 64))
        
        # Convert the image to grayscale (since multichannel is causing issues)
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        
        # Extract HOG features
        features, _ = hog(
            image_gray, 
            orientations=9, 
            pixels_per_cell=(8, 8), 
            cells_per_block=(2, 2), 
            block_norm='L2-Hys', 
            visualize=True, 
        )
        hog_features.append(features) # adds the extracted feature vector to hog_features
    #Convert to NumPy array    
    return np.array(hog_features)

# Extract HOG features for train and test images
train_features = extract_hog_features(train_images)
test_features = extract_hog_features(test_images)

#Convert the text labels (ball types) to numerical format using LabelEncoder
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Initialize and train SVM classifier
svm_classifier = svm.SVC(kernel='linear', C=1.0)
svm_classifier.fit(train_features, train_labels_encoded)

# Evaluate the Model on Test Data
# Predict on test data
test_predictions = svm_classifier.predict(test_features)

# Calculate accuracy
accuracy = accuracy_score(test_labels_encoded, test_predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
print(classification_report(test_labels_encoded, test_predictions, target_names=label_encoder.classes_))
