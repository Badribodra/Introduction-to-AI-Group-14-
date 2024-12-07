import os
import cv2  # for image processing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve


# paths to the datasets
train_path = r'D:\IntroToAI\Project\archive (sport Balls)\train'
test_path = r'D:\IntroToAI\Project\archive (sport Balls)\test'
image_size = (64, 64)  # Resize images for consistency

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
                    image = cv2.resize(image, image_size)  # Resize to (64, 64)
                    images.append(image)
                    labels.append(label)
                    
    return images, labels

# Load images
train_images, train_labels = load_images_from_folder(train_path)
test_images, test_labels = load_images_from_folder(test_path)

# Check shapes
print("Train image shapes:", {img.shape for img in train_images})
print("Test image shapes:", {img.shape for img in test_images})

# Convert images and labels to numpy arrays
train_images = np.array(train_images)
test_images = np.array(test_images)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Flatten images
train_images = train_images.reshape(len(train_images), -1)
test_images = test_images.reshape(len(test_images), -1)

# Convert the text labels (ball types) to numerical format using LabelEncoder
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Initialize and train the SVM classifier
svm_model = SVC(kernel='poly', C=1, degree=3, gamma='scale', coef0=1)  # poly resulted in better accuracy than linear and rbf
svm_model.fit(train_images, train_labels_encoded)


# Predict on test data
test_predictions = svm_model.predict(test_images)

# Decode the labels back to original form
test_predictions_labels = label_encoder.inverse_transform(test_predictions)

# Calculate accuracy
accuracy = accuracy_score(test_labels_encoded, test_predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
print(classification_report(test_labels_encoded, test_predictions, target_names=label_encoder.classes_))

# Generate confusion matrix
conf_matrix = confusion_matrix(test_labels_encoded, test_predictions)

# Plot the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Sports Ball Classification")
plt.show()

# Count the number of images per class in train and test sets
train_counts = np.unique(train_labels, return_counts=True)
test_counts = np.unique(test_labels, return_counts=True)

# Plot the distributions
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.barplot(x=train_counts[0], y=train_counts[1])
plt.title('Training Set Class Distribution')
plt.xlabel('Ball Type')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.barplot(x=test_counts[0], y=test_counts[1])
plt.title('Test Set Class Distribution')
plt.xlabel('Ball Type')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Plot learning curve
# model will be trained with 10%, 25%, 50%, 75%, and 100% of the training data.
train_sizes, train_scores, test_scores = learning_curve(SVC(kernel='linear'), train_images, train_labels_encoded, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training Accuracy')
plt.plot(train_sizes, test_scores_mean, label='Validation Accuracy')
plt.title('Learning Curve')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Binarize labels for precision-recall curve plotting
y_test_bin = label_binarize(test_labels_encoded, classes=np.arange(len(np.unique(test_labels_encoded))))
y_scores = svm_model.decision_function(test_images)  # decision_function gives scores to plot precision-recall

plt.figure(figsize=(12, 10))
for i in range(len(np.unique(test_labels_encoded))):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_scores[:, i])
    plt.plot(recall, precision, lw=2, label=f'Class {label_encoder.inverse_transform([i])[0]}')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Each Class")
plt.legend(loc='best')
plt.show()

