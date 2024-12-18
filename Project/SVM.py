import os
import cv2  # for image processing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Dynamically determine the base directory (where the script is located)
base_dir = os.path.dirname(os.path.abspath(__file__)) # Extracts the directory where the script resides and Gets the absolute path of the current script.

# Construct paths to the training and testing folders
dataset_dir = os.path.join(base_dir, 'archive (sport Balls)')
train_path = os.path.join(dataset_dir, 'train')
test_path = os.path.join(dataset_dir, 'test')

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
    return np.array(images), np.array(labels)

# Load images
print("Loading training data...")
train_images, train_labels = load_images_from_folder(train_path)
print("Loading testing data...")
test_images, test_labels = load_images_from_folder(test_path)

# Check shapes
print("Train image shapes:", {img.shape for img in train_images})
print("Test image shapes:", {img.shape for img in test_images})

# Flatten images
train_images = train_images.reshape(len(train_images), -1)
test_images = test_images.reshape(len(test_images), -1)

# Check shapes after flatening
print("Train image shapes after flatening:", train_images.shape)
print("Test image shapes after flatening:", test_images.shape)

# Scale features for PCA
scaler = StandardScaler()
train_images_scaled = scaler.fit_transform(train_images)
test_images_scaled = scaler.transform(test_images)

# Apply PCA
print("Applying PCA...")
pca = PCA(n_components=0.95)  # Retain 95% of the variance
train_images_pca = pca.fit_transform(train_images_scaled)
test_images_pca = pca.transform(test_images_scaled)

print(f"Number of PCA components: {pca.n_components_}")

# Check shapes after PCA
print("Train image shapes after PCA:", train_images_pca.shape)
print("Test image shapes after PCA:", test_images_pca.shape)

# Convert the text labels (ball types) to numerical format using LabelEncoder
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Initialize and train the SVM classifier
print("Training SVM...")
svm_model = SVC(kernel='poly', C=0.5, degree=6, gamma='scale', coef0=1)  
svm_model.fit(train_images_pca, train_labels_encoded)

# Predict on test data
print("Evaluating the model...")
test_predictions = svm_model.predict(test_images_pca)

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

# Compute the learning curve
train_sizes, train_scores, test_scores = learning_curve(
    SVC(kernel='poly', C=1, degree=5, gamma='scale', coef0=1),  
    train_images_pca, train_labels_encoded, 
    cv=5,  # 5-fold cross-validation
    scoring='accuracy', 
    n_jobs=-1,  # Use all available processors
    train_sizes=np.linspace(0.1, 1.0, 15),
    verbose = 1
)

# Calculate the mean and standard deviation for training and validation scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training Accuracy', marker='o')
plt.plot(train_sizes, test_scores_mean, label='Validation Accuracy', marker='o')

# Add shaded areas for standard deviation
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color = '#DDDDDD')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color = '#DDDDDD')

# Plot formatting
plt.title('Learning Curve')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Binarize labels for precision-recall curve plotting
y_test_bin = label_binarize(test_labels_encoded, classes=np.arange(len(np.unique(test_labels_encoded))))
y_scores = svm_model.decision_function(test_images_pca)  # decision_function gives scores to plot precision-recall

plt.figure(figsize=(12, 10))
for i in range(len(np.unique(test_labels_encoded))):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_scores[:, i])
    plt.plot(recall, precision, lw=2, label=f'Class {label_encoder.inverse_transform([i])[0]}')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Each Class")
plt.legend(loc='best')
plt.show()

