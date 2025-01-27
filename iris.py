import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Visualize the dataset categories using matplotlib
for i in range(3):
    plt.scatter(X[y == i, 0], X[y == i, 1], label=iris.target_names[i])

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Iris Dataset Visualization")
plt.legend()
plt.show()

# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict the classes for the testing dataset
y_pred = model.predict(X_test)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix Heatmap")
plt.show()

# Print evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Bar Chart for Class Distribution (True vs Predicted)
true_class_counts = np.bincount(y_test)
predicted_class_counts = np.bincount(y_pred)

bar_width = 0.35
index = np.arange(3)

plt.figure(figsize=(10, 6))
bar1 = plt.bar(index, true_class_counts, bar_width, label='True Labels', color='b')
bar2 = plt.bar(index + bar_width, predicted_class_counts, bar_width, label='Predicted Labels', color='r')

plt.xlabel('Iris Species')
plt.ylabel('Count')
plt.title('Class Distribution - True vs Predicted')
plt.xticks(index + bar_width / 2, iris.target_names)
plt.legend()
plt.show()

# Pie Chart for Predicted Class Distribution
predicted_labels_count = np.bincount(y_pred)

plt.figure(figsize=(7, 7))
plt.pie(predicted_labels_count, labels=iris.target_names, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
plt.title('Predicted Class Distribution')
plt.show()
