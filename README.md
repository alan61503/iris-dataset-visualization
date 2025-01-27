# 🌸 Iris Dataset Classification with Random Forest 🌸

This project demonstrates how to use a Random Forest Classifier to classify the Iris dataset and evaluate the model's performance. The Iris dataset contains measurements of sepals and petals from three species of Iris flowers, and the goal is to predict the species based on these features.

## 📸 Preview:

Here’s a preview of what the visualizations and results look like:

## 🧑‍💻 Libraries Used:
- `numpy`: For numerical operations
- `matplotlib`: For data visualization
- `seaborn`: For enhanced visualizations, including confusion matrix heatmap
- `sklearn`: For loading the dataset, splitting data, and building the Random Forest model

## 📊 Steps in the Project:
1. **Load the Iris dataset** 📥: We use `sklearn.datasets` to load the Iris dataset, which contains 4 features (sepal length, sepal width, petal length, petal width) for 150 samples from 3 species.
2. **Visualize the dataset** 👁️: Using `matplotlib`, we plot the first two features of the dataset (sepal length vs sepal width) and color-code the points based on their species.
3. **Split the dataset** 🔀: We split the data into a training set (80%) and a testing set (20%) using `train_test_split` from `sklearn.model_selection`.
4. **Train a Random Forest Classifier** 🌲: The model is trained on the training data using `RandomForestClassifier` from `sklearn.ensemble`.
5. **Make Predictions** 🔮: After training, we use the model to predict the species of the test set.
6. **Evaluate the Model** 🧐: We calculate and visualize the model's performance using a confusion matrix and the classification report.

## 🖼️ Visualizations:
- **Iris Dataset Visualization**: Scatter plot showing the relationship between sepal length and sepal width for different species.
- **Confusion Matrix Heatmap**: A heatmap showing the comparison of true vs predicted labels for the Iris dataset, providing insights into the model's performance.

## 📈 Evaluation Metrics:
- **Accuracy**: The percentage of correct predictions made by the model.
- **Classification Report**: Precision, recall, and F1-score for each species.

## 🔧 Requirements:
To run this project, ensure you have the following libraries installed:
```bash
pip install numpy matplotlib seaborn scikit-learn

1. **Iris Dataset Visualization**: A scatter plot showing the first two features (sepal length vs sepal width) of the Iris dataset. Each species is color-coded for easy identification.

   ![Iris Dataset Scatter Plot](https://via.placeholder.com/400x300.png)  
   *Example of Iris dataset visualization*

2. **Confusion Matrix Heatmap**: A heatmap visualizing the confusion matrix, showing the true vs predicted labels for the Iris species.

   ![Confusion Matrix Heatmap](https://via.placeholder.com/400x300.png)  
   *Example of confusion matrix heatmap*

3. **Evaluation Metrics**: After running the model, you'll get an accuracy score along with a detailed classification report showing the precision, recall, and F1-score for each species.

   ```plaintext
   Accuracy: 0.9667
   Classification Report:
                 precision    recall  f1-score   support

          setosa       1.00      1.00      1.00         10
      versicolor       0.92      1.00      0.96         10
       virginica       1.00      0.90      0.95         10

      accuracy                           0.97         30
     macro avg       0.97      0.97      0.97         30
  weighted avg       0.97      0.97      0.97         30



