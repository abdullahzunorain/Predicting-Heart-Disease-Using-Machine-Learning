# Predicting Heart Disease Using Machine Learning

This repository contains the implementation of various machine learning models to predict heart disease using a dataset that includes multiple health indicators. The project focuses on evaluating several algorithms and comparing their performance in terms of accuracy and other key metrics.

## Project Overview

Cardiovascular diseases (CVDs) are one of the leading causes of death globally. Early detection of heart disease can greatly improve the chances of effective treatment and prevention. In this project, we aim to predict whether a person is likely to suffer from heart disease based on medical features using several machine learning techniques.

## Project Structure

The repository is organized as follows:

- **Dataset**: Contains the dataset used for training and testing the models.
- **Notebooks**: Contains all the Jupyter notebooks where the models were developed and evaluated.
  - `00_Cardio_Disease_Detection.ipynb`: Initial exploratory data analysis (EDA) and preprocessing steps.
  - `01_Cardio_Disease_Pred (KNN).ipynb`: Implementation of the K-Nearest Neighbors (KNN) model.
  - `Decision_Tree.ipynb`: Implementation of the Decision Tree model.
  - `Logistic Regression Model.ipynb`: Implementation of the Logistic Regression model.
  - `Naive_Bayse.ipynb`: Implementation of the Naive Bayes model.
  - `RF_+_DT.ipynb`: Combination of Random Forest and Decision Tree models.
  - `RF_+_NB.ipynb`: Combination of Random Forest and Naive Bayes models.
  - `Random_Forest.ipynb`: Implementation of the Random Forest model.
- **Images and Plots**: Includes visualizations like normalized distribution plots and boxplots used for analysis.
- **Documentation**: 
  - `Methodology.docx`: Detailed explanation of the methods and processes followed during the project.
  - `README.md`: Current document providing an overview of the project.

## Methodology

1. **Data Preprocessing**: The dataset was cleaned and preprocessed, including handling missing values, normalizing features, and performing exploratory data analysis (EDA) to understand the data distribution.
2. **Model Training**: Various models were implemented and trained, including:
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Logistic Regression
   - Naive Bayes
   - Random Forest
   - Hybrid models (Random Forest + Decision Tree, Random Forest + Naive Bayes)
3. **Model Evaluation**: Each model was evaluated based on accuracy, precision, recall, F1-score, and ROC-AUC to determine the best performing model for heart disease prediction.
4. **Comparison**: Results from the models were compared to assess which algorithm performs best under different metrics.

## Results

The comparison between the models will help in understanding which algorithm provides the most reliable results for predicting heart disease. The evaluation metrics provide insight into the trade-offs between precision and recall, especially in medical diagnosis contexts where false positives and false negatives have different implications.

## Requirements

To run the notebooks, you need the following libraries:

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

You can install the required packages by running:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/abdullahzunorain/Predicting-Heart-Disease-Using-Machine-Learning.git
   ```
2. Navigate to the `Notebooks` folder.
3. Open any notebook using Jupyter and run the cells to see the model training and evaluation.

## Conclusion

This project aims to showcase the application of machine learning techniques in predicting heart disease and how different algorithms perform under similar conditions. The insights from this analysis can contribute to the ongoing research on heart disease diagnosis.
