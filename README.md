# PIMA Diabetes Prediction Using ANN

This repository contains a machine learning project using an Artificial Neural Network (ANN) to predict whether a person has diabetes based on the PIMA Indians Diabetes Database.

## Project Overview

The goal of this project is to predict the presence of diabetes using a dataset that contains information about individuals, such as their age, BMI, blood pressure, and glucose levels. The model is built using a simple ANN and trained on the PIMA Indians Diabetes dataset.

## Dataset

The dataset used in this project is the PIMA Indians Diabetes dataset, which contains medical data for PIMA Indian women. The dataset includes the following features:

- **Pregnancies**: Number of pregnancies the individual has had
- **Glucose**: Plasma glucose concentration
- **Blood Pressure**: Diastolic blood pressure (mm Hg)
- **Skin Thickness**: Triceps skinfold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg / (height in m)^2)
- **Diabetes Pedigree Function**: A function which scores the likelihood of diabetes based on family history
- **Age**: Age of the individual

The target variable is a binary outcome:

- **0**: No diabetes
- **1**: Diabetes

## Requirements

- Python 3.x
- Libraries:
  - NumPy
  - Pandas
  - Scikit-learn
  - Keras
  - Matplotlib

You can install all dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Model Architecture

We use an Artificial Neural Network (ANN) to classify whether a person has diabetes or not. The architecture consists of:

1. **Input Layer**: The input features are passed into the model. In this case, there are 8 input features.
2. **Hidden Layers**: The model uses two hidden layers with ReLU activation.
3. **Output Layer**: A single neuron with sigmoid activation to output a probability between 0 and 1.

## Steps to Run

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/ANN-for-Diabetes.git
   cd ANN-for-Diabetes
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the `app.py` script to train the model:
   ```bash
   python app.py
   ```

4. The model will output the accuracy of the prediction, and you can adjust hyperparameters as needed.

## Model Training

The model is trained using the following steps:

- **Data Preprocessing**: The data is loaded from a CSV file, and missing values are handled.
- **Splitting the Data**: The dataset is split into training and testing sets using `train_test_split`.
- **Model Compilation**: The model uses the Adam optimizer and binary crossentropy loss function, as it's a binary classification problem.
- **Training**: The model is trained for a set number of epochs.

## Evaluation

After training, the model’s accuracy is evaluated on the test set. You can modify the code to try different optimizers, number of layers, or epochs to improve performance.

## Example Usage

To make predictions after the model is trained:

```python
# Assuming X is the input data
predictions = model.predict(X)
preds = [round(pred[0]) for pred in predictions]
print(preds)
```

## Results

The model achieves a certain accuracy on the test data, which can be adjusted based on model parameters.

## Contributions

Feel free to fork this repository and submit pull requests. You can improve the model by trying different architectures, optimizers, or hyperparameter tuning.

## License

This project is licensed under the MIT License.
