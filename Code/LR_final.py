import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import GridSearchCV
#import matches_final
import matches2_final

def sigmoid(z):
    """
Compute the sigmoid function of the input array.

Parameters:
z (array): A numpy array or a scalar for which the sigmoid function is to be computed.

Returns:
array: Sigmoid of the input array.
"""
    return 1.0 / (1.0 + np.exp(-z))

def logistic_regression(X_train, y_train, X_test, y_test, learning_rate, batch_size, Niterations=100, score_type='accuracy'):
    """
Perform logistic regression with mini-batch gradient descent and hyperparameter tuning.

Parameters:
X_train (Dataframe): Training feature data.
y_train (DataFrame): Training target data.
X_test (DataFrame): Testing feature data.
y_test (DataFrame): Testing target data.
learning_rate (list): List of learning rates to try.
batch_size (list): List of batch sizes to try.
Niterations (int, optional): Number of iterations for gradient descent. Defaults to 100.
score_type (str, optional): Metric to optimize ('accuracy', 'precision', 'f1'). Defaults to 'accuracy'.

Prints:
Best score, learning rate, batch size, and confusion matrix. Plots the confusion matrix.
"""
    cost_values = []
    best_cost = 0
    best_learning_rate = 0
    best_batch_size = 0
    best_beta = None

    def predict_proba(X, beta):
        """
 Calculate the probability predictions

 Parameters:
 X (array-like): Feature data for prediction.
 beta (array-like): Coefficients of the logistic regression model.

 Returns:
 array-like: Probability predictions for each data point in X.
 """
        return sigmoid(X @ beta)

    def gradient(X, y, beta):
        """
      Compute the gradient of the cost function for logistic regression.

      Parameters:
      X (array): Feature data used for training.
      y (array): Target variable data used for training.
      beta (array): Current coefficients of the logistic regression model.

      Returns:
      array: Gradient of the cost function.
      """
        y_pred = predict_proba(X, beta)
        error = y_pred - y
        return (1.0 / X.shape[0]) * X.T @ error

    beta = np.random.randn(X_train.shape[1])

    for eta in learning_rate:
        for batch in batch_size:
            cost_values_eta = []
            for epoch in range(Niterations):
                indices = np.random.permutation(X_train.shape[0])
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]

                for i in range(0, X_train.shape[0], batch):
                    X_mini_batch = X_shuffled[i:i+batch]
                    y_mini_batch = y_shuffled[i:i+batch]

                    grad_mini_batch = gradient(X_mini_batch, y_mini_batch, beta)
                    beta -= eta * grad_mini_batch

                y_pred = predict_proba(X_train, beta)
                if score_type == 'accuracy':
                    cost = accuracy_score(y_train, np.round(y_pred))
                elif score_type == 'precision':
                    cost = precision_score(y_train, np.round(y_pred), average='weighted', zero_division=0)
                elif score_type == 'f1':
                    cost = f1_score(y_train, np.round(y_pred), average='weighted')
                cost_values_eta.append(cost)
            cost_values.append(cost_values_eta)
            if np.mean(cost_values_eta) > best_cost:
                best_cost = np.mean(cost_values_eta)
                best_learning_rate = eta
                best_batch_size = batch
                best_beta = beta

    print(f"Best {score_type} score: {best_cost}")
    print(f"Best learning rate: {best_learning_rate}")
    print(f"Best batch size: {best_batch_size}")

    clf = LogisticRegression(solver='lbfgs')
    param_grid = {'C': [0.001, 0.01, 0.1, 1,2]}
    grid = GridSearchCV(clf, param_grid, cv=5, scoring='f1_weighted')  # Based in f1 score
    grid.fit(X_train, y_train)
    print(f"Best {grid.scoring} score SK: {grid.best_score_}")
    print(f"Best learning rate SK: {grid.best_params_}")

    # Predict the target variable on the test data using the best hyperparameters
    y_pred = grid.predict(X_test)
    y_pred_proba = grid.predict_proba(X_test)[:, 1]

    # Print the accuracy of the model on the test data
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    print("Precision Score Sk: {:.3f}".format(precision))

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy Score Sk: {:.3f}".format(accuracy))

    

    # Confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
 
    # Normalize confusion matrix to get percentages
    conf_mat_percentage = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis] * 100
 
    print(conf_mat_percentage)
 
    # Plot the confusion matrix
    sns.heatmap(conf_mat_percentage, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Confusion Matrix Logistic Regression (Percentage)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Print the class distribution of the predictions in percentage
    class_distribution_pred_percentage = pd.Series(y_pred).value_counts(normalize=True) * 100
    print("Class Distribution in the Predictions (in %):")
    print(class_distribution_pred_percentage)

    
    
    

# Get the data
X_train, y_train, X_test, y_test, n_inputs, n_features = matches2_final.matches()
#X_train, y_train, X_test, y_test, n_inputs, n_features, n_categories = matches_final.matches()
n_categories = 1

learning_rates = [0.001, 0.01, 0.1, 1]
batch_sizes = [16, 32, 64, 128]




# Read the dataset
df = pd.read_csv('EPL_19-20_21-22.csv')
print('Percentage Home Win: ', 100*df['FTR'].value_counts()['H']/len(df['FTR']))
print('Percentage Away  Win: ', 100*df['FTR'].value_counts()['A']/len(df['FTR']))
print('Percentage Draw: ', 100*df['FTR'].value_counts()['D']/len(df['FTR']))

df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%d/%m/%Y')
df["HT"] = df["HomeTeam"].astype("category").cat.codes
df["AT"] = df["AwayTeam"].astype("category").cat.codes
df["day"] = df["Date"].dt.dayofweek


   
# Select relevant features (attributes) and target variable
features = ['HT', 'AT', 'day']
target = 'FTR'
df[target] = df[target].map({'H': 0, 'A': 1, 'D': 2}).astype(int)

#print(features)

df = df.dropna() 

# Calculate the index to split the dataset
split_index = int(0.8 * len(df))

# Split the dataset into training and testing sets
train = df.iloc[:split_index, :]
test = df.iloc[split_index:, :]

# Select relevant features and target variable for train set
X_train = train[features]
y_train = train[target]

# Select relevant features and target variable for test set
X_test = test[features]
y_test = test[target]


n_inputs, n_features = X_train.shape

logistic_regression(X_train, y_train, X_test, y_test, learning_rate=learning_rates, batch_size=batch_sizes, score_type='f1')

