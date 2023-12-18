import numpy as np
from sklearn.metrics import accuracy_score,precision_score, f1_score, confusion_matrix, make_scorer
from sklearn.model_selection import train_test_split,  GridSearchCV
from sklearn.neural_network import MLPClassifier
import pandas as pd
#import matches2_final
import matches_final
import seaborn as sns
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, hidden_activation, output_activation):
        """
        Initialize a neural network with specified parameters.

        Parameters:
        - input_size (int): Number of input features.
        - hidden_size (int): Number of neurons in the hidden layer.
        - output_size (int): Number of neurons in the output layer.
        - learning_rate (float): Learning rate for weight updates.
        - hidden_activation (str): Activation function for the hidden layer ("sigmoid", "relu", "leaky_relu").
        - output_activation (str): Activation function for the output layer ("sigmoid", "relu", "leaky_relu").
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # Initialize weight matrices with correct dimensions
        self.hidden_weights = np.random.randn(self.input_size, self.hidden_size)
        self.hidden_bias = np.zeros(self.hidden_size) + 0.01
        self.output_weights = np.random.randn(self.hidden_size, self.output_size)
        self.output_bias = np.zeros(self.output_size) + 0.01

    def sigmoid(self, x):
        """
        Sigmoid activation function.

        Parameters:
        - x (numpy.ndarray): Input to the activation function.

        Returns:
        - numpy.ndarray: Output of the sigmoid activation.
        """
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        """
        ReLU activation function.

        Parameters:
        - x (numpy.ndarray): Input to the activation function.

        Returns:
        - numpy.ndarray: Output of the ReLU activation.
        """
        return np.maximum(0, x)

    def leaky_relu(self, x, alpha=0.01):
        """
        Leaky ReLU activation function.

        Parameters:
        - x (numpy.ndarray): Input to the activation function.
        - alpha (float): Slope for negative values.

        Returns:
        - numpy.ndarray: Output of the Leaky ReLU activation.
        """
        return np.where(x > 0, x, alpha * x)

    def feed_forward(self, X):
        """
        Perform feedforward pass through the neural network.

        Parameters:
        - X (numpy.ndarray): Input data.

        Returns:
        - tuple: Tuple containing probabilities and hidden layer activations.
        """
        z_h = np.dot(X, self.hidden_weights) + self.hidden_bias

        if self.hidden_activation == "sigmoid":
            a_h = self.sigmoid(z_h)
        elif self.hidden_activation == "relu":
            a_h = self.relu(z_h)
        elif self.hidden_activation == "leaky_relu":
            a_h = self.leaky_relu(z_h)

        z_o = np.dot(a_h, self.output_weights) + self.output_bias

        if self.output_activation == "sigmoid":
            probabilities = self.sigmoid(z_o)
        elif self.output_activation == "relu":
            probabilities = self.relu(z_o)
        elif self.output_activation == "leaky_relu":
            probabilities = self.leaky_relu(z_o)

        return probabilities, a_h

    def train(self, X, y, n_epochs):
        """
        Train the neural network.

        Parameters:
        - X (numpy.ndarray): Training input data.
        - y (numpy.ndarray): Training labels.
        - n_epochs (int): Number of training epochs.

        Returns:
        - list: List of accuracy scores during training.
        """
        accuracy_scores = []

        for epoch in range(n_epochs):
            probabilities, a_h = self.feed_forward(X)

            y_pred = (probabilities > 0.5).astype(int)
            accuracy = accuracy_score(y, y_pred)
            accuracy_scores.append(accuracy)

            error = np.array(y).reshape(-1, 1) - probabilities

            d_output_weights = np.dot(a_h.T, error)
            d_hidden_weights = np.dot(X.T, error @ self.output_weights.T)

            self.output_weights += self.learning_rate * d_output_weights
            self.hidden_weights += self.learning_rate * d_hidden_weights

        return accuracy_scores


def train_and_evaluate(hidden_activation_func, output_activation_func, n_hidden_neurons, learning_rate, n_epochs, X_train, y_train, X_test, y_test):
    """
    Train and evaluate the neural network with specified hyperparameters.

    Parameters:
    - hidden_activation_func (str): Activation function for the hidden layer.
    - output_activation_func (str): Activation function for the output layer.
    - n_hidden_neurons (int): Number of neurons in the hidden layer.
    - learning_rate (float): Learning rate for weight updates.
    - n_epochs (int): Number of training epochs.
    - X_train (numpy.ndarray): Training input data.
    - y_train (numpy.ndarray): Training labels.
    - X_test (numpy.ndarray): Test input data.
    - y_test (numpy.ndarray): Test labels.

    Returns:
    - tuple: Tuple containing test accuracy, precision, f1_score, and list of accuracy scores during training.
    """
    model = NeuralNetwork(n_features, n_hidden_neurons, n_categories, learning_rate, hidden_activation_func, output_activation_func)
    accuracy_scores = []
    precision_scores = []
    f1_scores = []

    for epoch in range(n_epochs):
        probabilities, a_h = model.feed_forward(X_train)

        y_pred = (probabilities > 0.5).astype(int)
        accuracy = accuracy_score(y_train, y_pred)
        precision = precision_score(y_train, y_pred, average='weighted', zero_division=1)  # Use zero_division=1 to set precision to 0.0
        f1 = f1_score(y_train, y_pred, average='weighted', zero_division=1)  # Use zero_division=1 to set f1_score to 0.0

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        f1_scores.append(f1)

        error = np.array(y_train).reshape(-1, 1) - probabilities

        d_output_weights = np.dot(a_h.T, error)
        d_hidden_weights = np.dot(X_train.T, error @ model.output_weights.T)

        model.output_weights += model.learning_rate * d_output_weights
        model.hidden_weights += model.learning_rate * d_hidden_weights

    # Evaluate the model on the test set
    probabilities, _ = model.feed_forward(X_test)
    y_pred = (probabilities > 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)  # Use zero_division=1 to set precision to 0.0
    test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)  # Use zero_division=1 to set f1_score to 0.0

    return test_accuracy, test_precision, test_f1, accuracy_scores


#Classification analysis
# Loading the data


#X_train, y_train, X_test, y_test, n_inputs, n_features,_ = matches2_final.matches()
X_train, y_train, X_test, y_test, n_inputs, n_features, n_categories = matches_final.matches()
n_categories = 1






#####
# Define hyperparameter values to iterate over
hidden_activation_funcs = ["sigmoid"]
output_activation_funcs = ["sigmoid", "relu", "leaky_relu"]
n_hidden_neurons_values = [5, 10, 15]
learning_rate_values = [0.001, 0.01, 0.1]
n_epochs_values = [500, 1000, 1500]

# Iterate over hyperparameters over and store results
results = []
results_sklearn = []

for hidden_activation_func in hidden_activation_funcs:
    for output_activation_func in output_activation_funcs:
        for n_hidden_neurons in n_hidden_neurons_values:
            for learning_rate in learning_rate_values:
                for n_epochs in n_epochs_values:
                    
                    test_accuracy, test_precision, test_f1, accuracy_scores = train_and_evaluate(hidden_activation_func, output_activation_func, n_hidden_neurons, learning_rate, n_epochs, X_train, y_train, X_test, y_test)
                    results.append({
                        'Hidden Activation': hidden_activation_func,
                        'Output Activation': output_activation_func,
                        'Hidden Neurons': n_hidden_neurons,
                        'Learning Rate': learning_rate,
                        'Epochs': n_epochs,
                        'Test Accuracy': test_accuracy,
                        'Test Precision': test_precision,
                        'Test F1 Score': test_f1
                    })
                    
                
                    


# Print or store the results as needed
for result in results:
    print(result)



# Create DataFrames and save results to Excel files
df_results = pd.DataFrame(results)
df_results_sorted = df_results.sort_values(by='Test F1 Score', ascending=False)
df_results_sorted.to_excel('results_sorted_f1_final2.xlsx', index=False)


# Create an MLPClassifier instance
mlp = MLPClassifier(max_iter=1000, early_stopping=True)

# Define the parameter grid for grid search
param_grid = {
    'hidden_layer_sizes': [(5,), (10,), (15,)],
    'activation': ['relu'],
    'alpha': [0.001, 0.01,0.1],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'solver':['adam','sgd']
}

# Use F1 score as the scoring metric for GridSearchCV
scorer = make_scorer(f1_score, average='weighted')

# Create GridSearchCV object
grid_search = GridSearchCV(mlp, param_grid, scoring=scorer, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Predict the target variable on the test data using the best hyperparameters
y_pred = best_model.predict(X_test)

# Print the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Print the precision of the model on the test data
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
print("Precision Score: {:.3f}".format(precision))

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score: {:.3f}".format(accuracy))

# Calculate and print the F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score: {:.3f}".format(f1))

# Confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)

# Normalize confusion matrix to get percentages
conf_mat_percentage = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis] * 100

print(conf_mat_percentage)

# Plot the confusion matrix
sns.heatmap(conf_mat_percentage, annot=True, fmt='.2f', cmap='Blues')
plt.title('Confusion Matrix MLP Classifier (Percentage)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Print the class distribution of the predictions in percentage
class_distribution_pred_percentage = pd.Series(y_pred).value_counts(normalize=True) * 100
print("Class Distribution in the Predictions (in %):")
print(class_distribution_pred_percentage)





