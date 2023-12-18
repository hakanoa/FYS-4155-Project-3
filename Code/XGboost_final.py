import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, confusion_matrix
#import matches2_final
import matches_final
import seaborn as sns


"""
 This code loads match data, creates an XGBoost classifier model, performs hyperparameter tuning using GridSearchCV,
 and evaluates the model on test data. This includes calculating feature importances, precision, accuracy, F1 score,
 and visualizing the confusion matrix.
 """

# Load the data
#X_train, y_train, X_test, y_test, n_inputs, n_features, selected_names = matches2_final.matches()
X_train, y_train, X_test, y_test, n_inputs, n_features, n_categories = matches_final.matches()

# Create the XGBClassifier model
xg_clf = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, gamma=0.1, reg_lambda=1)

# Fit the model on the training data
xg_clf.fit(X_train, y_train)

# Get feature importances
feature_importance = xg_clf.feature_importances_

# Print feature importances
print("Feature Importances for Initial Model:")
for i in range(n_features):
    print(f"Feature {i}: {feature_importance[i]}")

# CV Grid search
param_grid = {
    'max_depth': [5,7,10,11],
    'learning_rate': [0.3,0.1, 0.01],
    'gamma': [0.05,0.1,0.3,0.5],
    'reg_lambda': [0.005,0.01,0.1]
}

# Create the GridSearchCV object
grid_clf = GridSearchCV(xg_clf, param_grid, cv=5, scoring='f1_weighted')

# Fit the GridSearchCV object on the training data
grid_clf.fit(X_train, y_train)

# Print the best hyperparameters
best_params = grid_clf.best_params_
print("Best Hyperparameters:", best_params)

# After the model is trained, you can still access feature importances
feature_importance_best = grid_clf.best_estimator_.feature_importances_

# Print feature importances for the best model
print("\nBest Model Feature Importances:")
for i in range(n_features):
    print(f"Feature {i}: {feature_importance_best[i]}")

"""
# Visualize feature importances for case ii
importance_df = pd.DataFrame({
    'Feature': selected_names,
    'Importance': feature_importance_best
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
"""

# Predict the target variable on the test data using the best hyperparameters
y_pred = grid_clf.predict(X_test)

# Print the accuracy of the model on the test data
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
plt.title('Confusion Matrix XGBoost (Percentage)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Print the class distribution of the predictions in percentage
class_distribution_pred_percentage = pd.Series(y_pred).value_counts(normalize=True) * 100
print("Class Distribution in the Predictions (in %):")
print(class_distribution_pred_percentage)
