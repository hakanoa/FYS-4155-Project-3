import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score


def matches():
    """
Reads the English Premier League dataset, preprocesses it, and splits it into training and testing sets.
This function specifically processes the data for predicting full-time match results (Home win, Away win, Draw).

Returns:
X_train (DataFrame): Training feature data.
y_train (Series): Training target data.
X_test (DataFrame): Testing feature data.
y_test (Series): Testing target data.
n_inputs (int): Number of instances in the training data.
n_features (int): Number of features in the dataset.
"""
    # Read the dataset
    df = pd.read_csv('EPL_19-20_21-22.csv')
    #print('Percentage Home Win: ', 100*df['FTR'].value_counts()['H']/len(df['FTR']))
    #print('Percentage Away  Win: ', 100*df['FTR'].value_counts()['A']/len(df['FTR']))
    #print('Percentage Draw: ', 100*df['FTR'].value_counts()['D']/len(df['FTR']))
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%d/%m/%Y')
    df["HT"] = df["HomeTeam"].astype("category").cat.codes
    df["AT"] = df["AwayTeam"].astype("category").cat.codes
    df["day"] = df["Date"].dt.dayofweek
    
    
    
    
    
    # Select relevant features and target variable
    features = ['HT', 'AT', 'day']
    target = 'FTR'  # Full-Time Result (H: Home Win, A: Away Win, D: Draw) 
    df[target] = df[target].map({'H': 0, 'A': 1, 'D': 2}).astype(int)
    
    
    
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
    n_categories = 1
    class_distribution_train = y_train.value_counts()

    print("Class Distribution in the Train Set:")
    print(class_distribution_train)
    
    return X_train, y_train, X_test, y_test, n_inputs, n_features, n_categories
