import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import RandomOverSampler
import numpy as np

def matches():
    """
 Reads and preprocesses the English Premier League dataset for the prediction of full-time match results.
 This includes feature engineering, handling imbalanced classes, and feature selection.

 Returns:
 X_train_scaled (array): Scaled and feature-selected training data.
 y_train_resampled (array): Oversampled training target data.
 X_test_scaled (array): Scaled and feature-selected testing data.
 y_test (array): Testing target data.
 n_inputs (int): Number of instances in the training data.
 n_features (int): Number of features in the scaled and selected training data.
 selected_names (array): Names of the selected features after feature selection.
 """
    # Read the dataset
    df = pd.read_csv('EPL_19-20_23-24.csv')
    print('Percentage Home Win: ', 100*df['FTR'].value_counts()['H']/len(df['FTR']))
    print('Percentage Away  Win: ', 100*df['FTR'].value_counts()['A']/len(df['FTR']))
    print('Percentage Draw: ', 100*df['FTR'].value_counts()['D']/len(df['FTR']))
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%d/%m/%Y')
    df["HT"] = df["HomeTeam"].astype("category").cat.codes
    df["AT"] = df["AwayTeam"].astype("category").cat.codes
    df["day"] = df["Date"].dt.dayofweek
    
    rolling_cols = ["FTHG", "FTAG", "HTHG", "HTAG",
    "HS", "AS", "HST", "AST", "HF", "AF",
    "HC", "AC", "HY", "AY", "HR", "AR"]

    # Calculate rolling averages for selected features to capture the current form of the teams
    rolling_window = 5
    for feature in rolling_cols:
        df[f'HT_avg_{feature}'] = df.groupby('HT')[feature].transform(lambda x: x.rolling(rolling_window, min_periods=1).mean())
        df[f'AT_avg_{feature}'] = df.groupby('AT')[feature].transform(lambda x: x.rolling(rolling_window, min_periods=1).mean())
        
    # Select relevant features (attributes) and target variable
    features = ['HT', 'AT', 'day','B365H', 'B365D', 'B365A',"BWH", "BWD", "BWA","MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA"] +  [f'HT_avg_{feature}' for feature in rolling_cols] + [f'AT_avg_{feature}' for feature in rolling_cols]
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
    
    # Oversample the minority class
    oversampler = RandomOverSampler(random_state=31)
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)
    
    
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    # Select the K best features
    k = 10
    selector = SelectKBest(f_classif, k=k)
    X_train_scaled = selector.fit_transform(X_train_scaled, y_train_resampled)
    X_test_scaled = selector.transform(X_test_scaled)
    
    n_inputs, n_features = X_train_scaled.shape
    # Get the mask of selected features
    selected_mask = selector.get_support()

    # Get the names of the selected features
    selected_names = np.array(features)[selected_mask]
    
    def print_class_distribution(y, dataset_name):
        class_distribution = y.value_counts(normalize=True) * 100
        print(f"Class Distribution in {dataset_name} (in %):")
        print(class_distribution)
        

    print_class_distribution(y_train_resampled, 'y_train')
    print_class_distribution(y_test, 'y_test')
    
    
    return X_train_scaled, y_train_resampled, X_test_scaled, y_test, n_inputs, n_features, selected_names
