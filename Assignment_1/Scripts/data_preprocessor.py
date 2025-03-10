# import all necessary libraries here
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Impute Missing Values
# I included the exclude parameter incase to exclude the target column since there are downstream effects if you try to run the simple model later.
# I noticed if you use the strategy='mean' the simple model will not work because the target column cannot have anything but the descrete values and by running this specific imputation you will get decimal values which is a data type this logistic regression model will not take as it only predicts categorical. 

def impute_missing_values(data, strategy='mean', exclude_columns=[]):
    """
    Fill missing values in the dataset.
    :param data: pandas DataFrame
    :param strategy: str, imputation method ('mean', 'median', 'mode')
    :return: pandas DataFrame
    """
    # Imputation will only work on numeric types so I select only the columns in my dataset that are numerical.
    python_numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    messy_data_numeric_columns = data.select_dtypes(include=python_numeric_types)
    
    # I select only the columns that have null values.
    numeric_columns_nullability = messy_data_numeric_columns.isna().any()
    numeric_nullable_columns = numeric_columns_nullability[numeric_columns_nullability == True]
    numeric_nullable_relevant_columns = numeric_nullable_columns.drop(exclude_columns)

    # Conditional statements for the mean, median, and mode strategies.
    numeric_nullable_relevant_column_names = list(numeric_nullable_relevant_columns.index)
    if strategy == 'mean':
        fill_values = messy_data_numeric_columns[numeric_nullable_relevant_column_names].mean(numeric_only=True, skipna=True)
    elif strategy == 'median':
        fill_values = messy_data_numeric_columns[numeric_nullable_relevant_column_names].median(numeric_only=True, skipna=True)
    elif strategy == 'mode':
        fill_values = messy_data_numeric_columns[numeric_nullable_relevant_column_names].mode(axis=0, numeric_only=True, dropna=True).head(1).values.flatten().tolist()
    else: 
        raise Exception('Only enter mean, median, mode')

    columns_fill_na_values = list(zip(numeric_nullable_relevant_column_names, fill_values))

    messy_data_filled_na = data.copy()

    # For loop to interate and fill all the missing values.
    for column_name, fill_na_value in columns_fill_na_values:
        messy_data_filled_na[column_name] = messy_data_filled_na[column_name].fillna(fill_na_value)
        
    # The function will return the filled messy_dataset.
    return messy_data_filled_na

# I created simple test functions to test if my impute_missing_values(data) function would work on each strategy on a simple dataframe.
# In each case I create a test_dataframe that needs to be modified to fit an expected_dataframe.
# If the two dataframes equal then the test passes.

# This is the test for the mean strategy.
def impute_missing_values_mean_test():
    test_dataframe = pd.DataFrame({
        'Col1': [23, 45, 67, 89, np.nan, 34, 76, np.nan, 90, 12],
        'Col2': [56, np.nan, 78, 12, 45, 67, np.nan, 89, 23, 56],
        'Col3': [np.nan, 34, 56, 78, 90, 12, 45, 67, np.nan, 89],
        'Col4': [12, 23, 34, np.nan, 56, 78, 90, 12, 45, np.nan],
        'Col5': [67, 89, np.nan, 12, 23, 34, 56, np.nan, 78, 90]
    })
    expected_dataframe = pd.DataFrame({
        'Col1': [23, 45, 67, 89, 54.5, 34, 76, 54.5, 90, 12],
        'Col2': [56, 53.25, 78, 12, 45, 67, 53.250, 89, 23, 56],
        'Col3': [58.875, 34, 56, 78, 90, 12, 45, 67, 58.875, 89],
        'Col4': [12, 23, 34, 43.750, 56, 78, 90, 12, 45, 43.750],
        'Col5': [67, 89, 56.125, 12, 23, 34, 56, 56.125, 78, 90]
    })

    actual_dataframe = impute_missing_values(test_dataframe, strategy='mean')

    assert actual_dataframe.equals(expected_dataframe)

# This is the test for the median strategy.
def impute_missing_values_median_test():
    test_dataframe = pd.DataFrame({
        'Col1': [23, 45, 67, 89, np.nan, 34, 76, np.nan, 90, 12],
        'Col2': [56, np.nan, 78, 12, 45, 67, np.nan, 89, 23, 56],
        'Col3': [np.nan, 34, 56, 78, 90, 12, 45, 67, np.nan, 89],
        'Col4': [12, 23, 34, np.nan, 56, 78, 90, 12, 45, np.nan],
        'Col5': [67, 89, np.nan, 12, 23, 34, 56, np.nan, 78, 90]
    })
    expected_dataframe = pd.DataFrame({
        'Col1': [23, 45, 67, 89, 56.0, 34, 76, 56.0, 90, 12],
        'Col2': [56, 56.0, 78, 12, 45, 67, 56.0, 89, 23, 56],
        'Col3': [61.5, 34, 56, 78, 90, 12, 45, 67, 61.5, 89],
        'Col4': [12, 23, 34, 39.5, 56, 78, 90, 12, 45, 39.5],
        'Col5': [67, 89, 61.5, 12, 23, 34, 56, 61.5, 78, 90]
    })

    actual_dataframe = impute_missing_values(test_dataframe, strategy='median')

    assert actual_dataframe.equals(expected_dataframe)

# This is the test for the mode strategy.
def impute_missing_values_mode_test():
    test_dataframe = pd.DataFrame({
        'Col1': [23, 45, 67, 89, np.nan, 34, 76, np.nan, 90, 12],
        'Col2': [56, np.nan, 78, 12, 45, 67, np.nan, 89, 23, 56],
        'Col3': [np.nan, 34, 56, 78, 90, 12, 45, 67, np.nan, 89],
        'Col4': [12, 23, 34, np.nan, 56, 78, 90, 12, 45, np.nan],
        'Col5': [67, 89, np.nan, 12, 23, 34, 56, np.nan, 78, 90]
    })
    expected_dataframe = pd.DataFrame({
        'Col1': [23, 45, 67, 89, 12.0, 34, 76, 12.0, 90, 12],
        'Col2': [56, 56.0, 78, 12, 45, 67, 56.0, 89, 23, 56],
        'Col3': [12.0, 34, 56, 78, 90, 12, 45, 67, 12.0, 89],
        'Col4': [12, 23, 34, 12.0, 56, 78, 90, 12, 45, 12.0],
        'Col5': [67, 89, 12.0, 12, 23, 34, 56, 12.0, 78, 90]
    })

    actual_dataframe = impute_missing_values(test_dataframe, strategy='mode')

    assert actual_dataframe.equals(expected_dataframe)

# 2. Remove Duplicates
def remove_duplicates(data):
    """
    Remove duplicate rows from the dataset.
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """
    messy_data_removed_duplicates = data.drop_duplicates()
    # I needed to make sure that when removing the duplicates the indices of the dataframe also changed otherwise it is not logical since the data frame would still utilize the original indicies. 
    return messy_data_removed_duplicates.reset_index(drop=True)

# I created a simple test function to test if my remove_duplicates(data) function would work on a simple dataframe.
# I created a test_dataframe that needs to be modified to fit an expected_dataframe.
# If the two dataframes equal then the test passes.
def remove_duplicates_test():
    test_dataframe = pd.DataFrame({
        'Col1': [23, 45, 67, 89, 23, 34, 76, 23, 90, 23],
        'Col2': [56, 78, 78, 12, 56, 67, 89, 56, 23, 56],
        'Col3': [34, 34, 56, 78, 34, 12, 45, 34, 67, 34],
        'Col4': [12, 23, 34, 12, 12, 78, 90, 12, 45, 12],
        'Col5': [67, 89, 78, 12, 67, 34, 56, 67, 78, 67]
    })
    expected_dataframe = pd.DataFrame({
        'Col1': [23, 45, 67, 89, 34, 76, 90],
        'Col2': [56, 78, 78, 12, 67, 89, 23],
        'Col3': [34, 34, 56, 78, 12, 45, 67],
        'Col4': [12, 23, 34, 12, 78, 90, 45],
        'Col5': [67, 89, 78, 12, 34, 56, 78]
    })

    actual_dataframe = remove_duplicates(test_dataframe)

    # For Checking:
    # print("Actual DataFrame:\n", actual_dataframe)
    # print("Expected DataFrame:\n", expected_dataframe)
    # print(actual_dataframe.dtypes)
    # print(expected_dataframe.dtypes)

    assert actual_dataframe.equals(expected_dataframe)

# 3. Normalize Numerical Data

# I decided to add the exclude columns incase someone wanted to use the parameter but it is not necessary for this assignment.
def normalize_data(data, method='minmax', exclude_columns=[]):
    """Apply normalization to numerical features.
    :param data: pandas DataFrame
    :param method: str, normalization method ('minmax' (default) or 'standard')
    :return: pandas DataFrame
    """
    # Normalization will only work on numeric types so I select only the columns in my dataset that are numerical.
    python_numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    messy_data_numeric_columns = data.select_dtypes(include=python_numeric_types)
    messy_data_relavent_numeric_columns = messy_data_numeric_columns.drop(exclude_columns)

    # Conditional statements for the standard and minmax methods.
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise Exception("Only enter standard or minmax")

    messy_data_scaled = data.copy()
    messy_data_scaled = scaler.fit_transform(messy_data_relavent_numeric_columns)
    
    # This block of code needed to be written because there was an issue where my dataframe was being transformed into a numpy array. This creates challenges for the next preprocessing step as a numpy array is not a data type that I can input.
    # Here I make sure that the output I will get is a dataframe.
    messy_data_dataframe = pd.DataFrame(messy_data_scaled, columns=messy_data_relavent_numeric_columns.columns, index=data.index)

    # Below I am preserving object data type columns by merging
    final_messy_df = data.copy()
    final_messy_df[messy_data_dataframe.columns] = messy_data_dataframe

    return final_messy_df

# I created a simple test function to test if my normalize_data(data) function would work with my minmax and standard methods on a simple dataframe.
# I created a test_dataframe that needs to be modified.
# I print the original and new processed data to manually look at the difference and if the difference is what I expect the test passes.

# This is a test for the minmax strategy
def normalize_data_test_minmax():
    test_dataframe = pd.DataFrame({
        'Feature1': [10, 20, 30, 40, 50],
        'Feature2': [5, 15, 25, 35, 45],
        'Feature3': [100, 200, 300, 400, 500]
    })
    print("Original DataFrame:")
    print(test_dataframe)

    normalized_data = normalize_data(test_dataframe, method='minmax')
    print("\nNormalized DataFrame:")
    print(normalized_data)

# This is a test for the standard strategy
def normalize_data_test_StandardScaler():
    test_dataframe = pd.DataFrame({
        'Feature1': [5, 10, 15, 20, 25],
        'Feature2': [50, 60, 70, 80, 90],
        'Feature3': [1000, 2000, 3000, 4000, 5000]
    })
    print("Original DataFrame:")
    print(test_dataframe)

    normalized_data = normalize_data(test_dataframe, method='standard')
    print("\nNormalized DataFrame:")
    print(normalized_data)

# 4. Remove Redundant Features
def remove_redundant_features(data, threshold=0.9):
    """Remove redundant or duplicate columns.
    :param data: pandas DataFrame
    :param threshold: float, correlation threshold
    :return: pandas DataFrame
    """
    # Redundancy removal will only work on numeric types so I select only the columns in my dataset that are numerical.
    python_numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    messy_data_numeric_columns = data.select_dtypes(include=python_numeric_types)

    # Here I am creating a correlation matrix of absolute correlation values. 
    absolute_values = messy_data_numeric_columns.corr().abs()
    
    # Here I am selecting the upper triangle matrix to not include the diagonal values.
    upper_triangle = absolute_values.where(np.triu(np.ones(absolute_values.shape), k=1).astype(bool))

    # Here I have a for loop that iterates through the columns and drops a column if it is highly correlated with another column beyond the threshold provided (0.9).
    messy_data_drop = set()
    for column in upper_triangle.columns:
        if any(upper_triangle[column] > threshold):
            messy_data_drop.add(column)

    # errors='ignore' is there incase the function tries to drop non-existant columns.
    return data.drop(columns=list(messy_data_drop), errors='ignore')

# I created a simple test function to test if my remove_redundant_features(data) function would work on a simple dataframe.
# I created a test_dataframe that needs to be modified to fit an expected_dataframe.
# If the two dataframes equal then the test passes.
def remove_redundant_features_test():
    test_dataframe = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [2, 4, 6, 8, 10],
        'C': [5, 3, 6, 2, 7],
        'D': [10, 20, 30, 40, 50],
        'E': [9, 7, 5, 3, 1]
    })
    expected_dataframe = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'C': [5, 3, 6, 2, 7]
    })

    actual_dataframe = remove_redundant_features(test_dataframe, threshold=0.9)
    print("Expected DataFrame:\n", expected_dataframe)
    print("Actual DataFrame:\n", actual_dataframe)
    assert actual_dataframe.equals(expected_dataframe)
# ---------------------------------------------------

def simple_model(input_data, split_data=True, scale_data=False, print_report=False):
    """
    A simple logistic regression model for target classification.
    Parameters:
    input_data (pd.DataFrame): The input data containing features and the target variable 'target' (assume 'target' is the first column).
    split_data (bool): Whether to split the data into training and testing sets. Default is True.
    scale_data (bool): Whether to scale the features using StandardScaler. Default is False.
    print_report (bool): Whether to print the classification report. Default is False.
    Returns:
    None
    The function performs the following steps:
    1. Removes columns with missing data.
    2. Splits the input data into features and target.
    3. Encodes categorical features using one-hot encoding.
    4. Splits the data into training and testing sets (if split_data is True).
    5. Scales the features using StandardScaler (if scale_data is True).
    6. Instantiates and fits a logistic regression model.
    7. Makes predictions on the test set.
    8. Evaluates the model using accuracy score and classification report.
    9. Prints the accuracy and classification report (if print_report is True).
    """

    # if there's any missing data, remove the columns
    input_data.dropna(inplace=True)

    # split the data into features and target
    target = input_data.copy()[input_data.columns[0]]
    features = input_data.copy()[input_data.columns[1:]]

    # if the column is not numeric, encode it (one-hot)
    for col in features.columns:
        if features[col].dtype == 'object':
            features = pd.concat([features, pd.get_dummies(features[col], prefix=col)], axis=1)
            features.drop(col, axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

    if scale_data:
        # scale the data
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)

    # instantiate and fit the model
    log_reg = LogisticRegression(random_state=42, max_iter=100, solver='liblinear', penalty='l2', C=1.0)
    log_reg.fit(X_train, y_train)

    # make predictions and evaluate the model
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')

    # if specified, print the classification report
    if print_report:
        print('Classification Report:')
        print(report)
        print('Read more about the classification report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html and https://www.nb-data.com/p/breaking-down-the-classification')
    
    return None