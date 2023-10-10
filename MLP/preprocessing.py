import numpy as np

def train_test_val(data, sizes=(60, 20, 20)):
    """
    Split Data into Training, Testing, and Validation Sets

    This function takes a dataset and splits it into three sets: training, testing, and validation.

    Parameters:
    - data: The input dataset to be split.
    - sizes: A tuple representing the percentage split for training, testing, and validation.

    Returns:
    - train: The training set.
    - test: The testing set.
    - valid: The validation set.
    """
    # Convert percentage split to fractions.
    sizes = [x / 100 for x in sizes]
    
    # Calculate the total length of the dataset.
    length = len(data)
    
    # Calculate the number of elements for the non-training set.
    nonTrainLen = int(length * (sizes[1] + sizes[2]))
    
    # Randomly select indices for the non-training set without replacement.
    idx = np.random.choice(np.arange(0, length), size=nonTrainLen, replace=False)
    assert len(idx) == nonTrainLen
    
    # Calculate indices for the training set.
    idxTrain = np.setdiff1d(np.arange(0, length), idx)
    assert len(idx) == nonTrainLen
    
    # Calculate the number of elements for the test set.
    testLen = int(nonTrainLen * sizes[1] / (sizes[1] + sizes[2]))
    
    # Randomly select indices for the test set without replacement.
    idxTest = np.random.choice(np.arange(0, len(idx)), size=testLen, replace=False)
    assert len(idxTest) == testLen
    
    # Calculate the number of elements for the validation set.
    testVal = int(nonTrainLen * sizes[2] / (sizes[1] + sizes[2]))
    
    # Calculate indices for the validation set.
    idxValid = np.setdiff1d(np.arange(0, nonTrainLen), idxTest)
    assert len(idxValid) == testVal
    
    # Ensure that there is no overlap between validation and test sets.
    assert(list(np.intersect1d(idxValid, idxTest)) == [])
    
    # Ensure that there is no overlap between non-training and training sets.
    assert(list(np.intersect1d(idx, idxTrain)) == [])
    
    # Create arrays for the training, testing, and validation sets.
    train = np.array([data[i] for i in idxTrain])
    test = np.array([data[idx[i]] for i in idxTest])
    valid = np.array([data[idx[i]] for i in idxValid])
    
    return train, test, valid, (idxTrain, [idx[i] for i in idxTest], [idx[i] for i in idxValid])


def normalize_min_max(matrix):
    """
    Normalize a Matrix using Min-Max Scaling

    This function takes a matrix and normalizes it using Min-Max scaling, 
    ensuring that the values fall within the range [0, 1].

    Parameters:
    - matrix: The input matrix to be normalized.

    Returns:
    - normalized_matrix: The normalized matrix.
    """
    # Calculate the minimum and maximum values in the matrix.
    min_val = np.min(matrix, axis=0)
    max_val = np.max(matrix, axis=0)
    
    # Normalize the matrix using Min-Max scaling.
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    
    return normalized_matrix


def is_categorical(data):
    """
    Determine categorical columns in a data matrix.

    Args:
        data (numpy.ndarray): Input data matrix, where rows represent samples and columns represent features.

    Returns:
        list: A list of column indices that are considered categorical based on a threshold of unique values.
    """
    
    # Get the dimensions of the data matrix (number of rows and columns)
    n, m = data.shape
    
    # Create an empty list to store the indices of categorical columns
    categorical = []

    for i in range(m):
        # Get the set of all unique values in the current column
        all_values = set(data[:, i]) 
        
        # Check if the number of unique values is less than or equal to 10
        if len(all_values) <= 10:
            # If so, consider the column as categorical and add its index to the list
            categorical.append(i)
    
    # Return the list of indices of categorical columns
    return categorical


def one_hot_encoding(data: np.ndarray, categorical: list, return_idx = False) -> np.ndarray:
    """
    One-hot encodes categorical variables in a NumPy array.

    Args:
        data: A NumPy array containing the data to be encoded.
        categorical: A list of integers representing the indices of the categorical variables in the data array.
        return_idx: A boolean value indicating whether to return the indices of the encoded categorical variables.

    Returns:
        A NumPy array containing the one-hot encoded data. If `return_idx` is True, a tuple containing the encoded data and the indices of the encoded categorical variables is returned.
    """

     # Calculate the number of rows and columns in the data array.
    n, m = data.shape

    assert all(np.array(categorical)<m), "No index cannot exceed the amount of dimensions"

    # Create lists of the indices of the non-categorical and categorical variables.
    idx_non_categorical = [i for i in range(m) if i not in categorical]
    idx_categorical = [i for i in range(m) if i in categorical]

    # Create a list to store the indices of the encoded categorical variables.
    idx = idx_categorical

    # Create a new NumPy array containing the data from the non-categorical variables.
    data_processed = data[:, idx_non_categorical]

    # Iterate over the categorical variables and one-hot encode each variable.
    for col in idx_categorical:
        # Get the categorical variable and its unique values.
        categorical_col = data[:, col]
        vals = set(categorical_col)

        # For each column in cols_to_add, set the value of the column to 1 if the corresponding data point belongs to the corresponding category, and 0 otherwise.
        cols_to_add = [[] * len(vals)]
        for j in vals:
            cols_to_add = [1 if val == j else 0 for val in categorical_col]

            # Append the cols_to_add list to the data_processed array and update the idx list.
            data_processed = np.column_stack((data_processed, cols_to_add))
            idx = idx + [col]

    if return_idx:  return data_processed, idx
    else: return data_processed