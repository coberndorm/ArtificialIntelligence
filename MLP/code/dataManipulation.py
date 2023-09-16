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
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    
    # Normalize the matrix using Min-Max scaling.
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    
    return normalized_matrix