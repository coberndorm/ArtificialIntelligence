import numpy as np
import MLP.clustering as cl

# Calculate the Euclidean distance between two numpy arrays 'x' and 'y'
def euclidean(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(np.sum((x - y) ** 2))

# Calculate the Manhattan distance between two numpy arrays 'x' and 'y'
def manhattan(x: np.ndarray, y: np.ndarray) -> float:
    return np.sum(np.abs(x - y))

# Calculate the Lp (p-norm) distance between two numpy arrays 'x' and 'y' with an optional parameter 'p' (default is 1)
def lp(x: np.ndarray, y: np.ndarray, p=1) -> float:
    if p == "inf":
        return np.max(np.abs(x - y))
    return np.sum(np.abs(x - y) ** p) ** (1 / p)

# Calculate the cosine similarity or cosine distance between two numpy arrays 'x' and 'y' based on the 'similarity' flag
def cosine(x: np.ndarray, y: np.ndarray, similarity=False) -> float:
    if similarity:
        return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))
    else:
        return 1 - np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))
    
    # Calculate the Mahalanobis distance between two numpy arrays 'x' and 'y' using the inverse covariance matrix 'cov_inv'
def mahalanobis(x: np.ndarray, y: np.ndarray, cov_inv: np.ndarray) -> float:
    delta = x - y
    return np.sqrt(np.dot(np.dot(delta.T, cov_inv), delta))

# Calculate Mahalanobis distances between data points in 'data' using either sample covariance or Median Absolute Deviation (MAD)
def mahalanobis_distances(data: np.ndarray, type='cov') -> np.ndarray:

    assert type in ['cov', 'MAD'], "type must be 'cov' or 'MAD'"

    n = data.shape[0]
    m = data.shape[1]
    distances = np.zeros((n, n))

    if type == 'cov':
        # Calculate the sample covariance matrix and its inverse
        cov = np.array([[np.sum((x - np.mean(x)) * (y - np.mean(y))) for x in data.T] for y in data.T]) / (n - 1)
        cov_inv = np.linalg.inv(cov)

    elif type == 'MAD':
        # Calculate the covariance matrix using Median Absolute Deviation and its inverse
        cov = np.array([[np.sum((x - np.median(x)) * (y - np.median(y))) for x in data] for y in data]) / (n - 1)
        cov_inv = np.linalg.inv(cov)

    for i in range(n):
        for j in range(i, n):
            distance = mahalanobis(data[i], data[j], cov_inv)
            distances[i, j] = distance
            distances[j, i] = distance

    return distances

# Calculate a distance matrix between data points in 'data_X' and 'data_Y' using a specified metric
def distance_matrix(data_X: np.ndarray, data_Y = None, metric = "euclidean", parameter = None) -> np.ndarray:
    ''' 
    Calculate a distance matrix between data points in 'data_X' and 'data_Y' using a specified metric.
    This function supports several distance metrics, including Euclidean, Manhattan, Lp, Cosine, and Mahalanobis (not recommended).
    The function checks for valid metric names and disallows using the "mahalanobis" metric (use "mahalanobis_distances" instead).
    Parameters:
    - data_X: A numpy array containing data points (shape: (n, d))
    - data_Y: An optional numpy array containing data points (shape: (m, d)). If None, calculate distances within data_X.
    - metric: A string specifying the distance metric to use (default is "euclidean").
    - parameter: An optional parameter to use with some distance metrics (e.g., p value for Lp distance).
    Output:
    - A numpy array representing the distance matrix between data points in data_X and data_Y (shape: (n, m)).
    '''

    assert metric in ["euclidean", "manhattan", "lp", "cosine", "mahalanobis"], "Metric not supported"
    assert metric != "mahalanobis", "Function does not suppor mahal distances, call mahalanobis_distances"

    if metric == "euclidean": metric = euclidean
    elif metric == "manhattan": metric = manhattan
    elif metric == "lp": metric = lp
    elif metric == "cosine": metric = cosine

    # Calculate the distance matrix between data points in 'data' using the specified metric
    def distance_matrix_self(data: np.ndarray, metric) -> np.ndarray:
        n = data.shape[0]
        distances = np.zeros((n,n))
        for i in range(n):
            for j in range(i,n):
                if parameter == None: distance = metric(data[i],data[j])
                else: distance = metric(data[i],data[j], parameter)
                distances[i,j] = distance
                distances[j,i] = distance

        return distances
    
    # Calculate the distance matrix between data points in 'data_X' and 'data_Y' using the specified metric    
    def distance_matrix_compare(data_X: np.ndarray, data_Y: np.ndarray, metric) -> np.ndarray:
        n = len(data_X); m = len(data_Y)
        distances = np.zeros((n,m))
        for i in range(n):
            for j in range(m):
                if parameter == None: distance = metric(data_X[i],data_Y[j])
                else: distance = metric(data_X[i],data_Y[j], parameter)
                distances[i,j] = distance

        return distances
        
    if data_Y is not None:
        return distance_matrix_compare(data_X, data_Y, metric)
    else:
        return distance_matrix_self(data_X, metric)

# Rearrange a distance matrix 'dist' to have sorted rows and columns
def sort_dist_matrix_distance(dist: np.ndarray) -> np.ndarray:
    """
    Sort a distance matrix 'dist' based on distances between data points and return the sorted matrix.

    Parameters:
        - dist: A numpy array representing a distance matrix (shape: (n, m)).

    Returns:
        - A new numpy array representing the sorted distance matrix (shape: (n, m)).
    """
    n, m = dist.shape
    new_axis_x = []  # Store the sorted order of rows
    new_axis_y = []  # Store the sorted order of columns
    values_used_x = set()  # Keep track of row indices used
    values_used_y = set()  # Keep track of column indices used

    # Create a list of tuples (i, j, distance) for all pairs (i, j) in the distance matrix
    # Assign a large distance (e.g., 1000) for the diagonal elements (i == j) to exclude them from sorting
    vector_dist = [(i, j, dist[i, j]) if i != j else (i, j, 1000) for i in range(n) for j in range(m)]

    # Sort the list of tuples based on the distance value
    vector_dist = sorted(vector_dist, key=lambda x: x[2])
    counter = 0

    # Iterate until all row and column indices are included in the sorted order
    while len(values_used_x) != n or len(values_used_y) != m:
        if vector_dist[counter][0] not in values_used_x:
            new_axis_x.append(vector_dist[counter][0])
            values_used_x.add(vector_dist[counter][0])

        if vector_dist[counter][1] not in values_used_y:
            new_axis_y.append(vector_dist[counter][1])
            values_used_y.add(vector_dist[counter][1])

        counter += 1

    # Create a new distance matrix with sorted rows and columns
    new_dist_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            new_dist_matrix[i, j] = dist[new_axis_x[i], new_axis_y[j]]

    return new_dist_matrix


# Sort a distance matrix 'dist' based on given boxes or divide it into boxes if 'boxes' is None.
def sort_dist_matrix_boxes(dist: np.ndarray, boxes=None, div=4) -> np.ndarray:
    """
    Sort a distance matrix 'dist' based on given boxes or divide it into boxes if 'boxes' is None.
    
    Parameters:
        - dist: A numpy array representing a distance matrix (shape: (n, m)).
        - boxes: A dictionary containing pre-defined boxes for sorting. If None, the function divides the matrix into boxes.
        - div: The number of divisions to use when dividing the matrix into boxes.

    Returns:
        - A new numpy array representing the sorted distance matrix (shape: (n, m)).
    """

    # Create a new distance matrix based on the sorted order of rows and columns
    def new_dist_matrix(dist, new_axis_x, new_axis_y, n, m):
        new_dist_matrix = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                new_dist_matrix[i, j] = dist[new_axis_x[i], new_axis_y[j]]

        return new_dist_matrix
    
    # Use the provided 'boxes' or divide the distance matrix into boxes
    box = cl.boxes(dist, div, symmetrical=False) if boxes is None else boxes
    n, m = dist.shape
    new_axis_x = []  # Store the sorted order of rows
    new_axis_y = []  # Store the sorted order of columns
    values_used_x = set()  # Keep track of row indices used
    values_used_y = set()  # Keep track of column indices used
    amount_boxes = len(box)
    
    # Create arrays to store the index of the main box for each row and column
    index_main_box_x = np.zeros((n, amount_boxes))
    index_main_box_y = np.zeros((m, amount_boxes))

    # Populate the index arrays based on the boxes
    for key, val in box.items():
        for point in val:
            index_main_box_x[point[0]][key] += 1
            index_main_box_y[point[1]][key] += 1

    # Determine the main box index for each row and column
    index_main_box_x = [np.argmax(x) for x in index_main_box_x]
    index_main_box_y = [np.argmax(y) for y in index_main_box_y]

    # Iterate through the boxes and add row and column indices to the sorted order
    for key, val in box.items():
        for point in val:
            if point[0] not in values_used_x and int(key) == int(index_main_box_x[point[0]]):
                new_axis_x.append(point[0]); values_used_x.add(point[0])

            if point[1] not in values_used_y and int(key) == int(index_main_box_x[point[1]]):
                new_axis_y.append(point[1]); values_used_y.add(point[1])

            # If all rows and columns have been included, return the new sorted distance matrix
            if len(values_used_x) == n and len(values_used_y) == m:
                return new_dist_matrix(dist, new_axis_x, new_axis_y, n, m)


