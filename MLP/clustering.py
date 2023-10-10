import numpy as np

def generate_grid_vertices(dim: int, partitions: float, a = 0, b = 1) -> list(tuple()):
    """
    Generate all vertices in an n-dimensional grid with specified partitions.

    Parameters:
        - dim: The number of dimensions.
        - partitions: The distance between each vertex.
        - a: The lower bound of the grid.
        - b: The upper bound of the grid.

    Returns:
        - A list of tuples representing the coordinates of all grid vertices.
    """

    # Base case: 0-dimensional grid
    if dim == 0:
        return [()]
    # Recursive case: n-dimensional grid
    else:
        # Generate the vertices of an (n-1)-dimensional grid
        lower_dimension = generate_grid_vertices(dim - 1, partitions)
        vertices = []
        # Add vertices to the grid
        for coord in lower_dimension:
            # Add vertices in each dimension
            for offset in [i*((b-a)/(partitions-1)) for i in range(partitions)]:
                new_coord = coord + (offset,)
                vertices.append(new_coord)
        return vertices
    
# Sorts the different points distances into boxes.
def boxes(dist: np.ndarray, div=4, step = 0.5, sort="boxes", symmetrical=True) -> dict:
    """
    Sorts the different points' distances into boxes.

    Parameters:
        - dist: The distance matrix.
        - div: The number of boxes.
        - sort: Whether to create the boxes by a maximum distance ("distance") or amount of boxes ("boxes", default).
        - symmetrical: Whether the distance matrix is symmetrical (default is True).

    Returns:
        - boxes: A dictionary with the boxes as keys and the points in the boxes as values (list of tuples (i, j, dist)).
    """

    assert sort in ["boxes", "distance"], "The 'sort' parameter must be either 'boxes' or 'distance'."

    n, m = dist.shape
    max_dist = np.max(dist)
    min_dist = np.min(dist)
    epsilon = 0.000000000001

    # Calculate the step size
    if sort == "boxes": 
        step = (max_dist - min_dist) / div + epsilon
    else: 
        step = step + epsilon

    # If the distance matrix is not square, it's not symmetrical
    symmetrical = False if n != m else symmetrical

    boxes = dict()
    for i in range(n):
        # Start from i+1 if the matrix is symmetrical to avoid duplicate entries
        start = i + 1 if symmetrical else 0
        for j in range(start, m):
            # Calculate the box for the current distance
            box = int(np.floor((dist[i, j] - min_dist) / (step)))
            
            # Add the point to the box
            if box not in boxes.keys():
                boxes[box] = [(i, j, dist[i, j])]
            else:
                boxes[box].append((i, j, dist[i, j]))

    boxes_keys = list(boxes.keys())

    # Reorganize the boxes into a dictionary with sequential keys to start from 0
    boxes = {i: boxes[boxes_keys[i]] for i in range(len(boxes_keys))}
    
    # Sort the points in each box based on the distance value
    return {k: sorted(v, key=lambda x: x[2]) for k, v in boxes.items()}
            
# Function to cluster points based on distance matrix by dividing them into boxes
def boxes_cluster(dist: np.ndarray, clusters=4, step = 0.5, sort="boxes", symmetrical=True):
    """
    Cluster points based on a distance matrix by dividing them into boxes.

    Parameters:
        - dist: The distance matrix.
        - div: The number of boxes.
        - sort: Whether to create the boxes by a maximum distance ("distance") or amount of boxes ("boxes", default).
        - symmetrical: Whether the distance matrix is symmetrical (default is True).

    Returns:
        - clt: A dictionary where keys are cluster identifiers and values are lists of point indices.
    """

    assert sort in ["boxes", "distance"], "The 'sort' parameter must be either 'boxes' or 'distance'."

    n = len(dist)

    # Divide the distance matrix into boxes to assist in clustering
    box = boxes(dist, clusters, step, sort, True)
    clusters = len(box) if len(box) < clusters else clusters
    
    amount_boxes = len(box)
    main_box = np.zeros((n, amount_boxes))

    # Count how many times each point appears in a box
    for key, val in box.items():
        for point in val:
            main_box[point[0]][key] += 1

    clt = {i: [] for i in range(clusters)}

    # Assign points to clusters based on their appearances in boxes
    for point, appearances in enumerate(main_box):
        clt[np.argmax(appearances)].append(point)

    # Sort clusters by size (largest first)
    clt_sorted = sorted(clt, key=lambda k: -len(clt[k]))

    # Reorganize the clusters into a dictionary with sequential keys
    clt = {i: clt[clt_sorted[i]] for i in range(clusters)}

    return clt


# Function to cluster points based on their distances in a distance matrix
def nearby_cluster(dist, epsilon=0.1, max_clusters=4):
    """
    Cluster points based on their distances in a distance matrix.

    Parameters:
        - dist: The distance matrix (points to themselves).
        - epsilon: The maximum distance to consider two points as neighbors (default is 0.1).
        - max_clusters: The maximum number of clusters to create (default is 4).

    Returns:
        - clusters: A dictionary where keys are cluster identifiers and values are sets of point indices.
    """
    # Initialize an empty dictionary to store clusters
    clusters = dict()

    # Get the number of points and create a set of all point indices
    points_len = len(dist)
    points = set(range(0, points_len))

    current_cluster = 0

    # Continue clustering until all points are assigned to clusters
    while len(points) != 0:
        # Choose a random starting point
        start_point = np.random.choice(list(points))

        # Initialize the current cluster with the points that are within 'epsilon' of the starting point
        clusters[current_cluster] = {x for x in np.where(dist[start_point] < epsilon)[0]}
        current_cluster_checked = {start_point}

        # Continue expanding the current cluster until no new points can be added
        while len(current_cluster_checked) != len(clusters[current_cluster]):
            currently_checking = clusters[current_cluster].copy()
            for i in currently_checking:
                if i not in current_cluster_checked:
                    close_to_i = {x for x in np.where(dist[i] < epsilon)[0]}
                    clusters[current_cluster].update(close_to_i)
                    current_cluster_checked.add(i)

        # Remove points that have been assigned to a cluster from the set of unassigned points
        points.difference_update(clusters[current_cluster])
        current_cluster += 1

    # If there are more clusters than 'max_clusters', merge the smallest clusters
    while len(clusters) > max_clusters:
        # Find the smallest cluster by the number of points
        smallest_cluster = min(clusters, key=lambda k: len(clusters[k]))

        for i in clusters[smallest_cluster]:
            stop = False
            idx = 1

            while not stop:
                stop = True

                sets_containing_value = []
                # Find the closest point to 'i' that is not in the same set
                closest_point = np.argsort(dist[i])[idx]

                for key, clust in clusters.items():
                    if closest_point in clust:
                        sets_containing_value.append(key)
                        clusters[key].add(i)

                if sets_containing_value == [smallest_cluster]:
                    stop = False
                    idx += 1

        # Remove the smallest cluster
        clusters.pop(smallest_cluster)

    # Sort clusters by size (largest first)
    clusters_sorted = sorted(clusters, key=lambda k: -len(clusters[k]))

    # Reorganize the clusters into a dictionary with sequential keys
    clusters = {i: clusters[clusters_sorted[i]] for i in range(len(clusters_sorted))}

    return clusters


import numpy as np

def DBSCAN_cluster(dist, epsilon, min_samples):
    """
    Cluster data points using the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm.

    Parameters:
        - dist: 2D numpy array, a distance matrix where dist[i, j] represents the distance between data point i and j.
        - epsilon: float, maximum distance between two points for one to be considered as in the neighborhood of the other.
        - min_samples: int, the minimum number of data points required to form a dense region (cluster).

    Returns:
        - clusters: dict, a dictionary where keys represent cluster labels, and values are sets of data points in each cluster.
        - Noise points are assigned to cluster label -1.

    Note:
        - This function operates on a distance matrix of data points to themselves.
    """

    # Define a function to find neighbors within a given distance epsilon
    find_neighbors = lambda x: np.where(x <= epsilon)[0]
    n = dist.shape[0]  # Number of data points
    
    # Create a dictionary to store the clusters
    clusters = dict()
    
    # Initialize auxiliary variables
    points_len = len(dist)
    points = set(range(0, points_len))
    current_cluster = 1  # Start with the first cluster
    
    # Iterate over each data point
    for point in range(n):
        if point not in points:
            continue  # Skip if this point has been assigned to a cluster already
        
        # Find the neighbors of the current point within the epsilon distance
        neighbors = find_neighbors(dist[point])
        
        # Check if the current point has enough neighbors to form a cluster
        if len(neighbors) >= min_samples:
            # Initialize a new cluster and add the neighbors to it
            clusters[current_cluster] = {x for x in neighbors}
            current_cluster_checked = {point}  # Mark the current point as checked
            
            # Continue to expand the cluster until no more points can be added
            while len(current_cluster_checked) != len(clusters[current_cluster]):
                currently_checking = clusters[current_cluster].copy()
                for i in currently_checking:
                    if i not in current_cluster_checked:
                        # Find points close to 'i' and add them to the cluster
                        close_to_i = {x for x in np.where(dist[i] <= epsilon)[0]}
                        clusters[current_cluster].update(close_to_i)
                        current_cluster_checked.add(i)
            
            # Remove the points in this cluster from the unassigned points set
            points.difference_update(clusters[current_cluster])
            current_cluster += 1  # Move to the next cluster
    
    # Assign unassigned points to a noise cluster (-1)
    clusters[-1] = points
    
    return clusters
