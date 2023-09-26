import numpy as np

def euclidean(x:np.ndarray , y:np.ndarray) -> float:
    return np.sqrt(np.sum((x-y)**2))

def euclidean_distances(data:np.ndarray) -> np.ndarray:
    n = data.shape[0]
    distances = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            distance = euclidean(data[i],data[j])
            distances[i,j] = distance
            distances[j,i] = distance

    return distances

def mahalanobis(x:np.ndarray, y:np.ndarray, cov_inv:np.ndarray) -> float:
    delta = x-y
    return np.sqrt(np.dot(np.dot(delta.T, cov_inv), delta))

def mahalanobis_distances(data:np.ndarray, type='cov') -> np.ndarray:

    assert type in ['cov', 'MAD'], "type must be 'cov' or 'MAD'"

    n = data.shape[0]; m = data.shape[1]
    distances = np.zeros((n,n))

    if type == 'cov':
        cov =  np.array([[np.sum((x - np.mean(x)) * (y - np.mean(y))) for x in data.T] for y in data.T]) / (n-1)
        cov_inv = np.linalg.inv(cov)
        

    elif type == 'MAD':
        cov = np.array([[np.sum((x - np.median(x)) * (y - np.median(y))) for x in data] for y in data]) / (n-1)
        cov_inv = np.linalg.inv(cov)

    for i in range(n):
        for j in range(i,n):
            distance = mahalanobis(data[i],data[j], cov_inv)
            distances[i,j] = distance
            distances[j,i] = distance

    return distances

def manhattan(x:np.ndarray, y:np.ndarray) -> float:
    return np.sum(np.abs(x-y))

def manhattan_distances(data: np.ndarray) -> np.ndarray:
    n = data.shape[0]
    distances = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            distance = manhattan(data[i],data[j])
            distances[i,j] = distance
            distances[j,i] = distance

    return distances

def lp(x:np.ndarray, y:np.ndarray, p=1) -> float:
    if p == "inf":
        return np.max(np.abs(x-y))
    return np.sum(np.abs(x-y)**p)**(1/p)

def lp_distances(data:np.ndarray, p=1) -> np.ndarray:

    assert p > 0, "p must be greater than 0"
    assert type(p) == int or p == "inf", "p must be an integer or \"inf\""

    n = data.shape[0]
    distances = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            distance = lp(data[i],data[j],p)
            distances[i,j] = distance
            distances[j,i] = distance

    return distances

def cosine(x:np.ndarray, y:np.ndarray, similarity = False) -> float:
    if similarity:
        return np.dot(x,y)/(np.sqrt(np.dot(x,x))*np.sqrt(np.dot(y,y)))
    else:
        return 1 - np.dot(x,y)/(np.sqrt(np.dot(x,x))*np.sqrt(np.dot(y,y)))

def cosine_distances(data:np.ndarray, similarity = False) -> np.ndarray:

    n = data.shape[0]
    distances = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            distance = cosine(data[i],data[j], similarity)
            distances[i,j] = distance
            distances[j,i] = distance

    return distances