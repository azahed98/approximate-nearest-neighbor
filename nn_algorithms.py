import numpy as np
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.hashes import PCABinaryProjections
from sklearn.neighbors import NearestNeighbors

class LSHNeighbors(NearestNeighbors):
    def __init__(self, n_neighbors=5, 
                       radius=1.0, 
                       algorithm='auto', 
                       leaf_size=30, 
                       metric='minkowski', 
                       p=2, 
                       metric_params=None, 
                       n_jobs=None, 
                       **kwargs):
        super(LSHNeighbors, self).__init__(n_neighbors=n_neighbors, 
                                           radius=radius, 
                                           algorithm=algorithm, 
                                           leaf_size=leaf_size, 
                                           metric=metric, 
                                           p=p, 
                                           metric_params=metric_params, 
                                           n_jobs=n_jobs, 
                                           **kwargs)

    def fit(self, X, y=None, hash="randbinary"):
        X = np.array(X)
        assert len(X.shape) == 2, "X not 2-rank"
        dimension = X.shape[-1]
        if hash == "randbinary":
          rbp = RandomBinaryProjections('rbp', 10)
        elif hash == "pcabinary":
          rbp = PCABinaryProjections('rbp', 10, training_set=X)
        self.engine = Engine(dimension, lshashes=[rbp])
        index = 0
        for x in X:
            self.engine.store_vector(x, str(index))
            index += 1
            # count += index

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        if len(X.shape) == 1:
          results = self.engine.neighbours(X)
          # dists = [elem[2] for elem in results]
          dists = np.array([np.linalg.norm(X - elem[0]) for elem in results])
          indices = np.array([int(elem[1]) for elem in results])
          vectors = np.array([elem[0] for elem in results])
          return (dists, indices, vectors)
        elif len(X.shape) == 2:
          results = [self.engine.neighbours(x) for x in X]
          dists = np.array([[np.linalg.norm(X - elem[0]) for elem in result] for result in results])
          indices = np.array([[int(elem[1]) for elem in result] for result in results])
          vectors = np.array([[elem[0] for elem in result] for result in results])
          return (dists, indices, vectors)
        else:
          raise ValueError('X has rank higher than 2')

    def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity'):
        # Almost definitely don't need this
        raise NotImplementedError

    def radius_neighbors(self, X=None, radius=None, return_distance=True):
        raise NotImplementedError

    def radius_neighbors_graph(self, X=None, radius=None, mode='connectivity'):
        # Don't need this either
        raise NotImplementedError

    def set_params(self, **params):
        raise NotImplementedError


class DCINeighbors(NearestNeighbors):
    def __init__(self, n_neighbors=5, 
                       radius=1.0, 
                       algorithm='auto', 
                       leaf_size=30, 
                       metric='minkowski', 
                       p=2, 
                       metric_params=None, 
                       n_jobs=None, 
                       **kwargs):
        super(DCINeighbors, self).__init__(n_neighbors=n_neighbors, 
                                           radius=radius, 
                                           algorithm=algorithm, 
                                           leaf_size=leaf_size, 
                                           metric=metric, 
                                           p=p, 
                                           metric_params=metric_params, 
                                           n_jobs=n_jobs, 
                                           **kwargs)

    def fit(self, X, y=None):
        raise NotImplementedError

    def get_params(self, deep=True):
        raise NotImplementedError

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        raise NotImplementedError

    def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity'):
        # Almost definitely don't need this
        raise NotImplementedError

    def radius_neighbors(self, X=None, radius=None, return_distance=True):
        raise NotImplementedError

    def radius_neighbors_graph(self, X=None, radius=None, mode='connectivity'):
        # Don't need this either
        raise NotImplementedError

    def set_params(self, **params):
        raise NotImplementedError
