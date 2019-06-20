import numpy as np
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.hashes import PCABinaryProjections
from sklearn.neighbors import NearestNeighbors

from rbtree import RBTree


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
    """
    Dynamic Continuous Indexing: https://arxiv.org/pdf/1512.00442.pdf
    """
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



    def fit(self, X, m=25, L=2, y=None):
      assert len(X.shape) ==2, "X must be 2-rank"
      n = X.shape[0]
      d = X.shape[1]

      U = np.random.normal(size=(d, m * L))
      U = U / np.linalg.norm(U, axis=0, keepdims=True)
      T = [RBTree() for _ in range(m * L)]
      projected_data = np.matmul(X, U) #all the pbars
      for j in range(m):
        for l in range(L):
          for i in range(n):
            T[j + m * l].insert(KeyValeWrap(P[i][j + m * l], i))
      self.T = T
      self.X = X
      self.d = d
      self.n = n
      self.U = U
      self.L = L
      self.m = m
      self.P = projected_data

      raise NotImplementedError

    def kneighbors(self, X=None, n_neighbors=None, eps=1.0, return_distance=True):
        U = self.U
        C = np.zeros(shape=[self.L, self.n])
        qbar = np.matmul(U.T, X)
        S = [None for _ in range(self.L)]

        for i in range(self.n):
          for l in range(self.L):
            for j in range(self.m):
              


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


class KeyValWrap:
  def __init__(self, key, val):
    self.key = key
    self.val = val

   def __lt__(self,other):
        return (self.key<other.key)

    def __le__(self,other):
        return (self.key<=other.key)

    def __gt__(self,other):
        return (self.key>other.key)

    def __ge__(self,other):
        return (self.key>=other.key)





