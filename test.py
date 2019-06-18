import numpy as np
import time

from nn_algorithms import *
from sklearn.neighbors import NearestNeighbors

def main():
	n = 10000
	d = 20

	# X = np.array([i for i in range(n)]).reshape(n,1)
	# for j in range(1, d):
	# 	X_new = np.hstack([X, np.zeros((len(X),1))])
	# 	# print(X_new)
	# 	for i in range(1,n):
	# 		X_new = np.vstack([X_new, np.hstack([X, np.zeros((len(X),1)) + i])])
	# 	X = X_new
	# 	# print(X)
	X = np.random.normal(size=(n, d))
	print(X)
	print("=================")
	print("Dataset Created")
	print("=================")


	query = np.random.normal(size=(2, d))
	nn = NearestNeighbors()

	t0 = time.time()
	nn.fit(X)
	t1 = time.time()
	print("Vanilla NN Fit:", t1-t0)
	# print(nn.kneighbors(np.zeros(d)+1)[0])
	t0 = time.time()
	nn_query = nn.kneighbors(query)[0]
	t1 = time.time()
	print("Vanilla NN Query:", t1-t0)

	lsh = LSHNeighbors()
	t0 = time.time()
	lsh.fit(X, hash="pcabinary")
	t1 = time.time()
	print("LSH Fit:", t1-t0)

	t0 = time.time()
	lsh_query = lsh.kneighbors(query)
	t1 = time.time()
	print("LSH Query:", t1-t0)

	print("\nLSH Distance:", lsh_query[0][0][0])
	print("True Best Distance:", nn_query[0][0])

if __name__ == "__main__":
	main()