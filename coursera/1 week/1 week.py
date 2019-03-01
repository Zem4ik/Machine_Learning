import numpy as np

X = np.random.normal(loc= 1, scale=10, size=(2, 5))
print(X)

m0 = np.mean(X, axis=0)
m1 = np.mean(X, axis=1)
print("\n{} \n {}".format(m0, m1))

std0 = np.std(X, axis=0)
std1 = np.std(X, axis=1)
print("\n{} \n {}".format(std0, std1))

print(X - m0)
