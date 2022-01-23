import numpy as np

def my_cross_entropy(p,q):
    q_cor = list(map(lambda x: x + 1e-7, q))
    return -np.sum(p * np.log(q))

a = [0.7, 0.075, 0.075, 0.075, 0.075]
b = [0.2, 0.2, 0.2, 0.2, 0.2]
c = [0.4, 0.4, 0.05, 0.1, 0.05]
d = [0.5, 0.1, 0.1, 0.1, 0.2]
e = [0.6, 0.1, 0.1, 0.1, 0.1]
f = [0.8, 0.05, 0.05, 0.05, 0.05]
g = [0.9, 0.025, 0.025, 0.025, 0.025]

# a = [0.7, 0.3]
# b = [0.2, 0.8]
# c = [0.4, 0.6]
# d = [0.5, 0.5]
# e = [0.6, 0.4]
# f = [0.8, 0.2]
# g = [0.9, 0.1]

loss = my_cross_entropy(a, a)
print('Cross entorpy a,a:', loss)
loss = my_cross_entropy(a, b)
print('Cross entorpy a,b:', loss)
loss = my_cross_entropy(a, c)
print('Cross entorpy a,c:', loss)
loss = my_cross_entropy(a, d)
print('Cross entorpy a,d:', loss)
loss = my_cross_entropy(a, e)
print('Cross entorpy a,e:', loss)
loss = my_cross_entropy(a, f)
print('Cross entorpy a,f:', loss)
loss = my_cross_entropy(a, g)
print('Cross entorpy a,g:', loss)
