import matplotlib.pyplot as plt
import numpy as np

n = 1000 #no of random samples
x = np.random.rand(n,2)
insider = x[(np.square(x[:,0])+np.square(x[:,1])<=1)]
print(np.size(insider,0))
plt.scatter(x[:,0],x[:,1],s=1,c='blue')
plt.scatter(insider[:,0],insider[:,1],s=1,c='red')
plt.show()
x = np.square(x)
x = np.sum(x,axis=1)
square_points = 0
circle_points = 0

for i in range(0,n):
    if(x[i]<=1):
        circle_points+=1
        square_points+=1
    else:
        square_points+=1

pi = 4*(circle_points/square_points)
print(pi)
