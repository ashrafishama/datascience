import matplotlib.pyplot as plt
import numpy as np


n = np.arange(0.,5.,0.2)

plt.plot(n,n,'r--',n,n**2,'bs',n,n**3,'g^')
plt.show()

names = ['groupA','groupB','groupC']
vals = [1,10,100]

plt.figure(figsize=(9,6))

plt.subplot(131)
plt.bar(names,vals) #bar, like histograms
plt.subplot(132)
plt.scatter(names,vals) #scatter, like points
plt.subplot(133)
plt.plot(names,vals) #plot, like a joint line
plt.suptitle('Categorical Plotting')
plt.show()
