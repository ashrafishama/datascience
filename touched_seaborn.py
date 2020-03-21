import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

data = {'Name':['Ishita','Promi','Moumi'],'Age':[25,26,27]}

data = pd.DataFrame(data)
print(data)

with open('iris.csv') as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=',')
    line_count = 0
    for row in csv_reader:
        if(line_count==0):
            print(f'Attributes are {",".join(row)}')
            line_count +=1
        else:
            #print(f'{row[0]} {row[1]} {row[3]} {row[4]}')
            line_count +=1
    print(f'Number of instances are {line_count}.')

iris_dataset = pd.read_csv('iris.csv')
#print(iris_dataset)

#print(iris_dataset['sepallength'])
#print(np.average(iris_dataset['sepallength']))
#plt.matshow(iris_dataset.corr())
#plt.show()

#heat_map
corr = iris_dataset.corr()
heat_map = sb.heatmap(corr,vmin=-1,vmax=1,center=0,
                 cmap=sb.diverging_palette(128,255,n=200), square=True)

heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=45)
heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=45)
plt.show(heat_map)
