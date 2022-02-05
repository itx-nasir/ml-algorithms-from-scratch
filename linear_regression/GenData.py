# Import Libraries
import numpy  as np
import random as rd
import  pandas as pd
from matplotlib import pyplot

# ***********************************#

# Generate array of random floats

x = np.random.uniform(-8, 8, 100)
# initialize m and c
m = 1.4
c = 0.3
# Sort the array
x.sort()
# create an array for storing values of target variable
y = np.zeros(100)

# Find y for every value of x
for i in range(100):
    y[i] = m * x[i] + c + (rd.uniform(-2,2))


xdf=pd.DataFrame(x)
ydf=pd.DataFrame(y)
df=pd.DataFrame(columns=['x','y'])
df['x']=xdf[0]
df['y']=ydf[0]

#Save data to csv
df.to_csv('Data.csv',index=False)
print(df)
# Plot the generated dataset
pyplot.scatter(x, y)
pyplot.plot([min(x), max(x)], [min(y), max(y)], color='red')


# Save the plot
pyplot.savefig('plot.png')
pyplot.show()
