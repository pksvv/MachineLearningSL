
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def printing_db(X, y, model):
    markers = ['x','o','*']
    colors = ['red','blue','green']
    cmap = ListedColormap(colors[:len(np.unique(y))])
    res = 0.02
    #Plot regions
    x1min, x1max = X[:,0].min() -1, X[:,0].max() + 1
    x2min, x2max = X[:,1].min() -1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x1min,x1max,res),np.arange(x2min,x2max,res))

    output = model.predict(np.array([xx.ravel(), yy.ravel()]).T)
    output = output.reshape(xx.shape)
    plt.figure(figsize=(7,7))
    plt.pcolormesh(xx,yy, output, cmap=plt.cm.GnBu_r)

    #PLOT ALL SAMPLES
    for index, item in enumerate(np.unique(y)):
        plt.scatter(x=X[y == item, 0], y=X[y == item, 1],alpha=0.8, c=cmap(index),
        marker=markers[index], label=item)

    plt.xlabel('Petal length std')
    plt.ylabel('Petal width std')

    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    
    plt.legend(loc='best')
    plt.show()


