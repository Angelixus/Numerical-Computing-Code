import numpy as np
import matplotlib.pyplot as plt

"""
We are going to do kmeans, this algorithm clusterized the data, we can define the number of clusters, the optimum is trial and error
"""
np.random.seed(7)

x1 = np.random.standard_normal((100,2))*0.6+np.ones((100,2))
x2 = np.random.standard_normal((100,2))*0.5-np.ones((100,2))
x3 = np.random.standard_normal((100,2))*0.4-2*np.ones((100,2))+5
X = np.concatenate((x1,x2,x3),axis=0)

plt.plot(X[:,0],X[:,1],'k.')
plt.show()

k = 3

#%% STEP 1: Initialize centroids
# a = m + a1 (M - m) this allows us to go from the range [0, 1] to [m, M] (our centroids are on 0, 1)
mx, my = np.min(X, axis=0)
Mx, My = np.max(X, axis=0)
centroids = np.random.rand(k, X.shape[1]) #Centroids, objects similar to the ones we want to clusterize
print(centroids)
centroids[:, 0] = mx + centroids[:,0]*(Mx-mx)
centroids[:, 1] = my + centroids[:, 1]*(My - my)

plt.plot(X[:,0],X[:,1],'k.')
plt.plot(centroids[:,0],centroids[:,1],'mo',markersize=8, label='centroids')

plt.show()
#%% STEP 2: Assign centroids, labels
""" For each point (values on X):
    Calculate the euclidean distance between the point and the centroids
    Calculate the minimum between these 3 distances
"""
labels = np.zeros_like(X[:,1])
for i in range(len(labels)):
    d0 = ((X[i, 0] - centroids[0][0])**2) + ((X[i, 1] - centroids[0][1])**2) # Distance between point and first centroid
    d1 = ((X[i, 0] - centroids[1][0])**2) + ((X[i, 1] - centroids[1][1])**2) # Distance between point and second centroid
    d2 = ((X[i, 0] - centroids[2][0])**2) + ((X[i, 1] - centroids[2][1])**2) # Distance between point and third centroid

    labels[i] = np.argmin([d0, d1, d2]) # argmin te dice el indice donde se situa el minimo, labels guarda a que cluster pertenece cada punto

plt.plot(X[labels==0,0],X[labels==0,1],'r.', label='cluster 1')
plt.plot(X[labels==1,0],X[labels==1,1],'b.', label='cluster 2')
plt.plot(X[labels==2,0],X[labels==2,1],'g.', label='cluster 3')

plt.plot(centroids[:,0],centroids[:,1],'mo',markersize=8, label='centroids')

plt.legend(loc='best')
plt.show()

#%% STEP 3: Recalculate the centroids
"""
The x coordinate of each is np.mean(X[labels==0, i]) having i the index of the given cluster (Optimize the cluster)
We have to move them because the centroid is not in the center of the cluster because they where initialized randomly
With the mean we are moving the centroid to the center of its assigned cluster, since we move centroids, we have to reasign the points to their new best cluster
"""

for n in range(7):
    centroids[0] = np.mean(X[labels==0]) ## labels==0 is a boolean matrix that puts true when the value at the row is 0
    centroids[1] = np.mean(X[labels==1])
    centroids[2] = np.mean(X[labels==2])

    plt.plot(X[labels==0,0],X[labels==0,1],'r.', label='cluster 1')
    plt.plot(X[labels==1,0],X[labels==1,1],'b.', label='cluster 2')
    plt.plot(X[labels==2,0],X[labels==2,1],'g.', label='cluster 3')

    plt.plot(centroids[:,0],centroids[:,1],'mo',markersize=8, label='centroids')
    plt.legend()
    plt.show()

    for i in range(len(labels)):
        d0 = ((X[i, 0] - centroids[0][0])**2) + ((X[i, 1] - centroids[0][1])**2) # Distance between point and first centroid
        d1 = ((X[i, 0] - centroids[1][0])**2) + ((X[i, 1] - centroids[1][1])**2) # Distance between point and second centroid
        d2 = ((X[i, 0] - centroids[2][0])**2) + ((X[i, 1] - centroids[2][1])**2) # Distance between point and third centroid

        labels[i] = np.argmin([d0, d1, d2]) # argmin te dice el indice donde se situa el minimo, labels guarda a que cluster pertenece cada punto