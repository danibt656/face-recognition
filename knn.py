import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Generate random set of blue & red dots
trainData = np.random.randint(0, 100, (25,2)).astype(np.float32)

responses = np.random.randint(0,2,(25,1)).astype(np.float32)
# Assign colors (blue=1, red=0)
red = trainData[responses.ravel() == 0]
plt.scatter(red[:,0], red[:,1], 80, 'r', '^')
blue = trainData[responses.ravel() == 1]
plt.scatter(blue[:,0], blue[:,1], 80, 'b', 's')

# Create a new value and try to classify it
newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
plt.scatter(newcomer[:,0], newcomer[:,1], 80, 'g', 'o')

knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
ret, res, neigh, dist = knn.findNearest(newcomer, 3)

# Get farest neighbor
# max_dist = max(dist[0]).astype(np.float32)//10
# circle = plt.Circle(newcomer[0], radius=max_dist, color='r', fill=False)
# plt.gca().add_patch(circle)

print(f"result: {res}")
print(f"neighbors: {neigh}")
print(f"distance: {dist}")

plt.show()

