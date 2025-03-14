import cProfile
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Example multivariate data
time_series_1 = np.random.rand(1000, 10)  # 1000 time steps, 10 features
time_series_2 = np.random.rand(1000, 10)  # 1000 time steps, 10 features


# Function to compute DTW distance for multivariate data
def compute_dtw_distance(ts1, ts2):
    distance = sum(
        fastdtw(np.array([ts1[:, i]]).T, np.array([ts2[:, i]]).T, dist=euclidean)[0]
        for i in range(ts1.shape[1])
    )
    return distance


# Profile the DTW computation
cProfile.run('compute_dtw_distance(time_series_1, time_series_2)')
