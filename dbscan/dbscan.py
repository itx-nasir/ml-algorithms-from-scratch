import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
import numpy.typing as npt

class DBSCAN:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) implementation
    
    Parameters:
    -----------
    eps : float
        The maximum distance between two samples for one to be considered neighbors
    min_points : int
        The minimum number of points required to form a dense region
    """
    def __init__(self, eps: float, min_points: int):
        self.eps = eps
        self.min_points = min_points
        self.labels = None
        self.n_clusters = 0
        
        # Point labels
        self.OUTLIER = 0
        self.UNASSIGNED = -1
        self.CORE = -2
        self.EDGE = -3

    def find_neighbors(self, data: np.ndarray, point_id: int) -> List[int]:
        """Find all points within eps distance of point_id"""
        neighbors = []
        for i in range(len(data)):
            if np.linalg.norm(data[i] - data[point_id]) <= self.eps:
                neighbors.append(i)
        return neighbors

    def fit(self, data: np.ndarray) -> List[int]:
        """
        Fit DBSCAN clustering from features matrix
        
        Parameters:
        -----------
        data : array-like of shape (n_samples, n_features)
            Training instances to cluster
            
        Returns:
        --------
        labels : list
            Cluster labels for each point in the dataset
        """
        # Standardize the features
        data = StandardScaler().fit_transform(data)
        n_points = len(data)
        self.labels = [self.UNASSIGNED] * n_points
        
        # Find neighbors for all points
        neighbors_list = [self.find_neighbors(data, i) for i in range(n_points)]
        
        # Identify core and edge points
        core_points = []
        for i in range(n_points):
            if len(neighbors_list[i]) >= self.min_points:
                self.labels[i] = self.CORE
                core_points.append(i)
            else:
                for neighbor in neighbors_list[i]:
                    if len(neighbors_list[neighbor]) >= self.min_points:
                        self.labels[i] = self.EDGE
                        break
        
        # Expand clusters
        cluster_id = 1
        for point in range(n_points):
            if self.labels[point] == self.CORE:
                self._expand_cluster(point, cluster_id, neighbors_list)
                cluster_id += 1
        
        self.n_clusters = cluster_id - 1
        return self.labels

    def _expand_cluster(self, point: int, cluster_id: int, neighbors_list: List[List[int]]) -> None:
        """Expand cluster from core point"""
        from collections import deque
        queue = deque([point])
        self.labels[point] = cluster_id
        
        while queue:
            current = queue.popleft()
            neighbors = neighbors_list[current]
            
            for neighbor in neighbors:
                if self.labels[neighbor] in [self.CORE, self.EDGE]:
                    if self.labels[neighbor] != cluster_id:
                        self.labels[neighbor] = cluster_id
                        if self.labels[neighbor] == self.CORE:
                            queue.append(neighbor)

def plot_clusters(data: np.ndarray, labels: List[int], title: str = "DBSCAN Clustering"):
    """Plot the clusters with different colors"""
    plt.figure(figsize=(10, 7))
    
    # Colors for plotting
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
    
    # Plot points
    for i in range(max(labels) + 1):
        if i == 0:  # Outliers
            mask = np.array(labels) == i
            plt.scatter(data[mask, 0], data[mask, 1], 
                       c='gray', label='Outliers', alpha=0.5)
        else:
            mask = np.array(labels) == i
            plt.scatter(data[mask, 0], data[mask, 1], 
                       c=colors[i % len(colors)], label=f'Cluster {i}')
    
    plt.title(title)
    plt.legend()
    plt.show()

def test_dbscan(epsilon: float, min_points: int, data: np.ndarray):
    """Test DBSCAN with given parameters"""
    print(f'Testing DBSCAN with epsilon = {epsilon}, min_points = {min_points}')
    
    # Create and fit DBSCAN
    dbscan = DBSCAN(eps=epsilon, min_points=min_points)
    labels = dbscan.fit(data)
    
    # Plot results
    plot_clusters(data, labels, 
                 f'DBSCAN Clustering (eps={epsilon}, min_points={min_points})')
    
    # Print statistics
    print(f'Number of clusters found: {dbscan.n_clusters}')
    print(f'Number of outliers found: {labels.count(0)}\n')

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('Moon.csv')
    X = df.drop(['Y'], axis=1).to_numpy()
    
    # Test with different parameters
    test_dbscan(0.08, 5, X)  # Low epsilon, high min_points
    test_dbscan(0.1, 3, X)   # Balanced parameters
    test_dbscan(0.4, 2, X)   # High epsilon, low min_points
