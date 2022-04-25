#! /usr/bin/env python3
from typing import List
import numpy as np
from numpy.typing import NDArray


class Cluster():
    def __init__(self, data: List[NDArray]):
        self.n = len(data)
        self.data = data
        self.centroid: NDArray = np.mean(self.data, axis=0)

    def __str__(self):
        out = ""
        out += "\nn: " + self.n.__repr__()
        out += "\ndata: " + self.data.__repr__()
        out += "\ncentroid: " + self.centroid.__repr__()
        return out

    def __len__(self):
        return self.data.__len__()

    def pop(self, index: int):
        element = self.data.pop(index)

        self.centroid = (self.centroid * self.n - element) / (self.n - 1)
        self.n -= 1
        return element

    def append(self, element: NDArray):
        self.data.append(element)

        self.centroid = (self.centroid * self.n + element) / (self.n + 1)
        self.n += 1
        return self.n

    def peek(self, index: int):
        """
        preview cluster without indexed element
        """
        element = self.data[index]

        centroid = (self.centroid * self.n - element) / (self.n - 1)
        return element, centroid

    def peek_with(self, element: NDArray):
        """
        preview cluster with element
        """
        return (self.centroid * self.n + element) / (self.n + 1)


class KMeans():
    def __init__(self, data: List[NDArray], k=4):
        self._k = k
        np.random.shuffle(data)
        self.data = data
        splits = np.array_split(self.data, self._k)
        splits = [[row for row in cluster] for cluster in splits]
        self.clusters = list(map(
            lambda cluster: Cluster(cluster),
            splits))

    def __str__(self):
        out = ""
        for cluster in self.clusters:
            out += cluster.__str__()
        return out

    def objective_function(self, cluster):
        centroid = np.average(cluster, axis=0)
        diff = np.subtract(cluster, centroid)
        return np.sum(np.square(diff))

    def hart_wong_update(self):
        def optimizer(x: NDArray, c_n: Cluster, c_m: Cluster):
            n_len = len(c_n)
            m_len = len(c_m)

            diff_n = np.sum(np.square(c_n.centroid - x))
            diff_m = np.sum(np.square(c_m.centroid - x))

            return n_len * diff_n / (n_len - 1) - m_len * diff_m / (m_len + 1)

        best = [0, 0, 0, 0]
        for c_n in self.clusters:
            for (i, point) in enumerate(c_n.data):
                for c_m in self.clusters:
                    if c_n != c_m:
                        err = optimizer(point, c_n, c_m)
                        if err > best[0]:
                            best = [err, i, c_n, c_m]

        if best[0] != 0:
            _, i, c_n, c_m = best
        c_m.append(c_n.pop(i))


if __name__ == "__main__":
    def random_point():
        return np.random.randint(0, high=10, size=(1, 2))


    points = []
    for _ in range(25):
        points.append(random_point())
    for _ in range(25):
        points.append(random_point() + 15)

    clustering = KMeans(points, k=2)

    print(points)
    print(clustering)
    for _ in range(50):
        clustering.hart_wong_update()
    print(clustering)
