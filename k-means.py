"""
The program implements k-means algorithm from unsupervised learning. Program creates data blobs using sklearn utility 
make_blobs. User is asked to input number of clusters, number of data instances and number of iterations. 
"""

import math
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


class Datum:
    def __init__(self, instance):
        self.instance = instance
        label: Datum = None


def find_k_means(
    data: list[Datum], centers: list[Datum], some_large_number: float
) -> any:
    for datum in data:
        diff_min: float = some_large_number
        for center in centers:
            diff: float = math.dist(datum.instance, center.instance)
            if diff < diff_min:
                datum.label = center
                diff_min = diff

    for center in centers:
        center_data: list = []

        for datum in data:
            if datum.label == center:
                center_data.append(datum.instance)

        if len(center_data) != 0:
            center.instance = [round(sum(x) / len(x), 2) for x in zip(*center_data)]

    return data, centers


if __name__ == "__main__":
    k: int = int(input("Chose number of blobs (e.g. 4): "))
    data_points: int = int(input("Chose number of data instances (e.g. 400): "))

    # Make k data blobs using sklearn make_blobs utility
    X, y_true = make_blobs(
        n_samples=data_points, centers=k, cluster_std=0.60, random_state=0
    )
    plt.scatter(X[:, 0], X[:, 1], s=20)
    plt.savefig("blobs.svg")
    print("Close picture to continue!")
    plt.show()

    # Make k random centers
    centers: list[Datum] = [
        Datum([round(random.random(), 2), round(random.random(), 2)]) for i in range(k)
    ]

    some_large_number: float = 5 * np.std(X)
    data = [Datum(datum) for datum in X.tolist()]
    num_of_iter: int = int(input("Chose number of iterations (e.g. 50): "))

    for i in range(num_of_iter):
        data, centers = find_k_means(data, centers, some_large_number)

    print("\t Final centers:")

    for center in centers:
        print(center.instance)

    centers = np.array([center.instance for center in centers])
    plt.scatter(X[:, 0], X[:, 1], s=20)
    plt.scatter(centers[:, 0], centers[:, 1], c="red", s=100, alpha=0.9)
    plt.savefig("clusters.svg")
    print("Close picture to continue!")
    plt.show()
