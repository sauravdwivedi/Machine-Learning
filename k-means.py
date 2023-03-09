"""
The program implements k-means algorithm from unsupervised learning. k is chosen 3, but can be tweaked.
cluster centers converge in very few iterations, so number of iterations is set to 3, but can be tweaked.
"""

import math
import random


class Datum:
    def __init__(self, instance):
        self.instance = instance
        label: Datum = None


def find_k_means(
    data: list[Datum], centers: list[Datum], some_large_number: int
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
        center.instance = [round(sum(x) / len(x), 2) for x in zip(*center_data)]

    return data, centers


if __name__ == "__main__":
    k: int = 3
    data_points: int = 10
    data: list[Datum] = [
        Datum([round(random.random(), 2), round(random.random(), 2)])
        for i in range(data_points)
    ]
    centers: list[Datum] = [
        Datum([round(random.random(), 2), round(random.random(), 2)]) for i in range(k)
    ]
    num_of_iter: int = 3
    some_large_number: float = 1

    for i in range(num_of_iter):
        data, centers = find_k_means(data, centers, some_large_number)
        print(f"\t Iteration {i+1}")
        for datum in data:
            print(f"Datum: {datum.instance} has center {datum.label.instance}")

    print("\t Final centers:")

    for center in centers:
        print(center.instance)
