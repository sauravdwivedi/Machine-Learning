"""
The program implements k-means algorithm from unsupervised learning. Program creates data blobs using sklearn utility 
make_blobs. User is asked to input number of clusters, number of data instances and number of iterations. The resulting
animation with cluster centers pops up in Google Chrome. 
"""

import os
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
import glob
from PIL import Image
import contextlib


class Datum:
    def __init__(self, instance, label=None):
        self.instance = instance
        self.label = label


def label_data(
    data: list[Datum], centers: list[Datum], some_large_number: float
) -> None:
    for datum in data:
        diff_min: float = some_large_number
        for center in centers:
            diff: float = math.dist(datum.instance, center.instance)
            if diff < diff_min:
                datum.label = center
                diff_min = diff


def move_centers(data: list[Datum], centers: list[Datum]) -> None:
    for center in centers:
        center_data: list = []

        for datum in data:
            if datum.label == center:
                center_data.append(datum.instance)

        if len(center_data) != 0:
            center.instance = [sum(x) / len(x) for x in zip(*center_data)]


def make_gif() -> None:
    fp_in = "*.png"
    fp_out = "clusters.gif"

    with contextlib.ExitStack() as stack:
        imgs = (stack.enter_context(Image.open(f)) for f in sorted(glob.glob(fp_in)))
        img = next(imgs)
        img.save(
            fp=fp_out,
            format="GIF",
            append_images=imgs,
            save_all=True,
            duration=1000,
            loop=0,
        )


def setup():
    k: int = int(input("Choose number of blobs (e.g. 4): "))
    data_points: int = int(input("Choose number of data instances (e.g. 400): "))

    # Make k data blobs using random number generator
    # size = 10
    # data_raw = []

    # for i in range(k):
    #     point = [random.uniform(-size, size), random.uniform(-size, size)]
    #     for j in range(int(data_points / k)):
    #         data_raw.append(
    #             [
    #                 random.uniform(point[0] - size / 5, point[0] + size / 5),
    #                 random.uniform(point[1] - size / 5, point[1] + size / 5),
    #             ]
    #         )

    # X = np.array(data_raw)

    # Make k data blobs using sklearn make_blobs utility
    X, y_true = make_blobs(
        n_samples=data_points, centers=k, cluster_std=0.60, random_state=0
    )

    mean_data = np.mean(X, axis=0).tolist()
    std_dev = np.std(X)

    # Make k random centers within standard deviation range
    centers: list[Datum] = [
        Datum(
            [
                random.uniform(mean_data[0] - std_dev, mean_data[0] + std_dev),
                random.uniform(mean_data[1] - std_dev, mean_data[1] + std_dev),
            ]
        )
        for i in range(k)
    ]

    # Plot initial data and initial centers
    plt.scatter(X[:, 0], X[:, 1], s=10)
    plt.savefig("blobs.png")
    centers_init = np.array([center.instance for center in centers])
    plt.scatter(centers_init[:, 0], centers_init[:, 1], c="red", s=150, alpha=0.9)
    plt.savefig("centers.png")
    plt.clf()

    some_large_number: float = 100 * std_dev
    data = [Datum(datum) for datum in X.tolist()]
    num_of_iter: int = int(input("Choose number of iterations (e.g. 10): "))

    for i in range(num_of_iter):
        centers_plot = np.array([center.instance for center in centers])
        plt.scatter(centers_plot[:, 0], centers_plot[:, 1], c="red", s=150, alpha=0.9)
        label_data(data, centers, some_large_number)
        j: int = 0
        k: int = 0
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        clusters = []

        for center in centers:
            center_data: list = []

            for datum in data:
                if datum.label == center:
                    center_data.append(datum.instance)

            clusters.append(center_data)

        for cluster in clusters:
            if j < len(colors):
                color = colors[j]
            else:
                color = None
            if len(cluster) != 0:
                X = np.array(cluster)
                plt.scatter(X[:, 0], X[:, 1], c=color, s=10)
                j += 1

        plt.savefig(f"clusters-{i}.png")
        plt.clf()
        move_centers(data, centers)
        label_data(data, centers, some_large_number)
        centers_plot = np.array([center.instance for center in centers])
        plt.scatter(centers_plot[:, 0], centers_plot[:, 1], c="red", s=150, alpha=0.9)
        for cluster in clusters:
            if k < len(colors):
                color = colors[k]
            else:
                color = None
            if len(cluster) != 0:
                X = np.array(cluster)
                plt.scatter(X[:, 0], X[:, 1], c=color, s=10)
                k += 1
        plt.savefig(f"clusters-{i}{k}.png")
        plt.clf()

    print("\t Final centers:")

    for center in centers:
        print(round(center.instance[0], 2), round(center.instance[1], 2))


def main():
    setup()
    make_gif()
    # Clean individual PNG files
    os.system("rm *.png")
    os.system("open -a 'Google Chrome' clusters.gif")


if __name__ == "__main__":
    main()
