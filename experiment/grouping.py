from functools import partial
from multiprocessing import Pool
import os
import pickle
import sys
from typing import NamedTuple

import numpy as np
import spdivik.distance as dst
import spdivik.kmeans as km
import spdivik.score as sc
import spdivik.types as ty

Arguments = NamedTuple('Arguments', [
    ('pca_model_root', str),
    ('dataset_path', str),
    ('results_root', str)
])


def load_data(arguments: Arguments) -> ty.Data:
    dataset = np.load(arguments.dataset_path)
    model_path = os.path.join(arguments.pca_model_root, 'model.pkl')
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    transformed = model.transform(dataset)
    selection = np.load(os.path.join(arguments.pca_model_root, 'selection.npy'))
    structured = transformed[:, selection]
    return structured


def dunn_optimized_kmeans(pool: Pool) -> sc.Optimizer:
    distance = dst.ScipyDistance(dst.KnownMetric.euclidean)
    dunn = partial(sc.dunn, distance=distance)
    kmeans = km.KMeans(labeling=km.Labeling(distance),
                       initialize=km.PercentileInitialization(distance))
    optimized = sc.Optimizer(score=dunn,
                             segmentation_method=kmeans,
                             parameters=[
                                 sc.ParameterValues('number_of_clusters',
                                                    list(range(2, 11)))],
                             pool=pool)
    return optimized


def save(labels: ty.IntLabels, centroids: ty.Centroids, quality: float,
         arguments: Arguments):
    os.makedirs(arguments.results_root)
    fname = partial(os.path.join, arguments.results_root)
    np.savetxt(fname('labels.txt'), labels, fmt='%i')
    np.savetxt(fname('quality.txt'), np.array([quality]))
    np.save(fname('centroids.npy'), centroids)


def main():
    arguments = Arguments(*sys.argv[1:])
    data = load_data(arguments)
    with Pool() as pool:
        kmeans = dunn_optimized_kmeans(pool)
        labels, centroids, quality = kmeans(data)
    save(labels, centroids, quality, arguments)


if __name__ == '__main__':
    main()
