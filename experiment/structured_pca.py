"""Represent data as PCA components with best structure estimated by EXIMS

Arguments:
    path to datasets
    dataset filename part
    metadata filename part
    cache file path
    results root

"""
from functools import partial
import os
import pickle
import sys
from typing import List, NamedTuple

from functional import pipe
import numpy as np
import numpy.lib.format as fmt
from sklearn.externals import joblib
from spdata.types import Coordinates, Dataset
from tqdm import tqdm

from algorithm import exims_pca, ModelledDataset

Arguments = NamedTuple('Arguments', [
    ('datasets_root', str),
    ('data_suffix', str),
    ('metadata_suffix', str),
    ('cache_path', str),
    ('results_root', str)
])

only_directories = partial(filter, os.path.isdir)


def rooted_listdir(root):
    return [os.path.join(root, path) for path in os.listdir(root)]


subdirectories = pipe(rooted_listdir, only_directories, list)


class Memmap(Dataset):
    def __init__(self, spectra: np.memmap, coordinates: Coordinates, mz,
                 labels=None):
        self.spectra = spectra
        self.coordinates = coordinates
        self.mz = np.array(mz)
        self.labels = np.array(labels) if labels is not None else None
        self._validate()


def dummy_mz(spectra: np.memmap) -> np.ndarray:
    return np.arange(spectra.shape[1])


def load_dataset(root: str, data_suffix: str, metadata_suffix: str) -> Dataset:
    spectra = np.load(os.path.join(root, data_suffix), mmap_mode='r')
    metadata = np.loadtxt(os.path.join(root, metadata_suffix), skiprows=1,
                          delimiter=',', dtype=int)
    coordinates = Coordinates(x=metadata[:, 1], y=metadata[:, 2],
                              z=np.zeros_like(metadata[:, 1]))
    mz = dummy_mz(spectra)
    return Memmap(spectra, coordinates, mz)


def merge_datasets(datasets: List[Dataset], cache_path: str, shift: int=10) -> \
        Memmap:
    # Short intro on how to create empty memmap
    # https://stackoverflow.com/a/36749821/3067956
    n_spectra = sum([dataset.spectra.shape[0] for dataset in datasets])
    n_channels = datasets[0].spectra.shape[1]
    merged_spectra = fmt.open_memmap(cache_path, mode='w+', dtype=np.float32,
                                     shape=(n_spectra, n_channels))
    xs, ys, max_x, start = [], [], 0., 0
    for dataset in tqdm(datasets, desc='Merging'):
        x = dataset.coordinates.x - dataset.coordinates.x.min() + shift + max_x
        max_x = x.max()
        y = dataset.coordinates.y - dataset.coordinates.y.min()
        xs.append(x)
        ys.append(y)
        end = start + dataset.spectra.shape[0]
        merged_spectra[start:end] = dataset.spectra
        merged_spectra.flush()
    xs, ys = np.hstack(xs), np.hstack(ys)
    merged_coordinates = Coordinates(x=xs, y=ys, z=np.zeros_like(xs))
    readonly_spectra = np.load(cache_path, mmap_mode='r')
    return Memmap(readonly_spectra, merged_coordinates, datasets[0].mz)


def save(fname, obj):
    with open(fname, 'wb') as outfile:
        pickle.dump(obj, outfile)


def save_result(modelled: ModelledDataset, results_root: str):
    os.makedirs(results_root, exist_ok=True)
    fname = partial(os.path.join, results_root)
    save(fname('model.pkl'), modelled.model)
    np.save(fname('sorted_structness.npy'), modelled.features.score)
    np.save(fname('indices.npy'), modelled.features.index)
    np.save(fname('selection.npy'), modelled.features.selection)
    np.savetxt(fname('threshold.txt'), np.array([modelled.features.threshold]))
    np.save(fname('structured_dataset.npy'), modelled.structured_dataset)
    joblib.dump(modelled.transformed_dataset, fname('transformed_dataset.jbl'))
    joblib.dump(modelled, fname('full_result.jbl'))


if __name__ == '__main__':
    arguments = Arguments(*sys.argv[1:6])
    os.makedirs(arguments.results_root)
    datasets = [
        load_dataset(path, arguments.data_suffix, arguments.metadata_suffix)
        for path in subdirectories(arguments.datasets_root)
    ]
    merged_dataset = merge_datasets(datasets, arguments.cache_path)
    print(f'Merged {len(datasets)} datasets with total of '
          f'{merged_dataset.spectra.shape[0]} spectra.')
    number_of_components = merged_dataset.spectra.shape[1]
    print(f'Estimating {number_of_components} components...')
    modelled = exims_pca(merged_dataset, pca_components=number_of_components,
                         random_state=0)
    print(f'Selected {modelled.features.selection.sum()} features with'
          f' structness over {modelled.features.threshold}')
    print('Saving result')
    save_result(modelled, arguments.results_root)
    print(f'Removing cache from {arguments.cache_path}')
    del merged_dataset
    os.remove(arguments.cache_path)
