"""Force-directed layout algorithms"""
from __future__ import annotations

import typing
from collections.abc import Iterable

import numpy as np
import pandas as pd

T = typing.TypeVar("T")


def force_directed_layout(elements: Iterable[T], similarity_score: callable[[T, T], float], *,
                          n_iter: int = 500) -> pd.DataFrame:
    elements = list(elements)
    n_elems = len(elements)

    # The similarity map is a matrix that contains for each pair of elements (the index in the matrix corresponds to the
    # position in the input list) the similarity between these two points. Therefore, this matrix is symmetric with the main
    # diagonal being filled with zeros.
    similarity_map = np.zeros((n_elems, n_elems))

    # Initialize the similarity map
    for a_idx, a in enumerate(elements):
        for b_idx, b in enumerate(elements):
            if a_idx <= b_idx:
                continue
            current_score = similarity_score(a, b)
            similarity_map[a_idx, b_idx] = current_score
            similarity_map[b_idx, a_idx] = current_score

    # Potentially normalize the similarity map?
    normalized_similarity_map = similarity_map / np.max(similarity_map)

    # All elements of the input list will have 2D-coordinates assigned to them. These coordinates are stored in a matrix where
    # each row corresponds to the element with the same index in the input list. The x-axis corresponds to the first column
    # of the matrix and the y-value corresponds to the second column. We start with random coordinates and improve them over
    # time.
    element_coords = np.random.rand(n_elems, 2) * 10

    for __ in range(n_iter):
        distance_vectors = element_coords[:, np.newaxis, :] - element_coords[np.newaxis, :, :]  # dark numpy broadcasting magic
        euclidean_distances = np.linalg.norm(distance_vectors, axis=-1)
        normalized_distance_map = euclidean_distances / np.max(euclidean_distances)

        # tension = (similarity_map * np.max(euclidean_distances)) / (np.max(similarity_map) * euclidean_distances)
        shift_force = np.reshape(normalized_distance_map - normalized_similarity_map, (n_elems, n_elems, 1))
        shift_force = np.pad(shift_force, [(0, 0), (0, 0), (0, 1)], "edge") / 2

        shift_vectors = shift_force * distance_vectors
        shift_vectors = np.sum(shift_vectors, axis=1)

        element_coords += shift_vectors

    return pd.DataFrame({"element": elements, "x": element_coords[:, 0], "y": element_coords[:, 1]})
