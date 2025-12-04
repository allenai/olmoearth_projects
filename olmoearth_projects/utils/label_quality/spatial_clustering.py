"""How spatially clustered are my labels?"""

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from scipy.stats import mode


def relative_haversine(
    latlons_1: torch.Tensor, latlons_2: torch.Tensor
) -> torch.Tensor:
    """Calculate the great circle distance between two points on the earth.

    Latlons must be specified in radians.
    """
    dlon = latlons_2[:, 1] - latlons_1[:, 1]
    dlat = latlons_2[:, 0] - latlons_1[:, 0]

    a = (
        torch.sin(dlat / 2.0) ** 2
        + torch.cos(latlons_1[:, 0])
        * torch.cos(latlons_2[:, 0])
        * torch.sin(dlon / 2.0) ** 2
    )

    return torch.arcsin(torch.sqrt(a))


def spatial_clustering(df: gpd.GeoDataFrame, k: int = 5) -> float:
    """Spatial KNN.

    Given a dataset of labels with two columns (`label` and `geometry`),
    we run KNN using the geometry centroids as features. Highly clustered
    datasets will score highly.

    We assume the geometries are in WGS84 (latitude, longitude)
    """
    labels = df["label"].values
    # latitude , longitude = [y, x]
    features = torch.stack(
        [
            torch.from_numpy(np.radians(df.geometry.centroid.y.values)),
            torch.from_numpy(np.radians(df.geometry.centroid.x.values)),
        ],
        dim=-1,
    )
    # if labels are floats, then its a regression. If labels are ints or strings,
    # its classification
    regression = True
    if type(labels[0]) is str:
        regression = False
        labels, _ = pd.factorize(labels)
    elif (labels.astype(int) == labels).all():
        regression = False

    all_preds = []
    for i in range(features.shape[0]):
        test_feature = features[i].unsqueeze(dim=0).repeat(features.shape[0], 1)
        distances = relative_haversine(features, test_feature)
        # we skip the first index, which should be where test_feature == train_feature
        top_k_indices = (
            torch.topk(distances, k=k + 1, largest=False).indices[1:].numpy()
        )

        if not regression:
            all_preds.append(mode(labels[top_k_indices])[0])
        else:
            all_preds.append(labels[top_k_indices].mean())

    if regression:
        # MSE error
        print((labels - np.array(all_preds)) ** 2)
        return sum((labels - np.array(all_preds)) ** 2) / len(labels)
    else:
        # accuracy
        return sum(labels == np.array(all_preds)) / len(labels)
