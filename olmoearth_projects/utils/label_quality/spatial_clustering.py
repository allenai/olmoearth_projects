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


def spatial_clustering(df: gpd.GeoDataFrame, k: int = 5) -> dict[str | int, float]:
    """Spatial KNN.

    Given a dataset of labels with two columns (`label` and `geometry`),
    we run KNN using the geometry centroids as features. Highly clustered
    datasets will score highly (or low, if its a regression problem and
    we are measuring MSE).

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
    if (type(labels[0]) is str) or (labels.astype(int) == labels).all():
        regression = False
        labels, unique = pd.factorize(labels)

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
    all_preds_np = np.array(all_preds)
    if regression:
        # MSE error

        return {"regression_mse": sum((labels - all_preds_np) ** 2) / len(labels)}
    else:
        output_dict: dict[str | int, float] = {}
        # f1 score
        for label_idx, label_value in enumerate(unique):
            cat_labels = labels == label_idx
            cat_preds = all_preds_np == label_idx
            if sum(cat_preds) == 0:
                # no instances received this value as a prediction
                output_dict[f"{label_value}_f1"] = 0
            else:
                positives = cat_labels == 1
                tp = sum(cat_labels[positives] == cat_preds[positives])
                recall = tp / sum(cat_labels)
                precision = tp / sum(cat_preds)
                output_dict[f"{label_value}_f1"] = 2 / ((1 / recall) + (1 / precision))
        return output_dict
