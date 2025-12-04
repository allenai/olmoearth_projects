# Label Quality

`olmoearth_projects` demonstrates how OlmoEarth can be applied to downstream applications.
Specifically, given a set of labels, `olmoearth_projects` demonstrates how to finetune, evaluate and apply OlmoEarth over a spatial area.

The quality of the model's predictions depend on the quality of the labels.
Assessing the quality of the labels is best done by domain experts.
However, the functions in this folder also provide some indication of how well suited a set of labels are for mapping.

#### Spatial Clustering

This function assesses how spatially clustered classes are.
In general, we'd like different classes to be well spatially distributed:

```
xoxoxox
oxoxoxo
xoxoxox
```
is more desirable than
```
xxx
xxx
   ooo
   ooo
```
We measure this by running a spatial KNN on the dataset - for each instance in the dataset, we define its class
to be the mode of the K nearest (spatial) points. High accuracies indicate high spatial clustering.

```python
import geopandas as gpd
import pandas as pd

from olmoearth_projects.utils.label_quality.spatial_clustering import spatial_clustering

df = pd.DataFrame(
    {
        "City": ["Buenos Aires", "Brasilia", "Santiago", "Bogota", "Caracas"],
        "Country": ["Argentina", "Brazil", "Chile", "Colombia", "Venezuela"],
        # highly clustered labels
        "label": [0, 0, 0, 1, 1],
        "Latitude": [-34.58, -15.78, -33.45, 4.60, 10.48],
        "Longitude": [-58.66, -47.91, -70.66, -74.08, -66.86],
    }
)
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326"
)
assert spatial_clustering(gdf[["label", "geometry"]], k=1) == 1
```

### Spatial extent

This function assesses how much of the total labelled area each class occupies.
In general, we would like each class to occupy a large fraction of the total labelled area:

```
x xox x
ox x xo
x xox x
```
is more desirable then
```
x x x x
 x xoxo
x xoxox
```
