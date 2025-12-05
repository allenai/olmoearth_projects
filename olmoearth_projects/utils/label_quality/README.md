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
For each class, this is measured as `(area covered by all the labels in the class) / (area covered by all the labels)`.

### Label imbalance

This function assesses the fraction of labels belonging to each class: `(number of labels in a class) / (total number of labels)`.

### Examples

An example of how to run this is on an rslearn dataset is in [the `mozambique_lulc` project](../../projects/mozambique_lulc/check_label_quality.py):

```console
$ python olmoearth_projects/projects/mozambique_lulc/check_label_quality.py --ds_path /weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc/20251202 --split train

Checking label quality for 4881 instances.
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃         Check name ┃ Metric                ┃               Value ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│    label_imbalance │ Bare Ground           │ 0.12681827494365908 │
│    label_imbalance │ Trees                 │ 0.09813562794509322 │
│    label_imbalance │ Cropland              │ 0.27760704773611966 │
│    label_imbalance │ Flooded Vegetation    │  0.1024380249948781 │
│    label_imbalance │ Water                 │ 0.11391108379430445 │
│    label_imbalance │ Rangeland             │ 0.10530628969473468 │
│    label_imbalance │ Buildings             │  0.1757836508912108 │
│ spatial_clustering │ Bare Ground_f1        │  0.7763055339049103 │
│ spatial_clustering │ Trees_f1              │               0.918 │
│ spatial_clustering │ Cropland_f1           │  0.8201489890031926 │
│ spatial_clustering │ Flooded Vegetation_f1 │  0.6470588235294118 │
│ spatial_clustering │ Water_f1              │  0.5609756097560976 │
│ spatial_clustering │ Rangeland_f1          │  0.7097480832420592 │
│ spatial_clustering │ Buildings_f1          │  0.9638554216867469 │
│     spatial_extent │ Bare Ground           │   0.906388431021162 │
│     spatial_extent │ Trees                 │  0.8143211426450099 │
│     spatial_extent │ Cropland              │  0.8178565572914295 │
│     spatial_extent │ Flooded Vegetation    │  0.8195186876112993 │
│     spatial_extent │ Water                 │  0.8015534585021155 │
│     spatial_extent │ Rangeland             │  0.9892764881988351 │
│     spatial_extent │ Buildings             │  0.7256137393021044 │
└────────────────────┴───────────────────────┴─────────────────────┘
```
