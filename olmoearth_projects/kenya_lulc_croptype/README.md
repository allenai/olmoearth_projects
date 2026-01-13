# Kenya cropland and crop type mapping

The purpose of this project is to map cropland and maize in Kenya. We will focus on two counties: Taita-Taveta county and Machackos county.

We are specifically interested in the short rains (August - December) in 2024 and 2025.

For our first go, we will:

1. Collect labels online, mostly (entirely?) leaning on WorldCereals' Reference Data Module.


### 2026-01-12

Collect data from GeoGLAM and Harvest. Checking the label quality yields:

```
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃         Check name ┃ Metric                         ┃                  Value ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│        # instances │                                │                   5349 │
│    label_imbalance │ non_cropland_incl_perennial    │     0.3654888764255001 │
│    label_imbalance │ vegetables_fruits              │   0.017386427369601793 │
│    label_imbalance │ maize                          │     0.1518040755281361 │
│    label_imbalance │ dry_pulses_legumes             │   0.022434099831744252 │
│    label_imbalance │ other_permanent_crops          │   0.052346232940736585 │
│    label_imbalance │ fruits                         │   0.007478033277248084 │
│    label_imbalance │ wheat                          │   0.005608524957936063 │
│    label_imbalance │ grass_fodder_crops             │    0.06019816788184707 │
│    label_imbalance │ potatoes                       │    0.02075154234436343 │
│    label_imbalance │ other_oilseeds                 │   0.008786689100766498 │
│    label_imbalance │ permanent_crops                │  0.0001869508319312021 │
│    label_imbalance │ herb_spice_medicinal_crops     │   0.002243409983174425 │
│    label_imbalance │ root_tuber_crops               │   0.002243409983174425 │
│    label_imbalance │ herbaceous_vegetation          │     0.1417087306038512 │
│    label_imbalance │ shrubland                      │    0.10357076088988595 │
│    label_imbalance │ built_up                       │   0.017199476537670594 │
│    label_imbalance │ trees_mixed                    │    0.02056459151243223 │
│ spatial_clustering │ non_cropland_incl_perennial_f1 │     0.9693476552266177 │
│ spatial_clustering │ vegetables_fruits_f1           │    0.18446601941747576 │
│ spatial_clustering │ maize_f1                       │     0.4715274081958489 │
│ spatial_clustering │ dry_pulses_legumes_f1          │     0.2577777777777778 │
│ spatial_clustering │ other_permanent_crops_f1       │     0.6941896024464831 │
│ spatial_clustering │ fruits_f1                      │                    0.0 │
│ spatial_clustering │ wheat_f1                       │    0.27586206896551724 │
│ spatial_clustering │ grass_fodder_crops_f1          │    0.45523520485584223 │
│ spatial_clustering │ potatoes_f1                    │    0.21052631578947367 │
│ spatial_clustering │ other_oilseeds_f1              │    0.14084507042253522 │
│ spatial_clustering │ permanent_crops_f1             │                      0 │
│ spatial_clustering │ herb_spice_medicinal_crops_f1  │                   0.25 │
│ spatial_clustering │ root_tuber_crops_f1            │     0.7407407407407407 │
│ spatial_clustering │ herbaceous_vegetation_f1       │     0.3896961690885073 │
│ spatial_clustering │ shrubland_f1                   │     0.5123809523809524 │
│ spatial_clustering │ built_up_f1                    │    0.16296296296296298 │
│ spatial_clustering │ trees_mixed_f1                 │    0.11594202898550725 │
│     spatial_extent │ non_cropland_incl_perennial    │                    1.0 │
│     spatial_extent │ vegetables_fruits              │    0.14221053915323867 │
│     spatial_extent │ maize                          │     0.1506813572429614 │
│     spatial_extent │ dry_pulses_legumes             │    0.14340952692210965 │
│     spatial_extent │ other_permanent_crops          │   0.060810783275016114 │
│     spatial_extent │ fruits                         │    0.12496287721217952 │
│     spatial_extent │ wheat                          │    0.04621464822555264 │
│     spatial_extent │ grass_fodder_crops             │    0.13625229716420412 │
│     spatial_extent │ potatoes                       │    0.09626723863137074 │
│     spatial_extent │ other_oilseeds                 │    0.09950036391116568 │
│     spatial_extent │ permanent_crops                │                    0.0 │
│     spatial_extent │ herb_spice_medicinal_crops     │ 0.00022078429120778403 │
│     spatial_extent │ root_tuber_crops               │  6.954294582642153e-08 │
│     spatial_extent │ herbaceous_vegetation          │     0.1501992889525249 │
│     spatial_extent │ shrubland                      │    0.14333979010623907 │
│     spatial_extent │ built_up                       │     0.1060962971921383 │
│     spatial_extent │ trees_mixed                    │    0.09445086446004335 │
└────────────────────┴────────────────────────────────┴────────────────────────┘
```
Todo:
- aggregate into cropland and croptype labels
- materialize, etc.
