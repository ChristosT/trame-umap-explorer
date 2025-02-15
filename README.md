# Introduction
This is a standalone [trame](https://kitware.github.io/trame) prototype application for statistical  analysis of multi-channel volumetric datasets.
The original application is at [Kitware/multivariate-view](https://github.com/Kitware/multivariate-view).


## Instructions
Make sure you have access to the original file and the rba data saved using the multivariate-view application.

For sarcoma data run:

```
python trame_umap_explore_plotly.py --file thigh_sarcoma.h5 --normalize-separately 1 --rgba rgba_dataset_thigh_sarcoma.npy
```

For haadf_removed dataset  run:

```
python trame_umap_explore_plotly.py --file haadf_removed.h5 --rgba rgba_dataset_haadf_removed.npy
```

## Usage
1. Select dimensionality reduction method (umap,pca,t-sne) and click GO
2. Perform selection on scatter plot to see changes reflected on parallel coords and 3D data
3. You can also select ranges on parallel coords to select ranges in 3D data
4. For clustering pick a clustering method and modify its parameters. The clustering is applied on the fly
5. To incorporate spatial position of voxels enable the "Use ijk" switch and click Go.

# Example

Applying t-distributed stochastic neighbor embedding (t-SNE) to MRI data of a knee with a sarcoma.
Selecting one of clusters regions reveals the sarcoma.

![screenshot](https://github.com/user-attachments/assets/d028c350-7164-433f-b0ed-b9575b0a35d3)
