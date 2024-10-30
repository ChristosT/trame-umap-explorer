r"""
Installation requirements:
    pip install trame trame-vuetify trame-plotly trame-components sklearn umap-learn
"""

import plotly.graph_objects as go
import plotly.io as pio

from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3, plotly, html, vtk
import numpy as np
import re
import umap
import h5py
import numba
import os.path
import sklearn.cluster as cluster
from sklearn import decomposition
from sklearn.manifold import TSNE

# from openTSNE import TSNE
from volume_view import VolumeView
import argparse
from timeit import default_timer as timer


DEFAULTS = {
    "dimensionality_reduction_method": "umap",
    "min_dist": 0.0,
    "n_neighbors": 27,
    "sample_size": 2500,
    "spread": 21,
    "repulsion_strength": 1,
    "color_by": None,
    "clustering_method": None,
    "n_clusters": 1,
    "marker_size": 6,
    "min_samples": 2,
    "min_cluster_size": 5,
    "max_eps": 100,
    "metric": "euclidean",
    "perplexity": 30.0,
    "max_iter": 3000,
    "cluster_opacity": 0.2,
}
FILENAME = "CeCoFeGd_doi_10.1038_s43246-022-00259-x.h5"
LABELMAP_FILENAME = None  # "miec_rough_label_map.npy"
RGBA_FILENAME = None  # "rgba_dataset.npy"
#  the entire dataset in (points,nchannels format)
DATA = None
MANUAL_LABEL = None
DATA_MASK = None
RGB_DATA_SHAPE = None
CLUSTERS = None
NORMALIZE_SEPARATELY = False

# keep track of user provided constains in parallel coordinates
CURRENT_CONSTRAINS = dict()

# set a default style so that the colormap is fixed and can be reused for 3d view"
pio.templates.default = "plotly"
import plotly.colors as pcolors

COLORSCALE = "jet"
COLORS = pcolors.qualitative.__dict__["Plotly"]


def hex_to_float_rgb(hex_color):
    # Remove the '#' if it's there
    hex_color = hex_color.lstrip("#")
    # Convert to RGB
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def hex_to_int_rgb(hex_color):
    # Remove the '#' if it's there
    hex_color = hex_color.lstrip("#")
    # Convert to RGB
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


COLORS_RGB = [hex_to_float_rgb(x) for x in COLORS]
COLORS_INT_RGB = [hex_to_int_rgb(x) for x in COLORS]


def parse_arguments():
    global FILENAME, LABELMAP_FILENAME, RGBA_FILENAME, NORMALIZE_SEPARATELY
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label", help="labelmap used for clustering", default=LABELMAP_FILENAME
    )
    parser.add_argument("--file", help="file to read", default=FILENAME)
    parser.add_argument("--rgba", help="rgba volume for 3D view", default=RGBA_FILENAME)
    parser.add_argument(
        "--normalize-separately",
        help="normalize each channel separately",
        default=NORMALIZE_SEPARATELY,
    )

    args, _ = parser.parse_known_args()  # allow for --server arg
    if args.label:
        LABELMAP_FILENAME = args.label
    if args.file:
        FILENAME = args.file
    if args.rgba:
        RGBA_FILENAME = args.rgba

    if args.normalize_separately:
        print("NORMALIZE_SEPARATELY")
        NORMALIZE_SEPARATELY = args.normalize_separately


def load_hdf5_dataset(path):
    labels = []
    data = []

    with h5py.File(path, "r") as f:
        for key in f:
            labels.append(key)
            data.append(f[key][()])

    data = np.stack(data, axis=3)
    return labels, data


@numba.njit(cache=True, nogil=True)
def _calculate_padding_uniform(data: np.ndarray) -> np.ndarray:
    num_channels = data.shape[-1]
    zero_data = np.isclose(data, 0).sum(axis=3) == num_channels

    # This is the number to crop
    n = 0
    indices = np.array([n, -n - 1])
    while (
        zero_data[indices].all()
        & zero_data[:, indices].all()
        & zero_data[:, :, indices].all()
    ):
        n += 1
        indices = np.array([n, -n - 1])

    return n


def _remove_padding_uniform(data: np.ndarray, n) -> np.ndarray:
    if n != 0:
        data = data[n : -n - 1, n : -n - 1, n : -n - 1]
    return data


@numba.njit(cache=True, nogil=True)
def _normalize_data(data: np.ndarray, new_min: float = 0, new_max: float = 1):
    max_val = data.max()
    min_val = data.min()

    return (new_max - new_min) * (data.astype(np.float64) - min_val) / (
        max_val - min_val
    ) + new_min


def preprocess(filename, labelfile, drop_ones=False):
    global LABELS, DATA, NUMBER_OF_CHANNELS, MANUAL_LABEL, ORIGINAL_IDS_INDEX, CURRENT_CONSTRAINS, DATA_MASK
    LABELS, original_data = load_hdf5_dataset(filename)
    original_shape = original_data.shape
    print(f"Reading {filename}. shape {original_data.shape}")
    n_to_crop = _calculate_padding_uniform(original_data)
    original_data = _remove_padding_uniform(original_data, n_to_crop)
    data_shape = original_data.shape[:-1]
    num_channels = original_data.shape[-1]
    NUMBER_OF_CHANNELS = num_channels
    if NORMALIZE_SEPARATELY:
        # Normalize each channel to be between 0 and 1
        for i in range(original_data.shape[-1]):
            original_data[:, :, :, i] = _normalize_data(original_data[:, :, :, i])
    else:
        original_data = _normalize_data(original_data)

    for i in range(original_data.shape[-1]):
        to_remove = original_data[:, :, :, i] > 0.9
        original_data[to_remove] = 0

    # flatten
    raw_unpadded_flattened_data = original_data.reshape(
        np.prod(data_shape), num_channels
    )
    data = raw_unpadded_flattened_data.copy()

    # add ijk indices
    I = np.indices(data_shape).reshape(3, -1).T
    data = np.concatenate((data, I), axis=1)
    data[:, NUMBER_OF_CHANNELS] /= data_shape[0]
    data[:, NUMBER_OF_CHANNELS + 1] /= data_shape[1]
    data[:, NUMBER_OF_CHANNELS + 2] /= data_shape[2]

    # add flat index
    I = np.indices([data.shape[0]]).T
    data = np.concatenate((data, I), axis=1)
    ORIGINAL_IDS_INDEX = data.shape[1] - 1

    # get a view without the indices
    data_view = data[:, :num_channels]
    nonzero_indices = ~np.all(np.isclose(data_view, 0), axis=1)

    # drop ones
    mask = nonzero_indices
    if drop_ones:
        nonone_indices = ~np.any(np.isclose(data, 1), axis=1)
        mask = mask & nonone_indices

    DATA = data[mask]
    DATA_MASK = mask

    if labelfile is not None and os.path.isfile(labelfile):
        MANUAL_LABEL = np.load(labelfile)
        print(f"Reading {labelfile}. Shape: {MANUAL_LABEL.shape}")
        if MANUAL_LABEL.shape == original_shape[:-1]:
            if n_to_crop != 0:
                MANUAL_LABEL = _remove_padding_uniform(MANUAL_LABEL, n_to_crop)
        MANUAL_LABEL = MANUAL_LABEL.reshape(np.prod(MANUAL_LABEL.shape))
        MANUAL_LABEL = MANUAL_LABEL[mask]

    for i in range(len(LABELS)):
        CURRENT_CONSTRAINS[i] = []


def preprocess_rba(filename):
    if filename is None:
        return
    global RGB_DATA_SHAPE
    original_data = np.load(filename)
    data_shape = original_data.shape[:-1]
    num_channels = original_data.shape[-1]
    RGB_DATA_SHAPE = data_shape
    print(original_data[50, 50, 50])
    # flatten
    flattened_data = original_data.reshape(np.prod(data_shape), num_channels)
    nonzero_indices = ~np.all(np.isclose(flattened_data, 0), axis=1)
    print("VOLUME_DATA", original_data.shape)

    VOLUME_VIEW.set_data(original_data)
    mask_ref = VOLUME_VIEW.mask_reference
    mask_ref[nonzero_indices] = 1
    VOLUME_VIEW.mask_data.Modified()
    VOLUME_VIEW.volume_property.SetShade(1)



U = None
SCATTER_SELECTION = dict()
VOLUME_VIEW = VolumeView()
SAMPLE = None
MASK_SAMPLE = None
EMBEDDING_MAP = None


def sample_data(data, sample_size):
    # Randomly select unique row indices
    global SAMPLE, MASK_SAMPLE
    random_indices = np.random.choice(data.shape[0], sample_size, replace=False)
    SAMPLE = data[random_indices]
    if MANUAL_LABEL is not None:
        MASK_SAMPLE = MANUAL_LABEL[random_indices]


def umap_fit(
    points,
    min_dist,
    n_neighbors,
    spread,
    repulsion_strength,
    dimension,
    metric,
    use_coords,
):
    global EMBEDDING_MAP
    f = umap.UMAP(
        min_dist=min_dist,
        repulsion_strength=repulsion_strength,
        spread=spread,
        n_neighbors=n_neighbors,
        n_components=dimension,
        metric=metric,
    )
    if use_coords:
        dataset = points[:, : NUMBER_OF_CHANNELS + 3]
    else:
        dataset = points[:, :NUMBER_OF_CHANNELS]
    fit = f.fit(dataset)
    EMBEDDING_MAP = fit
    u = fit.transform(dataset)
    print("Done")
    return u


def pca_fit(points, dimension, use_coords):
    global EMBEDDING_MAP
    f = decomposition.PCA(n_components=dimension)
    if use_coords:
        dataset = points[:, : NUMBER_OF_CHANNELS + 3]
    else:
        dataset = points[:, :NUMBER_OF_CHANNELS]
    fit = f.fit(dataset)
    EMBEDDING_MAP = fit
    u = fit.transform(dataset)
    print("Done")
    return u


def tsne_fit(points, perplexity, max_iter, dimension, use_coords):
    global EMBEDDING_MAP
    f = TSNE(n_components=dimension, perplexity=perplexity, max_iter=max_iter, n_jobs=-1)
    if use_coords:
        dataset = points[:, : NUMBER_OF_CHANNELS + 3]
    else:
        dataset = points[:, :NUMBER_OF_CHANNELS]
    u = f.fit_transform(dataset)
    # TODO this requires TSNE from a openTSNE package which is slower with default parameters
    # u  = f.fit(dataset)
    # EMBEDDING_MAP = fit
    # u = fit.transform(dataset)
    print("Done")
    return u


server = get_server(client_type="vue3")
state = server.state
state.trame__title = "umap explorer"
state.cluster_info = None


def fit_data(points):
    if state.dimensionality_reduction_method == "umap":
        return umap_fit(
            points,
            state.min_dist,
            state.n_neighbors,
            state.spread,
            state.repulsion_strength,
            state.dimension,
            state.metric,
            state.use_coords,
        )
    elif state.dimensionality_reduction_method == "pca":
        return pca_fit(points, state.dimension, state.use_coords)
    elif state.dimensionality_reduction_method == "tsne":
        return tsne_fit(
            points, state.perplexity, state.max_iter, state.dimension, state.use_coords
        )


@state.change("color_by")
def on_color_by(color_by, **kwargs):
    if U is not None:
        server.controller.figure_scatter_update(scatter(U, COLOR_ARGS, state.dimension))


@state.change("sample_size")
def on_sample_size(sample_size, **kwargs):
    sample_data(DATA, sample_size)


@state.change(
    "clustering_method",
    "n_clusters",
    "min_cluster_size",
    "min_samples",
    "max_eps",
)
def on_clustering_method(**kwargs):
    global COLOR_ARGS
    if U is not None:
        COLOR_ARGS = get_color_args(
            color_by=state.color_by, clustering_method=state.clustering_method
        )
        server.controller.figure_scatter_update(scatter(U, COLOR_ARGS, state.dimension))
        server.controller.figure_parallel_coords_update(
            parallel_coords(color_args=COLOR_ARGS)
        )
        if state.clustering_method == "manual":
            update_volume_view(labels=MANUAL_LABEL)


@state.change("marker_size")
def on_marker_size(marker_size, **kwargs):
    global COLOR_ARGS
    if U is not None:
        COLOR_ARGS["marker"]["size"] = marker_size
        server.controller.figure_scatter_update(scatter(U, COLOR_ARGS, state.dimension))


def get_color_args(color_by=None, clustering_method=None):
    global CLUSTERS
    # TODO decouple color attributes with clustering
    size = state.marker_size
    if clustering_method is None:
        if color_by is None:
            color_args = {
                "marker_color": "blue",
                "marker": {"size": size, "opacity": 1.0},
            }
        else:
            if color_by < 10:
                color_args = {
                    "marker_color": SAMPLE[:, color_by],
                    "marker": {"colorscale": "blues", "size": size},
                }
            else:
                color0 = color_by // 10
                color1 = color_by % 10
                color_args = {
                    "marker_color": SAMPLE[:, color0] + SAMPLE[:, color1],
                    "marker": {"colorscale": "blues", "size": size},
                }
    else:
        if clustering_method == "kmeans":
            clustering_map = cluster.KMeans(n_clusters=state.n_clusters).fit(U)
            labels = clustering_map.predict(U)
            color_args = {
                "marker_color": labels,
                "marker": {"size": size, "opacity": 1.0,
                "colorscale": COLORSCALE,
                },
            }
            CLUSTERS = labels
        elif clustering_method == "hdbscan":
            clustering_map = cluster.HDBSCAN(
                min_cluster_size=state.min_cluster_size,
                min_samples=state.min_samples,
            )
            labels = clustering_map.fit_predict(U)
            labels += 1  # labels in hdbscan start from -1
            color_args = {
                "marker_color": labels,
                "marker": {"size": size, "opacity": 1.0,
                "colorscale": COLORSCALE,
                },
            }
            CLUSTERS = labels
        elif clustering_method == "optics":
            labels = cluster.OPTICS(
                min_samples=state.min_samples, max_eps=state.max_eps
            ).fit_predict(U)
            color_args = {
                "marker_color": labels,
                "marker": {"size": size, "opacity": 1.0,
                "colorscale": COLORSCALE,
                },
            }
            labels += 1  # labels in optics start from -1
            CLUSTERS = labels
        elif clustering_method == "manual":
            color_args = {"marker_color": list(MASK_SAMPLE), "marker": {"size": size,
                "colorscale": COLORSCALE,
            }}
            CLUSTERS = MASK_SAMPLE
        else:
            pass
        uniq_labels = np.unique(CLUSTERS)
        fields = dict()
        for l in uniq_labels:
            fields[str(l)] = {"opacity": DEFAULTS["cluster_opacity"]}
        state.cluster_info = fields
        state.dirty("cluster_info")
    return color_args


import matplotlib


def mask_points_with_polygon(points, x, y):
    polygon_points = list(zip(x, y))
    polygon = matplotlib.path.Path(polygon_points)
    is_inside = polygon.contains_points(points)
    return is_inside


def on_scatter_selected_event(packet):
    # print(packet)
    selection_data = packet["selected"]
    lasso_points = packet["lassoPoints"]
    box_range = packet["range"]
    if len(selection_data) == 0:
        return

    # update the parallel coordinates simple with the min/max for each channel
    scatter_selection = dict()
    for i in range(NUMBER_OF_CHANNELS):
        scatter_selection[i] = [
            [
                np.min([item["metadata"][i] for item in selection_data]),
                np.max([item["metadata"][i] for item in selection_data]),
            ]
        ]
    server.controller.figure_parallel_coords_update(
        parallel_coords(scatter_selection, color_args=COLOR_ARGS)
    )

    # using the selected IDS results in a too sparse volume view
    # ids = [item["metadata"][-1] for item in selection_data]
    # update_volume_view(mask_ids=ids)

    # in volume view we would like to use inference on the dimensionality
    # reduction method and evaluate mapped X,Y for each voxel but this takes
    # too much time.

    # As a remedy we use the range derived above
    ids = filter_on_constrains(scatter_selection)
    update_volume_view(ids)

    ## this is too slow

    # start = timer()
    # print("inference start")
    # we can also perfom inferece and perform on those values only.
    # new_points = EMBEDDING_MAP.transform(DATA[ids,:NUMBER_OF_CHANNELS])
    # end = timer()
    # print("inference done",end-start)
    # if lasso_points:
    #  x = lasso_points['x']
    #  y = lasso_points['y']
    # else:
    #  x = box_range['x']
    #  y = box_range['y']
    # print("filter points")
    # start = timer()
    # local_mask = mask_points_with_polygon(new_points,x,y)
    # gids =  new_points[local_mask,ORIGINAL_IDS_INDEX].astype(np.int32)
    # end = timer()
    # print("filter points done",end-start)
    # print(gids)
    # update_volume_view(gids)


def filter_on_constrains(constrains):
    """Filter DATA given a list of channel constrains in the form of a list of non
    overlapping lists.  e.g {0: [[1,3]], 2:[[ 0,1], [2,3]]} returns the (flat)
    ids inside DATA where channel 0 value is between 1,3 AND channel 2 value is
    in [0,1] OR [2,3]
    """
    condition = np.ones(DATA.shape[0], dtype=bool)
    for channel, channel_constrain_list in constrains.items():
        if channel_constrain_list:
            channel_constrain = False
            for constrain in channel_constrain_list:
                channel_constrain |= (DATA[:, channel] > constrain[0]) & (
                    DATA[:, channel] < constrain[1]
                )
            condition &= channel_constrain

    ids = DATA[condition, ORIGINAL_IDS_INDEX].astype(np.int32)
    return ids


def on_parallel_coords_select_event(selection_data):
    global CURRENT_CONSTRAINS
    # example event {'dimensions[0].constraintrange': [[0.3651979139717597, 0.48502199044130667]]}
    key = list(selection_data.keys())[0]
    channel = int(re.search(r"\d+", key).group())
    constrained_ranges = selection_data[key]
    # print(f"{channel=}")
    # print(f"{constrained_ranges=}")

    # reset constrain for this channel
    if constrained_ranges is None:
        CURRENT_CONSTRAINS[channel] = []
    else:
        # if it is one constrain we get [[a,b]] but if they are more [[[a,b],[c,d]]]
        if type(constrained_ranges[0][0]) is float:
            CURRENT_CONSTRAINS[channel] = constrained_ranges
        else:
            CURRENT_CONSTRAINS[channel] = constrained_ranges[0]

    ids = filter_on_constrains(CURRENT_CONSTRAINS)
    update_volume_view(ids)


def scatter(U, color_args=None, dimension=2):
    customdata = SAMPLE
    hovertemplate = "%{customdata[0]:.3f} <br> %{customdata[1]:.3f} <br> %{customdata[2]:.3f} <br> %{customdata[3]:.3f} "
    if state.clustering_method == "manual":
        customdata = np.column_stack((SAMPLE, MASK_SAMPLE))
        hovertemplate += "<br> %{customdata[7]}"
    if dimension == 2:
        plot = go.Scatter(
            x=U[:, 0],
            y=U[:, 1],
            mode="markers",
            **color_args,
            customdata=customdata,
            hovertemplate=hovertemplate,
            name="",
        )
    else:
        plot = go.Scatter3d(
            x=U[:, 0],
            y=U[:, 1],
            z=U[:, 2],
            mode="markers",
            **color_args,
            customdata=SAMPLE,
            hovertemplate="%{customdata[0]:.3f} <br> %{customdata[1]:.3f} <br> %{customdata[2]:.3f} <br> %{customdata[3]:.3f} ",
            name="",
        )
    return go.Figure(data=plot).update_layout(margin=dict(l=10, r=10, t=25, b=10))


def parallel_coords(constraintranges=None, color_args=None):
    dimensions = []
    # print(constraintranges)
    for i in range(NUMBER_OF_CHANNELS):
        dim = dict(
            range=[0, 1],
            label=LABELS[i],
            values=SAMPLE[:, i],
        )
        if constraintranges is not None:  # and i in constraintranges.keys():
            dim["constraintrange"] = constraintranges[i]

        dimensions.append(dim)

    return go.Figure(
        data=go.Parcoords(
            dimensions=dimensions,
            line=dict(
                color=color_args["marker_color"],
                colorscale= COLORSCALE,
            ),
        )
    )


def update_opacity():
    # print(state.cluster_info)
    if state.cluster_info is not None:
        update_volume_view(labels=CLUSTERS)
        # scatter plot
        opacities = [state.cluster_info[str(l)]["opacity"] for l in CLUSTERS]
        COLOR_ARGS["marker_color"] = [
            f"rgba{*COLORS_INT_RGB[label], opacities[i]}"
            for i, label in enumerate(CLUSTERS)
        ]
        server.controller.figure_scatter_update(scatter(U, COLOR_ARGS, state.dimension))
        # parallel coords do not support opacity at the moment
        # see https://github.com/plotly/plotly.js/issues/3964#issuecomment-502110263
        # server.controller.figure_parallel_coords_update(parallel_coords(color_args=COLOR_ARGS))


def update_volume_view(mask_ids=None, labels=None):
    if mask_ids is not None:
        mask_ref = VOLUME_VIEW.mask_reference
        mask_ref[:] = False
        mask_ref[mask_ids] = True
        VOLUME_VIEW.mask_data.Modified()
        server.controller.view_update()
    elif labels is not None:
        print("apply manual")
        print(state.cluster_info)
        mask_ref = VOLUME_VIEW.mask_reference
        new_data = np.zeros((*mask_ref.shape, 4), dtype=np.float64)

        # we have a full data mask
        if np.prod(mask_ref[DATA_MASK].shape) == labels.shape[0]:
            rgba = [
                (*COLORS_RGB[label], state.cluster_info[str(label)]["opacity"])
                for label in labels
            ]
            print(labels.shape)
            print(rgba[0])
            print(new_data[DATA_MASK].shape)
            new_data[DATA_MASK] = np.array(rgba)  # .flatten()
            new_data = new_data.reshape((*RGB_DATA_SHAPE, 4))
            print(new_data.shape)
            VOLUME_VIEW.set_data(new_data)
            mask_ref = VOLUME_VIEW.mask_reference
            mask_ref[DATA_MASK] = 1
            VOLUME_VIEW.mask_data.Modified()
            server.controller.view_update()
            print("updated")
        # we need to predict the full mask
        else:
            pass


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
def update_plot():
    global U, COLOR_ARGS
    U = fit_data(SAMPLE)
    COLOR_ARGS = get_color_args(
        color_by=state.color_by, clustering_method=state.clustering_method
    )
    server.controller.figure_parallel_coords_update(
        parallel_coords(color_args=COLOR_ARGS)
    )
    server.controller.figure_scatter_update(
        scatter(
            U,
            COLOR_ARGS,
            dimension=state.dimension,
        )
    )


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

state.trame__title = "Plotly"

with SinglePageWithDrawerLayout(server) as layout:
    layout.title.set_text("trame+plotly+umap")

    with layout.toolbar as tb:
        vuetify3.VTextField(
            type="number",
            label="sample_size",
            v_model_number=("sample_size", DEFAULTS["sample_size"]),
            hide_details=True,
            raw_attrs=[':min="100"', ':step="100"'],
        )

    with layout.drawer:
        vuetify3.VSelect(
            label="method",
            v_model=(
                "dimensionality_reduction_method",
                DEFAULTS["dimensionality_reduction_method"],
            ),
            items=(
                "reduction_methods",
                [
                    {"title": "umap", "value": "umap"},
                    {"title": "pca", "value": "pca"},
                    {"title": "t-sne", "value": "tsne"},
                ],
            ),
        )
        vuetify3.VBtn("Go", click=update_plot)
        vuetify3.VBtn("Reset 3D camera", click=server.controller.reset_camera)
        # pca hyperparameters
        with vuetify3.VCard(
            classes="mb-2 mx-1", v_show="dimensionality_reduction_method === 'pca'"
        ):
            with vuetify3.VCardText():
                pass

        # umap hyperparameters
        with vuetify3.VCard(
            classes="mb-2 mx-1", v_show="dimensionality_reduction_method === 'umap'"
        ):
            with vuetify3.VCardText():
                vuetify3.VSlider(
                    label="min_dist",
                    v_model=("min_dist", DEFAULTS["min_dist"]),
                    min=0,
                    max=100,
                    step=0.01,
                    hide_details=True,
                    thumb_label=True,
                )
                vuetify3.VSlider(
                    label="n_neighbors",
                    v_model=("n_neighbors", DEFAULTS["n_neighbors"]),
                    min=1,
                    max=1000,
                    step=1,
                    hide_details=True,
                    thumb_label=True,
                )
                vuetify3.VSlider(
                    label="spread",
                    v_model=("spread", DEFAULTS["spread"]),
                    min=1,
                    max=100,
                    step=1,
                    hide_details=True,
                    thumb_label=True,
                )
                vuetify3.VSlider(
                    label="repulsion_strength",
                    v_model=(
                        "repulsion_strength",
                        DEFAULTS["repulsion_strength"],
                    ),
                    min=1,
                    max=100,
                    step=1,
                    hide_details=True,
                    thumb_label=True,
                )
                vuetify3.VSelect(
                    v_model=("metric", DEFAULTS["metric"]),
                    label="metric",
                    items=(
                        "umap_metric_function",
                        [
                            {"title": "euclidean", "value": "euclidean"},
                            {"title": "manhattan", "value": "manhattan"},
                            {"title": "chebyshev", "value": "chebyshev"},
                            {"title": "minkowski", "value": "minkowski"},
                            {"title": "canberra", "value": "canberra"},
                            {"title": "braycurtis", "value": "braycurtis"},
                            {"title": "haversine", "value": "haversine"},
                            {"title": "mahalanobis", "value": "mahalanobis"},
                            {"title": "wminkowski", "value": "wminkowski"},
                            {"title": "seuclidean", "value": "seuclidean"},
                            {"title": "cosine", "value": "cosine"},
                            {"title": "correlation", "value": "correlation"},
                            {"title": "hamming", "value": "hamming"},
                            {"title": "jaccard", "value": "jaccard"},
                            {"title": "dice", "value": "dice"},
                            {"title": "russellrao", "value": "russellrao"},
                            {"title": "kulsinski", "value": "kulsinski"},
                            {"title": "rogerstanimoto", "value": "dice"},
                            {"title": "sokalmichener", "value": "sokalmichener"},
                            {"title": "sokalsneath", "value": "sokalsneath"},
                            {"title": "yule", "value": "yule"},
                        ],
                    ),
                )
        with vuetify3.VCard(
            classes="mb-2 mx-1", v_show="dimensionality_reduction_method === 'tsne'"
        ):
            with vuetify3.VCardText():
                vuetify3.VSlider(
                    label="perplexity",
                    v_model=("perplexity", DEFAULTS["perplexity"]),
                    min=0,
                    max=100,
                    step=1,
                    hide_details=True,
                    thumb_label=True,
                )

                vuetify3.VSlider(
                    label="max_iter",
                    v_model=("max_iter", DEFAULTS["max_iter"]),
                    min=10,
                    max=10000,
                    step=1,
                    hide_details=True,
                    thumb_label=True,
                )
        # common style options
        with vuetify3.VCard(classes="mb-2 mx-1"):
            with vuetify3.VCardText():
                vuetify3.VSelect(
                    v_model=("color_by", None),
                    items=(
                        "channel",
                        [
                            {"title": "None", "value": None},
                            {"title": "0", "value": 0},
                            {"title": "1", "value": 1},
                            {"title": "2", "value": 2},
                            {"title": "3", "value": 3},
                            {"title": "10", "value": 10},
                            {"title": "20", "value": 20},
                            {"title": "30", "value": 30},
                            {"title": "12", "value": 12},
                            {"title": "13", "value": 13},
                            {"title": "23", "value": 23},
                        ],
                    ),
                )
                vuetify3.VSlider(
                    label="marker size",
                    v_model=(
                        "marker_size",
                        DEFAULTS["marker_size"],
                    ),
                    min=0.1,
                    max=10,
                    step=0.01,
                    hide_details=True,
                    thumb_label=True,
                )
                vuetify3.VSwitch(
                    v_model=("use_coords", False),
                    label="Use ijk",
                    density="compact",
                    hide_details=True,
                    inset=True,
                    color="green",
                    classes="ml-2",
                    true_icon="mdi-check",
                    false_icon="mdi-close",
                )
                vuetify3.VSelect(
                    label="dimension",
                    v_model=("dimension", 2),
                    items=(
                        "dimensions",
                        [
                            {"title": "2", "value": 2},
                            {"title": "3", "value": 3},
                        ],
                    ),
                )

                vuetify3.VSelect(
                    label="clustering method",
                    v_model=(
                        "clustering_method",
                        DEFAULTS["clustering_method"],
                    ),
                    items=(
                        "clustering_methods",
                        [
                            {"title": "None", "value": None},
                            {"title": "kmeans", "value": "kmeans"},
                            {"title": "hdbscan", "value": "hdbscan"},
                            {"title": "optics", "value": "optics"},
                            {"title": "manual", "value": "manual"},
                        ],
                    ),
                )
                # kmeans panel
                with html.Div(
                    v_show="clustering_method === 'kmeans'",
                ):
                    vuetify3.VSpacer()
                    vuetify3.VSlider(
                        label="n_clusters",
                        v_model=("n_clusters", DEFAULTS["n_clusters"]),
                        min=1,
                        max=100,
                        step=1,
                        hide_details=True,
                        thumb_label=True,
                    )
                with html.Div(
                    v_show="clustering_method === 'hdbscan'",
                ):
                    vuetify3.VSpacer()
                    vuetify3.VSlider(
                        label="min_samples",
                        v_model=("min_samples", DEFAULTS["min_samples"]),
                        min=1,
                        max=1000,
                        step=1,
                        hide_details=True,
                        thumb_label=True,
                    )
                    vuetify3.VSlider(
                        label="min_cluster_size",
                        v_model=(
                            "min_cluster_size",
                            DEFAULTS["min_cluster_size"],
                        ),
                        min=1,
                        max=1000,
                        step=1,
                        hide_details=True,
                        thumb_label=True,
                    )
                # optics panel
                with html.Div(
                    v_show="clustering_method === 'optics'",
                ):
                    vuetify3.VSpacer()
                    vuetify3.VSlider(
                        label="min_samples",
                        v_model=("min_samples", DEFAULTS["min_samples"]),
                        min=1,
                        max=100,
                        step=1,
                        hide_details=True,
                        thumb_label=True,
                    )
                    vuetify3.VSlider(
                        label="max_eps",
                        v_model=("max_eps", DEFAULTS["max_eps"]),
                        min=1,
                        max=200,
                        step=1,
                        hide_details=True,
                        thumb_label=True,
                    )

                with vuetify3.VCard(
                    v_if="cluster_info && clustering_method != None",
                    classes="mb-2 mx-1",
                ):
                    vuetify3.VBtn("update opacity", click=update_opacity)
                    with vuetify3.VRow(
                        v_for=("data, name in cluster_info"),
                        key="name",
                        classes="mx-0 my-1",
                    ):
                        vuetify3.VLabel("Cluster opacity", classes="text-body-2 ml-1")
                        vuetify3.VSlider(
                            model_value=("data.opacity",),
                            min=0,
                            max=1,
                            step=0.05,
                            hide_details=True,
                            thumb_label=True,
                            update_modelValue="cluster_info[name].opacity = $event; flushState('cluster_info')",
                        )

                with vuetify3.VCard(classes="mb-2 mx-1"):
                    with vuetify3.VCardText():
                        html.Div("min_dist {{min_dist}}")
                        html.Div("n_neighbors {{n_neighbors}}")
                        html.Div("sample_size {{sample_size}}")
                        html.Div("spread {{spread}}")
                        html.Div("repulsion_strength {{repulsion_strength}}")
                        html.Div("dimension {{dimension}}")
    with layout.content:
        with vuetify3.VContainer(fluid=True, classes="fill-height"):
            with vuetify3.VCol(classes="fill-height"):
                with vuetify3.VRow(style="height:50%"):
                    figure_scatter = plotly.Figure(
                        style="flex:1",
                        display_logo=False,
                        display_mode_bar="true",
                        selected=
                        # "console.log($event)",
                        # (
                        #    on_scatter_selected_event,
                        #    "[$event.points.map((v)=>({x:v.x, y:v.y, id: v.pointIndex, metadata:v.customdata} ))]",
                        # ),
                        (
                            on_scatter_selected_event,
                            "[{selected: $event.points.map((v)=>({x:v.x, y:v.y, id: v.pointIndex, metadata:v.customdata} )) , lassoPoints: $event.lassoPoints, range: $event.range }]",
                        ),
                    )
                    server.controller.figure_scatter_update = figure_scatter.update

                    with vtk.VtkRemoteView(
                        VOLUME_VIEW.render_window, interactive_ratio=1, style="flex:1"
                    ) as html_view:
                        server.controller.reset_camera = html_view.reset_camera
                        server.controller.view_update = html_view.update

                with vuetify3.VRow(style="height:50%"):
                    figure_parallel_coords = plotly.Figure(
                        style="flex:1",
                        display_logo=False,
                        display_mode_bar="true",
                        # restyle="console.log('restyle',$event)",
                        restyle=(on_parallel_coords_select_event, "[$event[0]]"),
                    )
                    server.controller.figure_parallel_coords_update = (
                        figure_parallel_coords.update
                    )

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parse_arguments()
    preprocess(filename=FILENAME, labelfile=LABELMAP_FILENAME, drop_ones=False)
    preprocess_rba(filename=RGBA_FILENAME)
    server.controller.view_update()
    sample_data(DATA, DEFAULTS["sample_size"])
    server.start()
