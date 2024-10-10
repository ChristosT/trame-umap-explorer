r"""
Installation requirements:
    pip install trame trame-vuetify trame-plotly trame-components sklearn umap-learn
"""

import plotly.graph_objects as go

from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3, plotly, html
import numpy as np
import re
import umap
import sklearn.cluster as cluster
from sklearn import decomposition

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
    "marker_size": 1,
    "min_samples": 2,
    "min_cluster_size": 5,
    "max_eps": 100,
}
RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------
data = np.load("display_data.npz")["arr_0"]
sums = np.sum(data, axis=1)
data /= sums[:, None]
num_rows_to_select = 5000
# Randomly select unique row indices
random_indices = np.random.choice(data.shape[0], num_rows_to_select, replace=False)
points = data[random_indices]
U = None


def umap_fit(points, min_dist, n_neighbors, spread, repulsion_strength, dimension):
    fit = umap.UMAP(
        min_dist=min_dist,
        repulsion_strength=repulsion_strength,
        spread=spread,
        n_neighbors=n_neighbors,
        n_components=dimension,
    )
    u = fit.fit_transform(points)
    print("Done")
    return u


def pca_fit(points, dimension):
    fit = decomposition.PCA(n_components=dimension)
    u = fit.fit_transform(points)
    print("Done")
    return u


server = get_server(client_type="vue3")
state = server.state
state.trame__title = "umap explorer"


def sample_data(data, sample_size):
    global points
    random_indices = np.random.choice(data.shape[0], sample_size, replace=False)
    points = data[random_indices]


def fit_data(points):
    if state.dimensionality_reduction_method == "umap":
        return umap_fit(
            points,
            state.min_dist,
            state.n_neighbors,
            state.spread,
            state.repulsion_strength,
            state.dimension,
        )
    elif state.dimensionality_reduction_method == "pca":
        return pca_fit(points, state.dimension)


@state.change("color_by")
def on_color_by(color_by, **kwargs):
    if U is not None:
        server.controller.figure_scatter_update(scatter(U, COLOR_ARGS, state.dimension))


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
        server.controller.figure_parallel_coords_update(parallel_coords())


@state.change("marker_size")
def on_marker_size(marker_size, **kwargs):
    global COLOR_ARGS
    if U is not None:
        COLOR_ARGS["marker"]["size"] = marker_size

        server.controller.figure_scatter_update(scatter(U, COLOR_ARGS, state.dimension))


def get_color_args(color_by=None, clustering_method=None):
    size = state.marker_size
    if clustering_method is None:
        if color_by is None:
            color_args = {"marker_color": "blue", "marker": {"size": size}}
        else:
            if color_by < 10:
                color_args = {
                    "marker_color": points[:, color_by],
                    "marker": {"colorscale": "blues", "size": size},
                }
            else:
                color0 = color_by // 10
                color1 = color_by % 10
                color_args = {
                    "marker_color": points[:, color0] + points[:, color1],
                    "marker": {"colorscale": "blues", "size": size},
                }
    else:
        if clustering_method == "kmeans":
            labels = cluster.KMeans(n_clusters=state.n_clusters).fit_predict(U)
            color_args = {"marker_color": labels, "marker": {"size": size}}
        elif clustering_method == "hdbscan":
            labels = cluster.HDBSCAN(
                min_cluster_size=state.min_cluster_size,
                min_samples=state.min_samples,
            ).fit_predict(U)
            color_args = {"marker_color": labels, "marker": {"size": size}}
        elif clustering_method == "optics":
            labels = cluster.OPTICS(
                min_samples=state.min_samples, max_eps=state.max_eps
            ).fit_predict(U)
            color_args = {"marker_color": labels, "marker": {"size": size}}
        else:
            pass
    return color_args


def on_scatter_selected_event(selection_data):
    # example event [{'x': -0.1320136977614036, 'y': 0.772234734606532, 'metadata': [0.12115617198637327, 0, 0, 0.8788438280136267]}
    print(selection_data)


def on_parallel_coords_select_event(selection_data):
    # example event {'dimensions[0].constraintrange': [[0.3651979139717597, 0.48502199044130667]]}
    key = list(selection_data.keys())[0]
    channel = re.search(r"\d+", key).group()
    constrained_range = selection_data[key][0]
    print(f"{channel=}")
    print(f"{constrained_range=}")


def scatter(U, color_args=None, dimension=2):
    if dimension == 2:
        plot = go.Scatter(
            x=U[:, 0],
            y=U[:, 1],
            mode="markers",
            **color_args,
            customdata=points,
            hovertemplate="%{customdata[0]:.3f} <br> %{customdata[1]:.3f} <br> %{customdata[2]:.3f} <br> %{customdata[3]:.3f} ",
            name="",
        )
    else:
        plot = go.Scatter3d(
            x=U[:, 0],
            y=U[:, 1],
            z=U[:, 2],
            mode="markers",
            **color_args,
            customdata=points,
            hovertemplate="%{customdata[0]:.3f} <br> %{customdata[1]:.3f} <br> %{customdata[2]:.3f} <br> %{customdata[3]:.3f} ",
            name="",
        )
    return go.Figure(data=plot).update_layout(margin=dict(l=10, r=10, t=25, b=10))


def parallel_coords():
    return go.Figure(
        data=go.Parcoords(
            line=dict(color=COLOR_ARGS["marker_color"]),
            dimensions=list(
                [
                    dict(
                        range=[0, 1],
                        label="0",
                        values=points[:, 0],
                    ),
                    dict(
                        range=[0, 1],
                        label="1",
                        values=points[:, 1],
                    ),
                    dict(
                        range=[0, 1],
                        label="2",
                        values=points[:, 2],
                    ),
                    dict(
                        range=[0, 1],
                        label="3",
                        values=points[:, 3],
                    ),
                ]
            ),
        )
    )


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
def update_plot():
    global U, COLOR_ARGS
    sample_data(data, state.sample_size)
    U = fit_data(points)
    COLOR_ARGS = get_color_args(
        color_by=state.color_by, clustering_method=state.clustering_method
    )
    server.controller.figure_parallel_coords_update(parallel_coords())
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
        print(tb)

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
                ],
            ),
        )
        vuetify3.VBtn("Go", click=update_plot)
        with vuetify3.VCard(
            classes="mb-2 mx-1", v_show="dimensionality_reduction_method === 'pca'"
        ):
            with vuetify3.VCardText():
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
                    max=100,
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
                    max=3,
                    step=0.01,
                    hide_details=True,
                    thumb_label=True,
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
                        selected=(
                            on_scatter_selected_event,
                            "[$event.points.map((v)=>({x:v.x, y:v.y, metadata:v.customdata} ))]",
                        ),
                    )
                    server.controller.figure_scatter_update = figure_scatter.update
                    html.Div("hello", style="flex:1", classes="bg-red")
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
    server.start()
