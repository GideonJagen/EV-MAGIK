import sys
from pathlib import Path
import os

module_path = os.path.abspath(os.path.join(".."))
sys.path.append(Path(module_path))

import deeptrack as dt
import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib import cm
import itertools


def predict_trajectories(model, detections, radius=0.08, nframes=7, threshold=6):
    """
    Predict trajectories using the given model and detections.

    Args:
        model: The trajectory prediction model.
        detections (pandas.DataFrame): Detected objects.
        radius (float): Radius parameter for trajectory prediction.
        nframes (int): Number of frames parameter for trajectory prediction.
        threshold (int): Trajectory length threshold parameter for trajectory prediction.

    Returns:
        pandas.DataFrame: Predicted trajectories.
    """
    variables = dt.DummyFeature(
        radius=radius,
        output_type="edges",
        nofframes=nframes,
    )

    print("Predicting trajectories...")
    pred, gt, scores, graph = dt.models.gnns.get_predictions(
        detections, ["centroid"], model, **variables.properties()
    )
    print("Creating dataframe...")
    edges_df, nodes, _ = dt.models.gnns.df_from_results(pred, gt, scores, graph)

    print("Getting trajectories...")
    trajs = dt.models.gnns.to_trajectories(edges_df=edges_df)
    trajs = list(filter(lambda t: len(t) > threshold, trajs))
    trajs = [sorted(t) for t in trajs]

    object_index = np.zeros((nodes.shape[0]))
    for i, traj in enumerate(trajs):
        object_index[traj] = i + 1

    nodes_df = pd.DataFrame(
        {
            "frame": nodes[:, 0],
            "y": nodes[:, 1],
            "x": nodes[:, 2],
            "entity": object_index,
        }
    )
    nodes_df = nodes_df[nodes_df["entity"] != 0]
    print("Done!")

    return nodes_df


def save_trajectories(traj_df, path):
    """
    Save the predicted trajectories to a CSV file.

    Args:
        traj_df (pandas.DataFrame): Predicted trajectories.
        path (str): Path to save the CSV file.
    """
    traj_df.to_csv(path, index=False)


def load_trajectories(path):
    """
    Load the predicted trajectories from a CSV file.

    Args:
        path (str): Path to load the CSV file.

    Returns:
        pandas.DataFrame: Predicted trajectories.
    """
    return pd.read_csv(path)


def _f(d):
    max_scores = d.groupby("frame_y")["score"].transform("max")
    d.loc[d["score"] < max_scores, "prediction"] = False
    return d


def _to_trajectories(
    edges_df: pd.DataFrame,
    assert_no_splits=True,
    _type="prediction",
):
    edges_df["frame_diff"] = edges_df["frame_y"] - edges_df["frame_x"]
    if assert_no_splits and _type != "gt":
        edges_df_grouped = edges_df.groupby(["node_x"])
        edges_df = edges_df_grouped.apply(_f)
    edges_df["frame_diff"] = edges_df["frame_y"] - edges_df["frame_x"]
    edges_df_grouped = edges_df.groupby(["node_y"])
    edges_dfs = [_edge_df for _, _edge_df in edges_df_grouped]

    color = cm.viridis(np.linspace(0, 1, 250))
    color = itertools.cycle(color)
    t = defaultdict(lambda: next(color))

    x_iterator = iter(range(10000000))
    tr = {}
    sc = {}
    alpha = 0.1
    parents = set()
    d_alpha = (1 - alpha) / edges_df["prediction"].max()
    trajectories = []
    for _edge_df in edges_dfs:
        solutions = _edge_df.loc[_edge_df[_type] == 1.0, :]
        solutions = solutions.loc[
            solutions["frame_diff"] == solutions["frame_diff"].min(), :
        ]
        if _type == "prediction":
            solutions = solutions.loc[solutions["score"] == solutions["score"].max(), :]
        for i in range(len(solutions)):
            node_x = int(solutions[i : i + 1]["node_x"])
            alpha = 0.1 + d_alpha * int(solutions[i : i + 1]["frame_y"])

            if node_x not in t:
                tr[node_x] = next(x_iterator)
                sc[node_x] = float(solutions[i : i + 1]["score"])

            if node_x in parents:
                break

            t[str(int(solutions[i : i + 1]["node_y"]))] = t[str(node_x)]
            tr[int(solutions[i : i + 1]["node_y"])] = tr[node_x]
            parents.add(node_x)

    traj_dict = defaultdict(list)

    for key, val in tr.items():
        traj_dict[val].append(key)

    for val in traj_dict.values():
        trajectories.append(val)

    return trajectories
