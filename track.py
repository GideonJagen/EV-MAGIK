import sys
from pathlib import Path
import os
import gc

import tifffile
from tqdm import tqdm

module_path = os.path.abspath(os.path.join(".."))
sys.path.append(Path(module_path))

import deeptrack as dt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colormaps as cm

from collections import defaultdict
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


def plot_trajectories(tracks, filepath, outpath, pages_batch_size=1, stop=None):
    """
    Plot the predicted trajectories.

    Args:
        trajectories (pandas.DataFrame): Predicted trajectories.
        filepath (str): Path to the video file.
        outpath (str): Path to save the plot.
        pages_batch_size (int): Number of pages to process at a time, Default: 1.
        stop (int): Stops the plotting after frame with number stop is plotted, Default: None.
    """

    file = tifffile.TiffFile(Path(filepath))
    n_pages = len(file.pages)

    top = cm.get_cmap("Oranges_r")
    bottom = cm.get_cmap("Blues_r")

    colors = np.vstack(
        (
            top(np.linspace(0, 1, int(np.ceil(tracks["entity"].max() / 2)))),
            bottom(np.linspace(0, 1, int(np.ceil(tracks["entity"].max() / 2)))),
        )
    )
    np.random.shuffle(colors)

    if stop:
        iterations = int(stop // pages_batch_size + 1 + (stop % pages_batch_size > 0))
    else:
        iterations = int(
            tracks["frame"].max() // pages_batch_size
            + 1
            + (tracks["frame"].max() % pages_batch_size > 0)
        )

    for i in tqdm(range(iterations)):
        pages = file.pages[
            i * pages_batch_size : min((i + 1) * pages_batch_size, n_pages - 1)
        ]
        frames = np.array([page.asarray() for page in pages])
        frames = frames.astype(np.float32)

        # Plot
        for f, image in enumerate(frames):
            plt.figure(figsize=(15, 15))
            plt.imshow(image, cmap="gray")
            plt.text(0, 2, "Frame: " + str(f), fontsize=20, c="white")

            detections = tracks[(tracks["frame"] == i * pages_batch_size + f)]
            trail = tracks[
                (tracks["frame"] < i * pages_batch_size + f)
                & (tracks["frame"] >= i * pages_batch_size + f - 10)
            ]

            for t in trail["entity"].unique():
                entity_trail = trail[trail["entity"] == t]
                plt.plot(
                    entity_trail["x"],
                    entity_trail["y"],
                    color=colors[int(t)],
                    linewidth=1.5,
                )
            plt.scatter(
                detections["x"],
                detections["y"],
                linewidths=1.5,
                color=colors[detections["entity"].astype(int)],
                marker="o",
                s=200,
                facecolors="none",
            )

            plt.savefig(
                f"{Path(outpath)}/tracked_images/fig_{str(i*pages_batch_size + f).zfill(4)}.png"
            )

            # Clean up
            plt.clf()
            plt.close("all")
            gc.collect()

            if stop and i * pages_batch_size + f >= stop:
                break


def count_appearances(tracks_df):
    """
    Add a column to the tracks dataframe counting how many frames an entity has appeared in.

    Args:
        tracks_df (pandas.DataFrame): Tracks dataframe.

    Returns:
        pandas.DataFrame: Tracks dataframe with the 'appearances' column added.
    """
    tracks = tracks_df.copy()
    tracks["frame_count"] = tracks.groupby("entity")["frame"].cumsum()
    tracks["frame_count"] = tracks.groupby("entity")["frame_count"].transform(
        lambda x: x - x.iloc[0]
    )
    return tracks


def distance_traveled(tracks_df):
    """
    Calculate the distance traveled by each object in the tracks dataframe.

    Args:
        tracks_df (pandas.DataFrame): Tracks dataframe.

    Returns:
        pandas.DataFrame: Distance traveled by each object.
    """
    tracks = tracks_df.copy()
    tracks = delta_distance(tracks)
    tracks["traveled_y"] = tracks.groupby("entity")["delta_y"].cumsum()
    tracks["traveled_x"] = tracks.groupby("entity")["delta_x"].cumsum()

    tracks.sort_values(["frame", "entity"], inplace=True)

    return tracks


def delta_distance(tracks_df):
    sorted_tracks = tracks_df.sort_values(["entity", "frame"])
    sorted_tracks["delta_y"] = abs(sorted_tracks["y"].diff())
    sorted_tracks["delta_x"] = abs(sorted_tracks["x"].diff())
    sorted_tracks.loc[sorted_tracks["entity"].diff() != 0, ["delta_y", "delta_x"]] = 0
    return sorted_tracks


def remove_still_objects(
    tracks_df,
    cuttoff=10,
):
    """
    Remove still objects from the tracks dataframe.

    Args:
        tracks_df (pandas.DataFrame): Tracks dataframe.

    Returns:
        pandas.DataFrame: Tracks dataframe with still objects removed.
    """
    tracks = tracks_df.copy()
    for id in tracks["entity"].unique():
        entity = tracks[tracks["entity"] == id]
        if entity["y"].max() - entity["y"].min() < cuttoff:
            tracks = tracks[tracks["entity"] != id]
    return tracks


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
