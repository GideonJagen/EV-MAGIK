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

    nodes_df = pd.DataFrame(columns=["set", "frame", "y", "x", "entity"])

    for set_num in detections["set"].unique():

        set_detections = detections[detections["set"] == set_num].reset_index(drop=True)
        # set_detections = set_detections.reindex()

        print("Predicting trajectories...")
        pred, gt, scores, graph = dt.models.gnns.get_predictions(
            set_detections, ["centroid"], model, **variables.properties()
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

        nodes_df_set = pd.DataFrame(
            {
                "set": set_num,
                "frame": nodes[:, 0],
                "y": nodes[:, 1],
                "x": nodes[:, 2],
                "entity": object_index,
            }
        )
        nodes_df_set = nodes_df_set[nodes_df_set["entity"] != 0]
        nodes_df = pd.concat([nodes_df, nodes_df_set])
        print("Done!")

    return nodes_df


def plot_trajectories(
    tracks, filepath, outpath, pages_batch_size=1, set_nums="all", stop=None
):
    """
    Plot the predicted trajectories.

    Args:
        tracks (pandas.DataFrame): DataFrame containing the predicted trajectories.
        filepath (str): Path to the video file.
        outpath (str): Path to save the plot.
        pages_batch_size (int): Number of pages to process at a time, Default: 1.
        set_nums (list or int): Set numbers to plot trajectories for. If "all", plot trajectories for all sets. Default: "all".
        stop (int): Stops the plotting after frame with number stop is plotted, Default: None.
    """

    if not Path(f"{outpath}/tracked_images").exists():
        Path(f"{outpath}/tracked_images").mkdir(parents=True, exist_ok=True)

    if set_nums == "all":
        set_nums = tracks["set"].unique()
    elif type(set_nums) is not list and type(set_nums) is not tuple:
        set_nums = [set_nums]

    for i in set_nums:
        tracks_set = tracks[tracks["set"] == i]

        file = tifffile.TiffFile(Path(filepath[i]))
        n_pages = len(file.pages)

        top = cm.get_cmap("Oranges_r")
        bottom = cm.get_cmap("Blues_r")

        colors = np.vstack(
            (
                top(
                    np.linspace(0, 1, int(np.ceil(tracks_set["entity"].max() / 2)) + 1)
                ),
                bottom(
                    np.linspace(0, 1, int(np.ceil(tracks_set["entity"].max() / 2)) + 1)
                ),
            )
        )
        np.random.shuffle(colors)

        if stop:
            iterations = int(
                stop // pages_batch_size + 1 + (stop % pages_batch_size > 0)
            )
        else:
            iterations = int(
                tracks_set["frame"].max() // pages_batch_size
                + 1
                + (tracks_set["frame"].max() % pages_batch_size > 0)
            )

        for j in tqdm(range(iterations)):
            pages = file.pages[
                j * pages_batch_size : min((j + 1) * pages_batch_size, n_pages - 1)
            ]
            frames = np.array([page.asarray() for page in pages])
            frames = frames.astype(np.float32)

            # Plot
            for f, image in enumerate(frames):
                plt.figure(figsize=(15, 15))
                plt.imshow(image, cmap="gray")
                plt.text(0, 2, "Frame: " + str(f), fontsize=20, c="white")

                detections = tracks_set[
                    (tracks_set["frame"] == j * pages_batch_size + f)
                ]
                trail = tracks_set[
                    (tracks_set["frame"] < j * pages_batch_size + f)
                    & (tracks_set["frame"] >= j * pages_batch_size + f - 10)
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
                    f"{Path(outpath)}/tracked_images/fig_set_{i}_img_{str(j*pages_batch_size + f).zfill(4)}.png"
                )

                # Clean up
                plt.clf()
                plt.close("all")
                gc.collect()

                if stop and j * pages_batch_size + f >= stop:
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
    tracks["frame_count"] = tracks["frame"]
    tracks["frame_count"] = tracks.groupby(["set", "entity"])["frame_count"].transform(
        lambda x: x - x.iloc[0]
    )
    return tracks


def distance_traveled(tracks_df, delta_abs=False):
    """
    Calculate the distance traveled by each object in the tracks dataframe.

    Args:
        tracks_df (pandas.DataFrame): Tracks dataframe.
        delta_abs (bool): If True, calculate the absolute value of the delta distance traveled. Default: False.

    Returns:
        pandas.DataFrame: Distance traveled by each object.
    """
    tracks = tracks_df.copy()
    tracks = tracks.sort_values(["set", "entity", "frame"])
    for set_num in tracks["set"].unique():
        tracks_set = tracks[tracks["set"] == set_num]
        tracks_set = delta_distance(tracks_set, delta_abs)
        tracks_set["traveled_y"] = (
            tracks_set.abs().groupby("entity")["delta_y"].cumsum()
        )
        tracks_set["traveled_x"] = (
            tracks_set.abs().groupby("entity")["delta_x"].cumsum()
        )
        tracks.loc[
            tracks["set"] == set_num, ["delta_y", "delta_x", "traveled_y", "traveled_x"]
        ] = tracks_set[["delta_y", "delta_x", "traveled_y", "traveled_x"]].values

    # tracks.sort_values(["frame", "entity"], inplace=True)

    return tracks


def delta_distance(tracks_df, delta_abs):
    sorted_tracks = tracks_df.sort_values(["entity", "frame"])

    sorted_tracks["delta_y"] = sorted_tracks.groupby("entity")["y"].diff()
    sorted_tracks["delta_x"] = sorted_tracks.groupby("entity")["x"].diff()

    if delta_abs:
        sorted_tracks["delta_y"] = sorted_tracks["delta_y"].abs()
        sorted_tracks["delta_x"] = sorted_tracks["delta_x"].abs()

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
    for set_num in tracks["set"].unique():
        tracks_set = tracks[tracks["set"] == set_num]
        for id in tracks_set["entity"].unique():
            entity = tracks_set[tracks_set["entity"] == id]
            if entity["y"].max() - entity["y"].min() < cuttoff:
                tracks = tracks[(tracks["entity"] != id) | (tracks["set"] != set_num)]
                # tracks = tracks[(tracks["entity"] != id) | (tracks["set"] != set_num)]
    return tracks


# This is a very basic combine tracks function, will be improved in the future
# To include things like the y value and the distance between the tracks
def combine_tracks(tracks_df, channel_width, combine_length=10, remove_length=50):
    """
    Combine tracks that match in x value with existing tracks.

    Args:
        tracks_df (pandas.DataFrame): Tracks dataframe.

        combine_length (int): The maximum number of frames that can separate two tracks for them to be combined.

        remove_length (int): The maximum number of frames that can separate a track from its previous frames for it to be removed.

    Returns:
        pandas.DataFrame: Combined tracks dataframe.
    """
    tracks = tracks_df.copy()
    for entity in tracks["entity"].unique():
        entity_tracks = tracks[tracks["entity"] == entity]
        first_frame = entity_tracks["frame"].min()
        x_first_frame = entity_tracks[entity_tracks["frame"] == first_frame][
            "x"
        ].values[0]

        prev_frames = tracks[
            ((tracks["frame"] <= first_frame))
            & (tracks["x"] < x_first_frame + channel_width)
            & (tracks["x"] > x_first_frame - channel_width)
        ]

        if abs(prev_frames["frame"].values[0] - first_frame) < combine_length:
            # This works because the entity numbers are lower the earlier they appear
            # And the dataframe is sorted by entity number
            entity_prev = prev_frames["entity"].values[0]
            tracks.loc[tracks["entity"] == entity, "entity"] = entity_prev
        elif abs(prev_frames["frame"].values[0] - first_frame) < remove_length:
            tracks = tracks[tracks["entity"] != entity]

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
