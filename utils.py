import numpy as np
import pandas as pd
import tensorflow as tf
import deeptrack as dt

from matplotlib import pyplot as plt
from matplotlib import cm

from tqdm import tqdm
import gc
import tifffile
from pathlib import Path

import itertools
from collections import defaultdict
import warnings
import io
import sys


def load_models(
    loadstar_weight_name="Loadstar_EV_24462", magik_weight_name="MAGIK_MP_MPN"
):
    """
    Load the Loadstar and MAGIK models.

    Args:
        loadstar_weight_name (str): Name of the Loadstar weight file.
        magik_weight_name (str): Name of the MAGIK weight file.

    Returns:
        tuple: Tuple containing the loaded Loadstar and MAGIK models.
    """
    loadstar = load_loadstar_model(loadstar_weight_name)
    magik = load_magik_model(magik_weight_name)

    return loadstar, magik


def load_loadstar_model(weight_name="Loadstar_EV_24462"):
    """
    Load the Loadstar model.

    Args:
        weight_name (str): Name of the Loadstar weight file.

    Returns:
        deeptrack.models.LodeSTAR: Loaded Loadstar model.
    """
    loadstar = dt.models.LodeSTAR(input_shape=(None, None, 1))
    loadstar.load_weights(Path(f"../weights/loadstar/{weight_name}/{weight_name}"))
    return loadstar


def load_magik_model(weight_name="MAGIK_MP_MPN"):
    """
    Load the MAGIK model.

    Args:
        weight_name (str): Name of the MAGIK weight file.

    Returns:
        deeptrack.models.gnns.MAGIK: Loaded MAGIK model.
    """
    magik = dt.models.gnns.MAGIK(
        dense_layer_dimensions=(
            64,
            96,
        ),
        base_layer_dimensions=(
            96,
            96,
            96,
        ),
        number_of_node_features=2,
        number_of_edge_features=1,
        number_of_edge_outputs=1,
        edge_output_activation="sigmoid",
        output_type="edges",
        graph_block="MPN",
    )

    # Need to supress warnings for loading the model as MAGIK is not compatible expect_partial()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        magik.load_weights(Path(f"../weights/magik/{weight_name}.h5"))

    return magik


def test_loadstar(tiff_path, loadstar, alpha, cutoff, n_test_frames):
    """
    Test the Loadstar model on a TIFF file.

    Args:
        tiff_path (str): Path to the TIFF file.
        loadstar (deeptrack.models.LodeSTAR): Loaded Loadstar model.
        alpha (float): Alpha parameter for detection.
        cutoff (float): Cutoff parameter for detection.
        n_test_frames (int): Number of frames to test.

    Returns:
        list: List of detected objects.
    """
    detections_list = []

    file = tifffile.TiffFile(Path(tiff_path))

    pages = file.pages[:n_test_frames]
    frames = np.array([page.asarray() for page in pages])
    frames = frames.astype(np.float32)
    frames = frames - frames.mean()
    frames = frames / np.std(frames, axis=(0, 1), keepdims=True) / 3

    frames = np.expand_dims(frames, -1)

    stdout_trap = io.StringIO()
    sys.stdout = stdout_trap
    detections = loadstar.predict_and_detect(
        frames,
        alpha=alpha,
        beta=1 - alpha,
        cutoff=cutoff,
        mode="ratio",
    )
    sys.stdout = sys.__stdout__

    for j, detections in enumerate(detections):
        plt.figure(figsize=(10, 10))
        plt.imshow(frames[j], cmap="gray")
        plt.scatter(
            detections[:, 1],
            detections[:, 0],
            marker="o",
            color="r",
            s=100,
            facecolors="none",
        )

    return detections_list


def detect(image_paths, loadstar, batch_size=10, alpha=0.999, cutoff=1e-2, plot=False):
    """
    Detect objects in a TIFF file using the Loadstar model.

    Args:
        tiff_path (str): Path to the TIFF file.
        loadstar (deeptrack.models.LodeSTAR): Loaded Loadstar model.
        batch_size (int): Number of frames to process in each batch.
        alpha (float): Alpha parameter for detection.
        cutoff (float): Cutoff parameter for detection.
        plot (bool): Whether to plot the detected objects.

    Returns:
        pandas.DataFrame: Detected objects.
    """
    detections_list = []
    detections_df = pd.DataFrame()

    for j, tiff_path in enumerate(image_paths):

        file = tifffile.TiffFile(Path(tiff_path))

        n_pages = len(file.pages)
        iterations = n_pages // batch_size

        with tqdm(total=iterations, position=0, leave=True) as pbar:
            for i in tqdm(range(iterations), position=0, leave=True):
                pbar.update()
                pages = file.pages[
                    i * batch_size : min((i + 1) * batch_size, n_pages - 1)
                ]
                frames = np.array([page.asarray() for page in pages])
                frames = frames.astype(np.float32)
                frames = frames - frames.mean()
                frames = frames / np.std(frames, axis=(0, 1, 2), keepdims=True) / 3

                frames = np.expand_dims(frames, -1)

                stdout_trap = io.StringIO()
                sys.stdout = stdout_trap
                detections_all = loadstar.predict_and_detect(
                    frames, alpha=alpha, beta=1 - alpha, cutoff=cutoff, mode="ratio"
                )
                sys.stdout = sys.__stdout__

                detections_list.append(detections_all)

                if plot:
                    for j, detections in enumerate(detections_all):
                        plt.figure(figsize=(10, 10))
                        plt.imshow(frames[j], cmap="gray")
                        plt.scatter(
                            detections[:, 1],
                            detections[:, 0],
                            marker="o",
                            color="r",
                            s=100,
                            facecolors="none",
                        )
                        plt.savefig(
                            "../output/detected_images/fig_{}.png".format(
                                str(i * batch_size + j).zfill(4)
                            )
                        )

                        plt.cla()
                        plt.clf()
                        plt.close("all")
                        gc.collect()

        detections_df = pd.concat(
            [
                detections_df,
                _detections_to_df(detections_list, file.pages[0].shape, set=j),
            ]
        )

    return detections_df


def track_intensity(tracks_df, batch_size=10, tiff_path=None, mode="mean"):
    """
    Track the intensity of entities in a DataFrame.

    Args:
        tracks_df (pandas.DataFrame): The DataFrame containing the tracks data.
        batch_size (int, optional): The batch size for processing the tracks. Defaults to 10.
        tiff_path (str, optional): The path to the TIFF file. Defaults to None.
        mode (str, optional): The mode for tracking intensity. Can be "mean" or "max". Defaults to "mean".

    Returns:
        pandas.Series: A Series containing the tracked intensities for each entity.

    Raises:
        ValueError: If the column 'intensity' is not found in the tracks_df.
        ValueError: If an invalid mode is provided.
    """
    if "intensity" in tracks_df.columns:
        # Code to track intensity
        if mode == "mean":
            track_intensities = tracks_df.groupby("entity")["intensity"].mean()
        elif mode == "max":
            track_intensities = tracks_df.groupby("entity")["intensity"].max()
        else:
            raise ValueError("Invalid mode.")
        return track_intensities
    else:
        raise ValueError("Column 'intensity' not found in tracks_df.")


def detection_intensity(
    tiff_path,
    detections_df,
    batch_size=10,
    mode="pixel",
    kernel_size=None,  # Change to radius
):
    """
    Extract the intensities of detected objects in a TIFF file.

    Args:
        tiff_path (str): Path to the TIFF file.
        detections_df (pandas.DataFrame): Detected objects.
        batch_size (int): Number of frames to process in each batch.
        mode (str): Mode to extract intensity. Can be "pixel", "mean", or "max".
        kernel_size (int): Kernel size for mean and max modes.
    Returns:
        pandas.DataFrame: Detected objects with intensities.
    """
    if mode == "pixel":
        return _intensity_pixel(tiff_path, detections_df, batch_size)
    elif mode == "mean":
        if kernel_size is None:
            raise ValueError("Kernel size must be provided for mean mode.")
        return _intensity_kernel(
            tiff_path, detections_df, batch_size, mode, kernel_size
        )
    elif mode == "max":
        if kernel_size is None:
            raise ValueError("Kernel size must be provided for max mode.")
        return _intensity_kernel(
            tiff_path, detections_df, batch_size, mode, kernel_size
        )

    else:
        raise ValueError("Invalid mode.")


def warp_axis(detections_df, ratio, axis_name="centroid-0"):
    """
    Warp the x-axis of the detected objects.

    Args:
        detections_df (pandas.DataFrame): Detected objects.
        ratio (float): Ratio to warp the x-axis.

    Returns:
        pandas.DataFrame: Warped detected objects.
    """
    detections_df_w = detections_df.copy()
    detections_df_w[axis_name] = detections_df_w[axis_name] * ratio
    return detections_df_w


def unwarp_axis(detections_df, ratio, axis_name="centroid-0"):
    """
    Warp the x-axis of the detected objects.

    Args:
        detections_df (pandas.DataFrame): Detected objects.
        ratio (float): Ratio to warp the x-axis.

    Returns:
        pandas.DataFrame: Warped detected objects.
    """
    detections_df_w = detections_df.copy()
    detections_df_w[axis_name] = detections_df_w[axis_name] / ratio
    return detections_df_w


def save_detections(detection_df, path, full=False):
    """
    Save the detected objects to a CSV file.

    Args:
        detection_df (pandas.DataFrame): Detected objects.
        path (str): Path to save the CSV file.
        full (bool, optional): If True, save the full dataframe including all columns.
                               If False, save a modified dataframe without the "set", "label", and "solution" columns.
                               Defaults to False.
    """
    detection_df_d = detection_df.copy()
    if not full:
        detection_df_d.drop(columns=["label", "solution"], inplace=True)
    detection_df_d.to_csv(path, index=False)


def load_detections(path):
    """
    Loade detected objects from CSV file.

    Args:
        path (str): Path to load the CSV file.

    Returns:
        pandas.DataFrame: Detected objects.
    """
    # detection_df_d.drop(columns=["set", "label", "solution"], inplace=True)
    # detection_df_d.to_csv(path, index=False)
    detection_df = pd.read_csv(path)
    return detection_df


def _detections_to_df(detections_list, file_page_shape, set):
    size = 0
    for detections_all in detections_list:
        for detections in detections_all:
            size += detections.shape[0]
    print(size)

    detection_array = np.zeros((size, 6))

    traversed = 0
    list_traversed = 0
    for i, detections_all in enumerate(detections_list):
        for j, detections in enumerate(detections_all):
            detection_array[traversed : traversed + detections.shape[0], 0] = (
                list_traversed + j
            )
            detection_array[traversed : traversed + detections.shape[0], 4:] = (
                detections
            )
            traversed += detections.shape[0]
        list_traversed += len(detections_all)

    detection_array[:, 1] = set

    detection_df = pd.DataFrame(
        detection_array,
        columns=["frame", "set", "label", "solution", "centroid-1", "centroid-0"],
    ).astype(
        {
            "set": "int32",
            "frame": "int32",
            "label": "int32",
            "solution": "int32",
            "centroid-1": "float32",
            "centroid-0": "float32",
        }
    )
    detection_df["centroid-1"] /= file_page_shape[0]
    detection_df["centroid-0"] /= file_page_shape[1]

    return detection_df


def _intensity_pixel(tiff_path, detections_df, batch_size=10):
    detections_df = detections_df.copy()
    detections_df["intensity"] = np.nan

    file = tifffile.TiffFile(Path(tiff_path))

    n_pages = len(file.pages)
    iterations = n_pages // batch_size

    with tqdm(total=iterations, position=0, leave=True) as pbar:
        for i in tqdm(range(iterations), position=0, leave=True):
            pbar.update()
            pages = file.pages[i * batch_size : min((i + 1) * batch_size, n_pages - 1)]
            frames = np.array([page.asarray() for page in pages])
            frames = frames.astype(np.float32)

            # frames = np.expand_dims(frames, -1)

            detections = detections_df[
                (detections_df["frame"] >= i * batch_size)
                & (detections_df["frame"] < i * batch_size + batch_size)
            ]

            frame, x, y = (
                detections["frame"].values.astype(int) - i * batch_size,
                detections["x"].values.astype(int),
                detections["y"].values.astype(int),
            )

            intensities = frames[frame, y, x]

            detections_df.loc[
                (detections_df["frame"] >= i * batch_size)
                & (detections_df["frame"] < i * batch_size + batch_size),
                "intensity",
            ] = intensities

    return detections_df


def _intensity_kernel(tiff_path, detections_df, batch_size, mode, kernel_size):
    detections_df = detections_df.copy()
    detections_df["intensity"] = np.nan

    file = tifffile.TiffFile(Path(tiff_path))

    n_pages = len(file.pages)
    iterations = n_pages // batch_size

    with tqdm(total=iterations, position=0, leave=True) as pbar:
        for i in tqdm(range(iterations), position=0, leave=True):
            pbar.update()
            pages = file.pages[i * batch_size : min((i + 1) * batch_size, n_pages - 1)]
            frames = np.array([page.asarray() for page in pages])
            frames = frames.astype(np.float32)

            # frames = np.expand_dims(frames, -1)

            detections = detections_df[
                (detections_df["frame"] >= i * batch_size)
                & (detections_df["frame"] < i * batch_size + batch_size)
            ]

            frame, x, y = (
                detections["frame"].values.astype(int) - i * batch_size,
                detections["x"].values.astype(int),
                detections["y"].values.astype(int),
            )

            kernel_half = (kernel_size - 1) // 2
            intensities = []
            for f, cx, cy in zip(frame, x, y):
                xmin = max(0, cx - kernel_half)
                xmax = min(frames.shape[2] - 1, cx + kernel_half)
                ymin = max(0, cy - kernel_half)
                ymax = min(frames.shape[1] - 1, cy + kernel_half)
                area = frames[f, ymin : ymax + 1, xmin : xmax + 1]
                if mode == "mean":
                    intensity = np.mean(area)
                elif mode == "max":
                    intensity = np.max(area)
                else:
                    raise ValueError("Invalid mode.")
                intensities.append(intensity)
            intensities = np.array(intensities)

            detections_df.loc[
                (detections_df["frame"] >= i * batch_size)
                & (detections_df["frame"] < i * batch_size + batch_size),
                "intensity",
            ] = intensities

    return detections_df


def _intensities_max(tiff_path, detections_df, batch_size, kernel_size):
    detections_df_i = detections_df.copy()
    detections_df_i["intensity"] = np.nan

    file = tifffile.TiffFile(Path(tiff_path))

    n_pages = len(file.pages)
    iterations = n_pages // batch_size

    with tqdm(total=iterations, position=0, leave=True) as pbar:
        for i in tqdm(range(iterations), position=0, leave=True):
            pbar.update()
            pages = file.pages[i * batch_size : min((i + 1) * batch_size, n_pages - 1)]
            frames = np.array([page.asarray() for page in pages])
            frames = frames.astype(np.float32)
            frames = frames - frames.mean()
            frames = frames / np.std(frames, axis=(0, 1, 2), keepdims=True) / 3

            frames = np.expand_dims(frames, -1)

            for j, detections in enumerate(
                detections_df_i[detections_df_i["frame"] == i].values
            ):
                x, y = detections[-2], detections[-1]
                x, y = int(x * file.pages[0].shape[1]), int(y * file.pages[0].shape[0])

                intensity = frames[j, y, x]
                detections_df_i.loc[
                    (detections_df_i["frame"] == i)
                    & (detections_df_i["label"] == detections[2]),
                    "intensity",
                ] = intensity

    return detections_df_i
