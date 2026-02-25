# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
MapAnything Demo: Inference from calibrated RGBD images.

Usage:
    python demo_calibrated_rgbd_inference.py --help
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import json
import numpy as np
from scipy.spatial.transform import Rotation
import sys
import time
import torch

from mapanything.models import MapAnything
from mapanything.utils.filesystem import (
    collect_corresponding_file_paths,
    FileCollectorInput,
)
from mapanything.utils.image import preprocess_inputs


def convert_pose(pose_4x4):
    assert isinstance(pose_4x4, np.ndarray)
    assert (
        len(pose_4x4.shape) == 2
    ), f"Expected a matrix not a {len(pose_4x4.shape)}D array."
    assert (
        pose_4x4.shape[0] == 4 and pose_4x4.shape[1] == 4
    ), f"Expected a 4x4 matrix not {pose_4x4.shape[0]}x{pose_4x4.shape[1]}."

    q = Rotation.from_matrix(pose_4x4[:3, :3]).as_quat(scalar_first=True)
    t = pose_4x4[:3, 3]

    return q, t


def create_pose_graph_node(id: str, q: np.ndarray, t: np.ndarray) -> dict:
    assert isinstance(id, str)
    assert isinstance(q, np.ndarray)
    assert isinstance(t, np.ndarray)
    assert len(q.shape) == 1, f"Expected 1D array not {len(q.shape)}D."
    assert len(t.shape) == 1, f"Expected 1D array not {len(t.shape)}D."
    assert q.shape[0] == 4, f"Expected 4 quaternion elements not {q.shape[0]}."
    assert t.shape[0] == 3, f"Expected 3 translation elements not {t.shape[0]}."

    q_normalized = q / np.linalg.norm(q)

    pose3 = {
        "data_name": "pose3",
        "data": {
            "q": {
                "w": float(q_normalized[0]),
                "x": float(q_normalized[1]),
                "y": float(q_normalized[2]),
                "z": float(q_normalized[3]),
            },
            "t": {
                "x": float(t[0]),
                "y": float(t[1]),
                "z": float(t[2]),
            },
        },
    }

    return {
        "id": id,
        "data_items": [pose3],
    }


def save_pose_graph(
    id_and_camera_pose_list: list,
    map_config_file_path: str,
):
    assert isinstance(id_and_camera_pose_list, list)
    assert id_and_camera_pose_list, "Empty camera poses list."
    assert isinstance(map_config_file_path, str)
    assert os.path.isfile(
        map_config_file_path
    ), f"Map config file doesn't exist: '{map_config_file_path}'."

    with open(map_config_file_path) as map_config_file:
        map_config = json.load(map_config_file)

    map_pose_graph_folder = os.path.join(
        os.path.dirname(map_config_file_path),
        map_config["pose_graph_folder"],
    )
    os.makedirs(map_pose_graph_folder, exist_ok=True)

    map_pose_graph_file_path = os.path.join(
        map_pose_graph_folder,
        map_config["map_pose_graph"],
    )

    obj_file_path = os.path.splitext(map_pose_graph_file_path)[0] + ".obj"
    obj_file = open(obj_file_path, "w")
    graph_nodes = []

    for id, pose_4x4 in id_and_camera_pose_list:
        q, t = convert_pose(pose_4x4)
        # JSON stuff.
        graph_nodes.append(create_pose_graph_node(id, q, t))
        # OBJ stuff.
        obj_file.write(f"v {t[0]} {t[1]} {t[2]} 255 0 0\n")

    obj_file.close()

    with open(map_pose_graph_file_path, "w") as map_pose_graph_file:
        map_pose_graph = {
            "num_graph_nodes": len(id_and_camera_pose_list),
            "num_graph_edges": 0,
            "nodes": graph_nodes,
        }
        json.dump(map_pose_graph, map_pose_graph_file, indent=4)

    print(f"Saved:")
    print(f"* {obj_file_path}")
    print(f"* {map_pose_graph_file_path}")


def get_file_collector_inputs(map_config_file_path: str):
    assert isinstance(map_config_file_path, str)
    assert os.path.isfile(
        map_config_file_path
    ), f"Map config file doesn't exist: '{map_config_file_path}'."
    map_config_folder = os.path.dirname(map_config_file_path)

    with open(map_config_file_path) as map_config_file:
        map_config = json.load(map_config_file)

    images_header_file_path = os.path.join(
        map_config_folder, map_config["images_header"]
    )
    depth_header_file_path = os.path.join(map_config_folder, map_config["depth_header"])

    with open(images_header_file_path) as images_header_file:
        images_header = json.load(images_header_file)
    with open(depth_header_file_path) as depth_header_file:
        depth_header = json.load(depth_header_file)

    images_folder = os.path.join(
        os.path.dirname(images_header_file_path),
        images_header["directory"],
    )
    depth_folder = os.path.join(
        os.path.dirname(depth_header_file_path),
        depth_header["directory"],
    )

    return [
        FileCollectorInput(
            folder=images_folder,
            keyword=map_config["left_keyword"],
            file_extension=images_header["file_extension"],
        ),
        FileCollectorInput(
            folder=depth_folder,
            keyword=map_config["depth_keyword"],
            file_extension=depth_header["file_extension"],
        ),
    ]


def load_intrinsics(map_config_file_path):
    assert isinstance(map_config_file_path, str)
    assert os.path.isfile(
        map_config_file_path
    ), f"Map config file doesn't exist: '{map_config_file_path}'."

    with open(map_config_file_path) as map_config_file:
        map_config = json.load(map_config_file)

    camera_file_path = os.path.join(
        os.path.dirname(map_config_file_path),
        map_config["camera_parameters_file_path"],
    )
    assert os.path.isfile(camera_file_path), f"Camera parameters file doesn't exist: '{camera_file_path}'."

    with open(camera_file_path) as camera_file:
        camera_parameters = json.load(camera_file)

    fx = camera_parameters["left_intrinsics"]["fx"]
    fy = camera_parameters["left_intrinsics"]["fy"]
    cx = camera_parameters["left_intrinsics"]["cx"]
    cy = camera_parameters["left_intrinsics"]["cy"]

    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def load_inputs(map_config_file_path: str):
    file_collector_inputs = get_file_collector_inputs(map_config_file_path)
    assert (
        len(file_collector_inputs) == 2
    ), f"We need pairs not {len(file_collector_inputs)}-tuples."

    print("Collecting input files from:")
    for file_collector_input in file_collector_inputs:
        print(f"* {file_collector_input.folder}")

    corresponding_file_paths = collect_corresponding_file_paths(file_collector_inputs)
    assert corresponding_file_paths, "Couldn't load anything."

    intrinsics = load_intrinsics(map_config_file_path)
    ids = []
    views = []

    for id, file_paths in corresponding_file_paths.items():
        assert len(file_paths) == 2, f"We need pairs not {len(file_paths)}-tuples."
        bgr_image = cv2.imread(file_paths[0], cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        depth_image = cv2.imread(file_paths[1], cv2.IMREAD_ANYDEPTH) / 1000.0
        ids.append(id)
        views.append(
            {
                "img": rgb_image,
                "depth_z": depth_image.astype(np.float32),
                "intrinsics": intrinsics.astype(np.float32),
                "is_metric_scale": torch.tensor([True], device="cuda"),
            }
        )

    return ids, preprocess_inputs(views)


def main(map_config_file_path: str):
    assert isinstance(map_config_file_path, str)
    assert os.path.isfile(
        map_config_file_path
    ), f"Map file path doesn't exist: '{map_config_file_path}'."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading input data...")
    view_ids, views = load_inputs(map_config_file_path)
    assert views, "Didn't load anything."
    assert len(view_ids) == len(
        views
    ), f"Size mismatch: {len(view_ids)} vs {len(views)}."
    print(f"Loaded {len(views)} frames.")

    # Initialize model from HuggingFace
    if False:
        model_name = "facebook/map-anything-apache"
        print("Loading Apache 2.0 licensed MapAnything model...")
    else:
        model_name = "facebook/map-anything"
        print("Loading CC-BY-NC 4.0 licensed MapAnything model...")
    model = MapAnything.from_pretrained(model_name).to(device)

    # Run model inference with memory-efficient defaults
    print("Running inference...")
    start_time = time.perf_counter()
    outputs = model.infer(
        views,
        memory_efficient_inference=True,
        minibatch_size=1,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
    )
    elapsed_time_min = (time.perf_counter() - start_time) / 60.0
    print(f"Inference complete [took {elapsed_time_min:.2f} min].")

    # Save the timing to file.
    map_statistics_file_path = os.path.join(
        os.path.dirname(map_config_file_path),
        "map_statistics.txt",
    )
    with open(map_statistics_file_path, "w") as f:
        f.write(f"Mapping took {elapsed_time_min:.2f} minute(s).\n")

    # Save the camera poses.
    id_and_pose_list = []
    for id, prediction in zip(view_ids, outputs):
        camera_pose = prediction["camera_poses"][0].cpu().numpy()  # (4, 4)
        id_and_pose_list.append((id, camera_pose))
    save_pose_graph(id_and_pose_list, map_config_file_path)

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python demo_calibrated_rgbd_inference.py <path/to/map_config.json>"
        )
    else:
        map_config_file_path = sys.argv[1]
        main(map_config_file_path)
