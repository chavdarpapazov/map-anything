# Copyright 2026 Toyota Research Institute.  All rights reserved.
import collections
import glob
import os

FileCollectorInput = collections.namedtuple(
    "FileCollectorInput", ["folder", "keyword", "file_extension"]
)


def collect_corresponding_file_paths(file_collector_input):
    """
    This function collects and returns corresponding files in the provided input
    folders. Only file paths having the specified file extension and those whose
    file stem contains the provided keyword will be returned (the file stem is
    the file name minus the file extension). Files are considered corresponding
    if their file stems minus the keyword are identical. For example, assume we
    have some images and key-points like this:
    "/imgs/l_0.png", "/imgs/l_1.png", ..., "/imgs/l_N.png",
    "/kpts/l_0.kpt", "/kpts/l_1.kpt", ..., "/kpts/l_N.kpt",
    "/imgs/r_0.png", "/imgs/r_1.png", ..., "/imgs/r_N.png",
    "/kpts/r_0.kpt", "/kpts/r_1.kpt", ..., "/kpts/r_N.kpt",
    and assume the input to this function is
    [{"/imgs", "l_", ".png"}, {"/kpts", "l_", ".kpt"},
     {"/imgs", "r_", ".png"}, {"/kpts", "r_", ".kpt"}],
    then the output will be a map of key-to-4-tuples of the form:
    "0" -> ("/imgs/l_0.png", "/kpts/l_0.kpt", "/imgs/r_0.png", "/imgs/r_0.kpt")
    "1" -> ("/imgs/l_1.png", "/kpts/l_1.kpt", "/imgs/r_1.png", "/imgs/r_1.kpt")
    ...
    "N" -> ("/imgs/l_N.png", "/kpts/l_N.kpt", "/imgs/r_N.png", "/imgs/r_N.kpt")
    The function only returns complete tuples, that is tuples with non-empty
    strings (incomplete tuples may arise if, for example, some images do not have
    corresponding keypoints)."""
    assert isinstance(file_collector_input, list)
    tuple_size = len(file_collector_input)
    corresponding_file_paths = {}

    # Collect the corresponding file paths.
    for k in range(tuple_size):
        assert isinstance(file_collector_input[k], FileCollectorInput)
        folder = file_collector_input[k].folder
        keyword = file_collector_input[k].keyword
        file_extension = file_collector_input[k].file_extension

        assert os.path.isdir(folder), f"Input directory '{folder}' doesn't exist."
        file_paths = glob.glob(os.path.join(folder, f"*{file_extension}"))

        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            assert len(file_name) > len(file_extension)
            file_stem = file_name[: -len(file_extension)]

            if keyword not in file_stem:
                continue

            key = file_stem.replace(keyword, "")
            if key not in corresponding_file_paths:
                corresponding_file_paths[key] = []
            corresponding_file_paths[key].append(file_path)
            assert (
                len(corresponding_file_paths[key]) <= tuple_size
            ), f"Too many file paths for key '{key}'."

    # Remove incomplete tuples, i.e., ones that contain an empty file path.
    for key in list(corresponding_file_paths.keys()):
        if len(corresponding_file_paths[key]) < tuple_size:
            del corresponding_file_paths[key]

    return corresponding_file_paths
