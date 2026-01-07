#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import warnings
from pathlib import Path
from typing import TypeVar

import imageio
import numpy as np

JsonLike = str | int | float | bool | None | list["JsonLike"] | dict[str, "JsonLike"] | tuple["JsonLike", ...]
T = TypeVar("T", bound=JsonLike)


def write_video(video_path, stacked_frames, fps):
    """
    Write video from stacked frames.
    
    Args:
        video_path: Path to save the video file
        stacked_frames: Array of frames, shape (t, h, w, c) or (t, c, h, w) or similar
        fps: Frames per second for the video
    
    The function normalizes the frame format to HWC (height, width, channels) with
    uint8 dtype and 1-4 channels, as required by imageio.
    """
    # Convert to numpy array if not already
    frames = np.asarray(stacked_frames)
    
    # Handle different input shapes
    if frames.ndim == 4:
        # Could be (t, h, w, c) or (t, c, h, w)
        # Check if channel dimension is first or last by comparing sizes
        t, dim1, dim2, dim3 = frames.shape
        # If first dimension after time is small (likely channels), it's CHW format
        if dim1 < 8:  # Channels are typically 1, 3, or 4
            # Shape is (t, c, h, w) -> convert to (t, h, w, c)
            frames = np.transpose(frames, (0, 2, 3, 1))
        # Otherwise assume (t, h, w, c) format
    elif frames.ndim == 3:
        # Single channel image (t, h, w) -> add channel dimension (t, h, w, 1)
        frames = frames[..., np.newaxis]
    elif frames.ndim == 5:
        # Extra batch dimension (b, t, h, w, c) or (b, t, c, h, w)
        # Take first batch element
        frames = frames[0]
        # Recursively handle the 4D case
        if frames.shape[1] < 8:  # Likely (t, c, h, w)
            frames = np.transpose(frames, (0, 2, 3, 1))
    else:
        raise ValueError(
            f"Unsupported frame shape: {frames.shape}. "
            "Expected (t, h, w, c), (t, c, h, w), or (b, t, h, w, c)"
        )
    
    # Now frames should be (t, h, w, c)
    t, h, w, c = frames.shape
    
    # Ensure channel count is valid (1, 2, 3, or 4)
    if c > 4:
        # If more than 4 channels, take first 3 (RGB) or convert to grayscale
        if c >= 3:
            frames = frames[..., :3]  # Take RGB channels
            c = 3
        else:
            # Take first channel
            frames = frames[..., :1]
            c = 1
    elif c == 0:
        raise ValueError(f"Invalid channel count: {c}")
    
    # Normalize pixel values to [0, 255] uint8 range
    if frames.dtype == np.float32 or frames.dtype == np.float64:
        # Check if values are in [0, 1] range
        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            # Values might be in [0, 255] range already, just convert type
            frames = np.clip(frames, 0, 255).astype(np.uint8)
    elif frames.dtype != np.uint8:
        # Convert other integer types to uint8
        frames = frames.astype(np.uint8)
    
    # Filter out DeprecationWarnings raised from pkg_resources
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "pkg_resources is deprecated as an API", category=DeprecationWarning
        )
        imageio.mimsave(video_path, frames, fps=fps)


def deserialize_json_into_object(fpath: Path, obj: T) -> T:
    """
    Loads the JSON data from `fpath` and recursively fills `obj` with the
    corresponding values (strictly matching structure and types).
    Tuples in `obj` are expected to be lists in the JSON data, which will be
    converted back into tuples.
    """
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)

    def _deserialize(target, source):
        """
        Recursively overwrite the structure in `target` with data from `source`,
        performing strict checks on structure and type.
        Returns the updated version of `target` (especially important for tuples).
        """

        # If the target is a dictionary, source must be a dictionary as well.
        if isinstance(target, dict):
            if not isinstance(source, dict):
                raise TypeError(f"Type mismatch: expected dict, got {type(source)}")

            # Check that they have exactly the same set of keys.
            if target.keys() != source.keys():
                raise ValueError(
                    f"Dictionary keys do not match.\nExpected: {target.keys()}, got: {source.keys()}"
                )

            # Recursively update each key.
            for k in target:
                target[k] = _deserialize(target[k], source[k])

            return target

        # If the target is a list, source must be a list as well.
        elif isinstance(target, list):
            if not isinstance(source, list):
                raise TypeError(f"Type mismatch: expected list, got {type(source)}")

            # Check length
            if len(target) != len(source):
                raise ValueError(f"List length mismatch: expected {len(target)}, got {len(source)}")

            # Recursively update each element.
            for i in range(len(target)):
                target[i] = _deserialize(target[i], source[i])

            return target

        # If the target is a tuple, the source must be a list in JSON,
        # which we'll convert back to a tuple.
        elif isinstance(target, tuple):
            if not isinstance(source, list):
                raise TypeError(f"Type mismatch: expected list (for tuple), got {type(source)}")

            if len(target) != len(source):
                raise ValueError(f"Tuple length mismatch: expected {len(target)}, got {len(source)}")

            # Convert each element, forming a new tuple.
            converted_items = []
            for t_item, s_item in zip(target, source, strict=False):
                converted_items.append(_deserialize(t_item, s_item))

            # Return a brand new tuple (tuples are immutable in Python).
            return tuple(converted_items)

        # Otherwise, we're dealing with a "primitive" (int, float, str, bool, None).
        else:
            # Check the exact type.  If these must match 1:1, do:
            if type(target) is not type(source):
                raise TypeError(f"Type mismatch: expected {type(target)}, got {type(source)}")
            return source

    # Perform the in-place/recursive deserialization
    updated_obj = _deserialize(obj, data)
    return updated_obj
