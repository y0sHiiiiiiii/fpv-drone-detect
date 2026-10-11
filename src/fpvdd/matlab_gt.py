"""Read box labels from MATLAB Computer Vision Toolbox ``groundTruth`` .mat files.

``scipy.io.loadmat`` cannot decode the ``groundTruth`` MCOS object; its properties end up as
raw bytes in ``__function_workspace__``. That buffer is an ordinary MAT5 element stream, so
this module parses it into plain Python values and looks for the struct that holds
``LabelDefinitions`` and ``LabelData``. ``LabelData`` is a struct array with one element per
frame and one field per label; each field is an ``N x 4`` double matrix of
``[x, y, w, h]`` rows (MATLAB pixel coordinates), empty when the label is absent.
"""

import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PixelBox = tuple[float, float, float, float]
FrameLabels = dict[str, list[PixelBox]]

_MI_MATRIX = 14
_MI_COMPRESSED = 15
_NUMERIC = {
    1: "<i1", 2: "<u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4",
    7: "<f4", 9: "<f8", 12: "<i8", 13: "<u8",
}  # fmt: skip
_MX_CELL, _MX_STRUCT, _MX_OBJECT, _MX_CHAR = 1, 2, 3, 4
_MX_NUMERIC = set(range(6, 16))


class MatParseError(ValueError):
    """The buffer is not a MAT5 element stream we understand."""


@dataclass
class MatArray:
    """A parsed miMATRIX element."""

    mx_class: int
    dims: tuple[int, ...]
    name: str = ""
    data: Any = None  # ndarray (numeric), str (char), list[MatArray] (cell)
    fields: list[str] = field(default_factory=list)  # struct/object field names
    elements: list[dict[str, "MatArray"]] = field(default_factory=list)  # struct/object values
    children: list["MatArray"] = field(default_factory=list)  # opaque/other: nested matrices


def iter_elements(buf: bytes) -> Iterator[tuple[int, bytes]]:
    """Yield ``(type, payload)`` for each element; compressed elements are inflated.

    Regular elements have an 8-byte tag and are padded to 8 bytes. Small elements (payload
    of at most 4 bytes) pack type and size into one 4-byte tag and always occupy 8 bytes.
    """
    off = 0
    while off + 8 <= len(buf):
        word = struct.unpack_from("<I", buf, off)[0]
        if word >> 16:  # small element
            mi_type, nbytes, start = word & 0xFFFF, word >> 16, off + 4
            next_off = off + 8
        else:
            mi_type, nbytes = struct.unpack_from("<II", buf, off)
            start = off + 8
            end = start + nbytes
            # compressed elements are not padded
            next_off = end if mi_type == _MI_COMPRESSED else end + (-end % 8)
        if mi_type == 0 and nbytes == 0:
            return  # zero padding at the end of a buffer
        if start + nbytes > len(buf):
            raise MatParseError(f"element at {off} overruns the buffer")
        payload = buf[start : start + nbytes]
        if mi_type == _MI_COMPRESSED:
            yield from iter_elements(zlib.decompress(payload))
        else:
            yield mi_type, payload
        off = next_off


def _numeric(mi_type: int, payload: bytes) -> np.ndarray:
    if mi_type not in _NUMERIC:
        raise MatParseError(f"unexpected numeric element type {mi_type}")
    return np.frombuffer(payload, dtype=_NUMERIC[mi_type])


def parse_matrix(payload: bytes) -> MatArray:
    """Parse the payload of one miMATRIX element."""
    if not payload:
        return MatArray(mx_class=0, dims=(0, 0))
    elems = iter_elements(payload)
    try:
        _, flags = next(elems)
        dims_type, dims_raw = next(elems)
    except StopIteration as exc:
        raise MatParseError("miMATRIX without flags/dims") from exc
    mx_class = int(np.frombuffer(flags[:4], dtype="<u4")[0]) & 0xFF
    dims = tuple(int(d) for d in _numeric(dims_type, dims_raw))
    name_elem = next(elems, None)
    name = name_elem[1].decode("latin-1") if name_elem else ""
    arr = MatArray(mx_class, dims, name)
    rest = list(elems)

    if mx_class in _MX_NUMERIC:
        if rest:
            values = _numeric(*rest[0]).astype(np.float64)
            arr.data = values.reshape(dims, order="F") if values.size else values.reshape(0, 0)
        else:
            arr.data = np.zeros((0, 0))
    elif mx_class == _MX_CHAR:
        arr.data = _decode_char(*rest[0]) if rest else ""
    elif mx_class == _MX_CELL:
        arr.data = [parse_matrix(p) for t, p in rest if t == _MI_MATRIX]
    elif mx_class in (_MX_STRUCT, _MX_OBJECT):
        if mx_class == _MX_OBJECT:
            rest = rest[1:]  # class name
        name_len = int(_numeric(*rest[0])[0])
        raw_names = rest[1][1]
        arr.fields = [
            raw_names[i : i + name_len].split(b"\0")[0].decode("latin-1")
            for i in range(0, len(raw_names), name_len)
        ]
        values = [parse_matrix(p) for t, p in rest[2:] if t == _MI_MATRIX]
        n = len(arr.fields)
        if n and len(values) % n:
            raise MatParseError(f"struct with {n} fields has {len(values)} values")
        arr.elements = [
            dict(zip(arr.fields, values[i : i + n], strict=True))
            for i in range(0, len(values) if n else 0, n or 1)
        ]
    else:
        arr.children = [parse_matrix(p) for t, p in rest if t == _MI_MATRIX]
    return arr


def _decode_char(mi_type: int, payload: bytes) -> str:
    if mi_type == 16:
        return payload.decode("utf-8", errors="replace")
    return "".join(chr(int(c)) for c in _numeric(mi_type, payload))


def parse_stream(buf: bytes) -> list[MatArray]:
    """Parse a MAT5 element stream into its top-level matrices."""
    return [parse_matrix(p) for t, p in iter_elements(buf) if t == _MI_MATRIX]


def _walk(arr: MatArray) -> Iterator[MatArray]:
    yield arr
    nested: list[MatArray] = list(arr.children)
    if isinstance(arr.data, list):
        nested += arr.data
    for element in arr.elements:
        nested += element.values()
    for child in nested:
        yield from _walk(child)


def find_label_data(roots: list[MatArray]) -> MatArray:
    """Return the ``LabelData`` struct array of the first groundTruth found."""
    for root in roots:
        for arr in _walk(root):
            if arr.mx_class == _MX_STRUCT and {"LabelData", "LabelDefinitions"} <= set(arr.fields):
                return arr.elements[0]["LabelData"]
    raise MatParseError("no groundTruth (LabelData/LabelDefinitions) found")


def frame_labels(label_data: MatArray) -> list[FrameLabels]:
    """Convert a LabelData struct array into per-frame ``{label: [boxes]}``."""
    if label_data.mx_class != _MX_STRUCT:
        raise MatParseError(f"LabelData has unsupported MATLAB class {label_data.mx_class}")
    names = [f for f in label_data.fields if f != "Time"]
    frames: list[FrameLabels] = []
    for idx, element in enumerate(label_data.elements):
        labels: FrameLabels = {}
        for name in names:
            values = element[name].data
            if not isinstance(values, np.ndarray) or values.size == 0:
                continue
            if values.ndim != 2 or values.shape[1] != 4:
                raise MatParseError(f"frame {idx} {name}: expected N x 4 boxes, got {values.shape}")
            labels[name] = [tuple(float(v) for v in row) for row in values]
        frames.append(labels)
    return frames


def load_groundtruth(path: Path) -> list[FrameLabels]:
    """Per-frame boxes by label name from a MATLAB groundTruth .mat file."""
    import scipy.io

    workspace = scipy.io.loadmat(str(path)).get("__function_workspace__")
    if workspace is None:
        raise MatParseError(f"{path}: no __function_workspace__ (not a groundTruth export)")
    # The workspace buffer starts with an 8-byte preamble before the element stream.
    return frame_labels(find_label_data(parse_stream(workspace.tobytes()[8:])))
