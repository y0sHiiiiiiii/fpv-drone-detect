import io

import numpy as np
import pytest
import scipy.io

from fpvdd.matlab_gt import (
    MatParseError,
    find_label_data,
    frame_labels,
    iter_elements,
    parse_stream,
)

LABELS = ("AIRPLANE", "BIRD", "DRONE", "HELICOPTER")
MAT_HEADER = 128


def _label_data(frames: list[dict[str, list[list[float]]]]) -> np.ndarray:
    """Build a MATLAB-like LabelData struct array (n x 1) with one N x 4 matrix per label."""
    dtype = [("Time", object)] + [(name, object) for name in LABELS]
    data = np.empty((len(frames), 1), dtype=dtype)
    for i, frame in enumerate(frames):
        data[i, 0]["Time"] = np.array([[float(i)]])
        for name in LABELS:
            boxes = frame.get(name, [])
            data[i, 0][name] = np.array(boxes, dtype=float) if boxes else np.zeros((0, 0))
    return data


def _stream(frames, compress: bool = False, wrap_in_cell: bool = True) -> bytes:
    """MAT5 element stream (file without its 128-byte header) holding a groundTruth-like struct."""
    gt = {
        "DataSource": {"Source": "video.mp4"},
        "LabelDefinitions": {"Name": "DRONE"},
        "LabelData": _label_data(frames),
    }
    payload = {"props": np.array([[gt]], dtype=object) if wrap_in_cell else gt}
    buf = io.BytesIO()
    scipy.io.savemat(buf, payload, do_compression=compress)
    return buf.getvalue()[MAT_HEADER:]


FRAMES = [
    {},
    {"DRONE": [[10.5, 20.5, 30.0, 40.0]]},
    {"BIRD": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]},  # several boxes, one label
    {"BIRD": [[1, 1, 2, 2]], "DRONE": [[50, 60, 5, 5]]},
]


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("wrap_in_cell", [False, True])
def test_roundtrip(compress: bool, wrap_in_cell: bool) -> None:
    roots = parse_stream(_stream(FRAMES, compress, wrap_in_cell))
    frames = frame_labels(find_label_data(roots))
    assert len(frames) == len(FRAMES)
    assert frames[0] == {}
    assert frames[1] == {"DRONE": [(10.5, 20.5, 30.0, 40.0)]}
    # rows must come out as boxes, not as columns (MATLAB stores column-major)
    assert frames[2]["BIRD"] == [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)]
    assert set(frames[3]) == {"BIRD", "DRONE"}


def test_small_elements_are_parsed() -> None:
    # small element: type 5 (int32), 4 bytes, packed tag; followed by a normal uint8 element
    small = (5 | (4 << 16)).to_bytes(4, "little") + (7).to_bytes(4, "little")
    normal = (2).to_bytes(4, "little") + (3).to_bytes(4, "little") + b"abc" + b"\0" * 5
    elements = list(iter_elements(small + normal))
    assert elements == [(5, (7).to_bytes(4, "little")), (2, b"abc")]


def test_truncated_stream_rejected() -> None:
    stream = _stream(FRAMES)
    with pytest.raises(MatParseError):
        parse_stream(stream[: len(stream) // 2])


def test_missing_groundtruth_rejected() -> None:
    buf = io.BytesIO()
    scipy.io.savemat(buf, {"x": np.eye(2)})
    with pytest.raises(MatParseError, match="no groundTruth"):
        find_label_data(parse_stream(buf.getvalue()[MAT_HEADER:]))


def test_wrong_box_shape_rejected() -> None:
    frames = [{"DRONE": [[1.0, 2.0, 3.0]]}]  # 3 columns
    with pytest.raises(MatParseError, match="N x 4"):
        frame_labels(find_label_data(parse_stream(_stream(frames))))
