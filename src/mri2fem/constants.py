from typing import TypedDict


class SegmentationLabels(TypedDict):
    CSF: list[int]
    WM_LEFT: list[int]
    WM_RIGHT: list[int]
    GM_LEFT: list[int]
    GM_RIGHT: list[int]
    GM_CEREBELLUM_LEFT: list[int]
    GM_CEREBELLUM_RIGHT: list[int]
    BRAIN_STEM: list[int]
    V3: list[int]
    V4: list[int]
    LV: list[int]
    LV_INF: list[int]
    VENTRICLE: list[int]
    NONE: list[int]
    FLUID: list[int]
    CEREBRUM: list[int]


SYNTHSEG_LABELS = {
    "CSF": [24],
    "WM_LEFT": [2],
    "WM_RIGHT": [41],
    "GM_LEFT": [3],
    "GM_RIGHT": [42],
    "GM_CEREBELLUM_LEFT": [8],
    "GM_CEREBELLUM_RIGHT": [47],
    "BRAIN_STEM": [16],
    "V3": [14],
    "V4": [15],
    "LV": [4, 43],
    "LV_INF": [5, 44],
    "VENTRICLE": [14, 15, 4, 43, 5, 44],
    "NONE": [0],
}
SYNTHSEG_LABELS["FLUID"] = (
    SYNTHSEG_LABELS["CSF"] + SYNTHSEG_LABELS["VENTRICLE"] + SYNTHSEG_LABELS["NONE"]
)
SYNTHSEG_LABELS["CEREBRUM"] = (
    SYNTHSEG_LABELS["WM_LEFT"]
    + SYNTHSEG_LABELS["WM_RIGHT"]
    + SYNTHSEG_LABELS["GM_LEFT"]
    + SYNTHSEG_LABELS["GM_RIGHT"]
)


MM2M = 1e-3
HOLE_THRESHOLD = 1000
