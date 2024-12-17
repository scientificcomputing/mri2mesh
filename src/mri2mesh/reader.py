from __future__ import annotations
from pathlib import Path
import logging
import typing

import numpy as np
import numpy.typing as npt

from dataclasses import dataclass, field
import nibabel as nib

from .segmentation_labels import NEUROQUANT_LABELS, SYNTHSEG_LABELS

logger = logging.getLogger(__name__)


@dataclass  # (slots=True) # - will be added when we drop python3.9
class Segmentation:
    img: np.ndarray
    labels: dict[str, list[int]] = field(default_factory=lambda: SYNTHSEG_LABELS)
    resolution: npt.NDArray[np.float64] = field(default_factory=lambda: np.array((1, 1, 1)))
    origin: npt.NDArray[np.float64] = field(default_factory=lambda: np.array((0, 0, 0)))
    padding: int = 0
    _hemisphere_min_distance: float = 0.0

    def add_padding(self, padding: int) -> None:
        self.img = np.pad(self.img, padding)
        self.origin = -np.array(self.resolution) * padding

    @property
    def hemisphere_min_distance(self) -> float:
        return self._hemisphere_min_distance

    @hemisphere_min_distance.setter
    def hemisphere_min_distance(self, min_distance: float) -> None:
        if (
            min_distance > 0
            and "LEFT_CEREBRAL_CORTEX" in self.labels
            and "RIGHT_CEREBRAL_CORTEX" in self.labels
            and "CSF" in self.labels
        ):
            from .morphology import seperate_labels

            l1 = self.labels["LEFT_CEREBRAL_CORTEX"]
            l2 = self.labels["RIGHT_CEREBRAL_CORTEX"]
            self.img = seperate_labels(self.img, l1, l2, min_distance, self.labels["CSF"][0])

        self._hemisphere_min_distance = min_distance


def read(
    input: Path,
    input_robust: Path | None = None,
    label_name: typing.Literal["synthseg", "neuroquant"] = "synthseg",
    padding: int = 5,
) -> Segmentation:
    logger.info(f"Loading segmentation from {input}")
    seg = nib.load(input)
    img = np.pad(seg.get_fdata(), padding)
    logger.debug(f"Loaded segmentation with shape {img.shape} and padding {padding}")

    if label_name == "synthseg":
        labels = SYNTHSEG_LABELS
    elif label_name == "neuroquant":
        labels = NEUROQUANT_LABELS
    else:
        raise ValueError(f"Invalid labels {label_name}")

    if input_robust:
        img_rob = np.pad(nib.load(input_robust).get_fdata(), padding)

        img[img_rob == 0] = 0
        img[img_rob == labels["BRAIN_STEM"]] = labels["BRAIN_STEM"]
    resolution: npt.NDArray[np.float64] = np.array(seg.header["pixdim"][1:4])
    origin: npt.NDArray[np.float64] = -np.array(resolution) * padding
    return Segmentation(
        img=img,
        labels=labels,
        resolution=resolution,
        origin=origin,
        padding=padding,
    )
