import logging

logger = logging.getLogger(__name__)


def dispatch(name: str) -> None:
    logger.info(f"Listing segmentation labels for {name}")
    if name == "synthseg":
        for label, values in SYNTHSEG_LABELS.items():
            print(f"{label}: {values}")
    elif name == "neuroquant":
        for label, values in NEUROQUANT_LABELS.items():
            print(f"{label}: {values}")
    else:
        raise ValueError(f"Unknown segmentation labels {name}")


SYNTHSEG_LABELS = {
    "BACKGROUND": [0],
    "LEFT_CEREBRAL_WHITE_MATTER": [2],
    "LEFT_CEREBRAL_CORTEX": [3],
    "LEFT_LATERAL_VENTRICLE": [4],
    "LEFT_INFERIOR_LATERAL_VENTRICLE": [5],
    "LEFT_CEREBELLUM_WHITE_MATTER": [7],
    "LEFT_CEREBELLUM_CORTEX": [8],
    "LEFT_THALAMUS": [10],
    "LEFT_CAUDATE": [11],
    "LEFT_PUTAMEN": [12],
    "LEFT_PALLIDUM": [13],
    "THIRD_VENTRICLE": [14],
    "FOURTH_VENTRICLE": [15],
    "BRAIN_STEM": [16],
    "LEFT_HIPPOCAMPUS": [17],
    "LEFT_AMYGDALA": [18],
    "LEFT_ACCUMBENS_AREA": [26],
    "CSF": [24],
    "LEFT_VENTRAL_DC": [28],
    "RIGHT_CEREBRAL_WHITE_MATTER": [41],
    "RIGHT_CEREBRAL_CORTEX": [42],
    "RIGHT_LATERAL_VENTRICLE": [43],
    "RIGHT_INFERIOR_LATERAL_VENTRICLE": [44],
    "RIGHT_CEREBELLUM_WHITE_MATTER": [46],
    "RIGHT_CEREBELLUM_CORTEX": [47],
    "RIGHT_THALAMUS": [49],
    "RIGHT_CAUDATE": [50],
    "RIGHT_PUTAMEN": [51],
    "RIGHT_PALLIDUM": [52],
    "RIGHT_HIPPOCAMPUS": [53],
    "RIGHT_AMYGDALA": [54],
    "RIGHT_ACCUMBENS_AREA": [58],
    "RIGHT_VENTRAL_DC": [60],
    "CEREBRUM": [2, 41, 3, 42],
    "FLUID": [24, 14, 15, 4, 43, 5, 44, 0],
    "VENTRICLE": [14, 15, 4, 43, 5, 44],
    "CEREBRAL_WHITE_MATTER": [2, 41],
    "CEREBRAL_CORTEX": [3, 42],
    "CEREBELLUM_WHITE_MATTER": [7, 46],
    "CEREBELLUM_CORTEX": [8, 47],
    "CEREBELLUM_GRAY_MATTER": [8, 47],
}


NEUROQUANT_LABELS = {
    "BACKGROUND": [0],
    "CENTRAL_WHITE_MATTER": [1],
    "CORTICAL_GRAY_MATTER": [2],
    "THIRD_VENTRICLE": [3],
    "FOURTH_VENTRICLE": [4],
    "FIFTH_VENTRICLE": [5],
    "LATERAL_VENTRICLE": [6],
    "INF_LAT_VENTRICLE": [7],
    "LAT_VENTRICLE_CHOROID_PLEXUS": [8],
    "CEREBELLUM_WHITE_MATTER": [9],
    "CEREBELLUM_GRAY_MATTER": [10],
    "HIPPOCAMPUS": [11],
    "AMYGDALA": [12],
    "THALAMUS": [13],
    "CAUDATE": [14],
    "PUTAMEN": [15],
    "PALIDUM": [16],
    "VENTRAL_DC": [17],
    "NUCLEUS_ACCUMBENS": [18],
    "BRAIN_STEM": [19],
    "INF_LAT_VENTRICLE_CHOROID_PLEXUS": [20],
    "CEREBRAL_WHITE_MATTER_HYPOINTENSITY": [21],
    "EXTERIOR": [22],
    "UNKNOWN": [23],
    "POST_SUP_TEMPORAL_SULCUS": [24],
    "CAUD_ANT_CINGULATE": [25],
    "PREMOTOR": [26],
    "CORPUS_CALLOSUM": [27],
    "CUNEUS_CORTEX": [28],
    "ENTORHINAL_CORTEX": [29],
    "FUSIFORM_CORTEX": [30],
    "INF_PARIETAL_CORTEX": [31],
    "INF_TEMPORAL_CORTEX": [32],
    "ISTHMUS_CINGULATE": [33],
    "LAT_OCCIPITAL": [34],
    "LAT_ORBITOFRONTAL_GYRUS": [35],
    "LINGUAL_GYRUS": [36],
    "MED_ORBITOFRONTAL_GYRUS": [37],
    "MID_TEMPORAL_GYRUS": [38],
    "PARAHIPPOCAMPAL_GYRUS": [39],
    "PARACENTRAL": [40],
    "PARS_OPERCULARIS": [41],
    "PARS_ORBITALIS": [42],
    "PARS_TRIANGULARIS": [43],
    "PERICALCARINE": [44],
    "PRI_SENSORY_CORTEX": [45],
    "POST_CINGULATE_GYRUS": [46],
    "PRI_MOTOR_CORTEX": [47],
    "MED_PARIETAL_CORTEX": [48],
    "ROST_ANT_CINGULATE": [49],
    "ANT_MID_FRONTAL": [50],
    "SUP_FRONTAL_GYRUS": [51],
    "SUP_PARIETAL_LOBULE": [52],
    "SUP_TEMPORAL": [53],
    "SUPRAMARGINAL_GYRUS": [54],
    "FRONTAL_POLE": [55],
    "TEMPORAL_POLE": [56],
    "TRANSVERSE_TEMPORAL": [57],
    "INSULA": [58],
    "FLUID": [3, 4, 5, 6, 7, 8, 20],
    "VENTRICLE": [3, 4, 5, 6, 7, 8],
}
