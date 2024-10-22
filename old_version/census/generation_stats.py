import numpy as np


def get_sex_stats():
    layers = {
        0: [
            [63, 60, 64, 60, 63],
            [57, 56, 58, 57, 57],
            [57, 60, 63, 57, 61],
            [54, 53, 55, 54, 53],
            [57, 54, 54, 53, 54],
            [58, 58, 56, 56, 60],
            [56, 57, 56, 58, 56],
            [63, 59, 60, 59, 63],
            [60, 64, 63, 60, 57],
            [58, 60, 61, 62, 63]
        ],
        1: [
            [54, 54, 54, 55, 54],
            [58, 57, 57, 56, 57],
            [53, 54, 55, 53, 54],
            [56, 52, 53, 52, 53],
            [54, 51, 52, 51, 51],
            [52, 56, 52, 51, 53],
            [54, 55, 53, 55, 53],
            [56, 57, 55, 53, 54],
            [55, 52, 52, 52, 54],
            [58, 57, 56, 56, 63]
        ],
        2: [
            [55, 52, 52, 55, 52],
            [54, 56, 57, 57, 58],
            [54, 55, 54, 53, 55],
            [51, 51, 54, 53, 53],
            [51, 51, 52, 52, 52],
            [53, 54, 52, 52, 52],
            [53, 53, 52, 51, 52],
            [55, 53, 56, 53, 53],
            [54, 55, 53, 55, 53],
            [54, 54, 55, 54, 55]
        ]
    }
    return layers


def get_race_stats():
    layers = {
        0: [
            [70, 65, 70],
            [80, 80, 70],
            [80, 80, 80],
            [70, 75, 70],
            [80, 80, 85],
            [76, 80, 75],
            [70, 70, 65],
            [75, 75, 80],
            [70, 70, 70],
            [70, 75, 75]
        ],
        1: [
            [60, 60, 60],
            [60, 55, 60],
            [70, 75, 80],
            [60, 65, 60],
            [60, 65, 70],
            [60, 60, 60],
            [60, 60, 65],
            [60, 65, 65],
            [75, 70, 80],
            [65, 65, 65]
        ],
        2: [
            [55, 55, 55],
            [65, 60, 65],
            [60, 65, 55],
            [55, 55, 60],
            [55, 55, 55],
            [65, 60, 65],
            [65, 60, 65],
            [60, 60, 60],
            [60, 60, 60],
            [60, 65, 55]
        ]
    }
    return layers


# info = get_sex_stats()
info = get_race_stats()
for k, v in info.items():
    v = np.mean(v, 1)
    print("Mean: %.2f | Max: %.2f" % (np.mean(v), np.max(v)))
