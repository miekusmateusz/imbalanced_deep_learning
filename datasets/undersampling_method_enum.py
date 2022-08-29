from enum import Enum


class UnderSamplingMethod(str, Enum):
    TOMEK_LINKS = 'tomek_links'
    EDITED_NEAREST_NEIGHBOUR = 'edited_nearest_neighbour'
    RANDOM_UNDERSAMPLING = 'random_undersampling'
