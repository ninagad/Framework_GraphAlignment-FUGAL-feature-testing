from enum import Enum


class PCAEnums(Enum):
    def __repr__(self):
        return self.name

    NO_PCA = 0
    ONE_COMPONENT = 1
    TWO_COMPONENTS = 2
    THREE_COMPONENTS = 3
    FOUR_COMPONENTS = 4
