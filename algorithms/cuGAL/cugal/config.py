"""Configuration of cuGAL."""

from dataclasses import dataclass
import dataclasses
from enum import Enum
from typing import Optional
from enums.pcaEnums import PCAEnums
from enums.scalingEnums import ScalingEnums

import torch


class SinkhornMethod(str, Enum):
    """The method used for Sinkhorn-Knopp."""

    STANDARD = "STANDARD"
    """The standard Sinkhorn-Knopp method.
    This is faster, but usually requires double-precision floats.
    """

    LOG = "LOG"
    """Perform Sinkhorn-Knopp in logarithmic space.
    Computational intensive but numerically stable and optimized for the GPU.
    """


class HungarianMethod(Enum):
    """The method used for Hungarian algorithm."""

    SCIPY = "SCIPY"
    GREEDY = "GREEDY"
    DENSE = "DENSE"
    SPARSE = "SPARSE"


@dataclass(frozen=True)
class Config:
    """Configuration of the CUGAL algorithm."""

    safe_mode: bool = False
    """If true, the algorithm will check for NaNs and Infs."""

    device: str = "cpu"
    """The torch device used for computations."""

    dtype: torch.dtype = torch.float64
    """The data type used for computations."""

    sinkhorn_regularization: float = 1
    """Regularization of the cost matrix when running Sinkhorn.

    Higher values can help with numeric stability, but can lower accuracy."""

    sinkhorn_method: SinkhornMethod = SinkhornMethod.LOG
    """The version of Sinkhorn used."""

    sinkhorn_iterations: int = 500
    """The maximum number of sinkhorn iterations performed."""

    sinkhorn_threshold: float = 1e-3
    """The marginal error threshold tolerated when running Sinkhorn."""

    sinkhorn_eval_freq: int = 10
    """How many Sinhorn iterations performed between checking for the potential of stopping."""

    sinkhorn_momentum: bool = True
    """If true, the Sinkhorn algorithm will use momentum."""

    mu: float = 0.5
    """The contribution of node features in finding the alignment."""

    nu: float = None
    """The contribution of node features in finding the alignment."""

    scaling: ScalingEnums = ScalingEnums.NO_SCALING
    """The scaling of the features."""

    pca: PCAEnums = PCAEnums.NO_PCA
    """The PCA method used."""

    iter_count: int = 15
    """The number of iterations to perform."""

    frank_wolfe_iter_count: int = 10
    """The number of Frank-Wolfe iterations to perform."""

    frank_wolfe_threshold: float | None = None
    """The max difference of the objective before stopping when running Frank-Wolfe."""

    use_sparse_adjacency: bool = False # sparse adjacency is not implemented
    """Use sparse matrix representation for adjacency matrices."""

    use_sinkhorn_warm_start: bool = True

    sinkhorn_momentum_start: Optional[int] = None

    recompute_distance: bool = True
    """Avoid storing distance matrix by doing recalculating each iteration."""

    hungarian_method: HungarianMethod = HungarianMethod.SCIPY
    """The version of Hungarian algorithm used."""

    def convert_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Convert tensor to correct type and dtype."""

        return input.to(dtype=self.dtype, device=self.device)

    def to_dict(self) -> dict:
        config_dict = dataclasses.asdict(self)
        config_dict['dtype'] = str(self.dtype).removeprefix("torch.")
        config_dict['sinkhorn_method'] = self.sinkhorn_method.value
        config_dict['hungarian_method'] = self.hungarian_method.value
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: dict):
        config_dict['dtype'] = getattr(torch, config_dict['dtype'])
        config_dict['sinkhorn_method'] = SinkhornMethod(
            config_dict['sinkhorn_method'])
        config_dict['hungarian_method'] = HungarianMethod[config_dict['hungarian_method']]
        return cls(**config_dict)
