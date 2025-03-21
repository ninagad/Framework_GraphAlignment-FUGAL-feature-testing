from enum import Enum, auto


class GraphEnums(Enum):

    def __repr__(self):
        return self.name

    INF_EUROROAD = auto()
    CA_NETSCIENCE = auto()
    BIO_CELEGANS = auto()
    SOCFB_BOWDOIN47 = auto()
    VOLES = auto()
    MULTIMAGNA = auto()
    NWS_K7 = auto()
    NWS_K70 = auto()
    NWS_P_0_point_5 = auto()
    NWS_N_K7 = auto()
    NWS_N_PROP_K = auto()
    SBM = auto()
    SBM_INTP_5_PERCENT = auto()
    SBM_INTP_15_PERCENT = auto()
    ER = auto()
