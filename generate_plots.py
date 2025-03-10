import subprocess
from enum import Enum, auto
import os
import sys
import argparse


class GraphEnums(Enum):


    INF_EUROROAD = auto()
    CA_NETSCIENCE = auto()
    BIO_CELEGANS = auto()
    SOCFB_BOWDOIN47 = auto()
    VOLES = auto()
    MULTIMAGMA = auto()
    NWS_K7 = auto()
    NWS_K70 = auto()
    NWS_P_0_point_5 = auto()
    NWS_N_K7 = auto()
    NWS_N_PROP_K = auto()
    SBM = auto()
    SBM_INTP_5_PERCENT = auto()
    SBM_INTP_15_PERCENT = auto()
    ER = auto()



class PlotGenerator():
    def __init__(self):
        self.venv_python = self.activate_venv()

        self.baseline_dict = {GraphEnums.INF_EUROROAD: 20,
                              GraphEnums.CA_NETSCIENCE: 19,
                              GraphEnums.BIO_CELEGANS: 36,
                              GraphEnums.SOCFB_BOWDOIN47: 44,
                              GraphEnums.VOLES: 60,
                              GraphEnums.MULTIMAGMA: 59,
                              GraphEnums.NWS_K7: 110,
                              GraphEnums.SBM: 111,
                              GraphEnums.ER: 112
                              }

        # Instantiate plot+type dict
        noise = "Noise-level"
        non_community = 'p'
        community = 'External p'
        nws_k = 'k'
        nws_n = 'n'

        self.xaxis_dict = {}

        for graph in GraphEnums:
            self.xaxis_dict[graph] = noise

        self.xaxis_dict[GraphEnums.NWS_K7] = non_community
        self.xaxis_dict[GraphEnums.NWS_K70] = non_community
        self.xaxis_dict[GraphEnums.ER] = non_community
        self.xaxis_dict[GraphEnums.SBM] = community
        self.xaxis_dict[GraphEnums.SBM_INTP_5_PERCENT] = community
        self.xaxis_dict[GraphEnums.SBM_INTP_15_PERCENT] = community
        self.xaxis_dict[GraphEnums.NWS_P_0_point_5] = nws_k
        self.xaxis_dict[GraphEnums.NWS_N_K7] = nws_n
        self.xaxis_dict[GraphEnums.NWS_N_PROP_K] = nws_n


    @staticmethod
    def activate_venv():
        """Activate the virtual environment in a cross-platform way."""
        if os.name == "nt":  # Windows
            venv_python = os.path.join("venv", "Scripts", "python.exe")
        else:  # macOS/Linux
            venv_python = os.path.join("venv", "bin", "python")

        # Ensure the virtual environment exists
        if not os.path.exists(venv_python):
            print("Virtual environment not found! Please create it using:")
            print("python -m venv venv")
            sys.exit(1)

        return venv_python

    def generate_plots(self, source_dict: dict, output_dir: str):
        for graph, source in source_dict.items():

            xaxis = self.xaxis_dict[graph]

            # Add baseline if it exists, otherwise no baseline
            try:
                baseline = self.baseline_dict[graph]

                print(70 * '-')
                print(
                    f'Running with baseline: {baseline}, source: {source}, outputdir: {output_dir}, xaxis: {xaxis}')
                print(70 * '-')
                subprocess.run(
                    [self.venv_python, "plot.py",
                     "--baseline", str(baseline),
                     "--source", str(source),
                     "--outputdir", output_dir,
                     "--xaxis", xaxis,
                     "--yaxis", self.yaxis], capture_output=True, check=True)

            # Baseline not defined -> run without baseline
            # subprocess.SubprocessError -> FileNotFoundError, catches the cases where the baseline does not have a frob file.
            except (KeyError, subprocess.SubprocessError) as e:
                print(70 * '-')
                print(f'Running WITHOUT baseline, source: {source}, outputdir: {output_dir}, xaxis: {xaxis}')
                print(70 * '-')
                subprocess.run(
                    [self.venv_python, "plot.py",
                     "--source", str(source),
                     "--outputdir", output_dir,
                     "--xaxis", xaxis,
                     "--yaxis", self.yaxis])

    def generate_top_tree_combinations(self):
        #top_three_comb_sources = {GraphEnums.INF_EUROROAD: 73,
        #                          GraphEnums.CA_NETSCIENCE: 71,
        #                          GraphEnums.SOCFB_BOWDOIN47: 74,
        #                          GraphEnums.VOLES: 72,
        #                          }

        top_three_comb_sources = {GraphEnums.INF_EUROROAD: 158,
                                  GraphEnums.CA_NETSCIENCE: 157,
                                  #GraphEnums.SOCFB_BOWDOIN47: 161,
                                  GraphEnums.VOLES: 160,
                                  }

        output_dir = 'top-3-combinations'

        # Generate top-3-combinations plots
        self.generate_plots(top_three_comb_sources, output_dir)

    def generate_mu_test(self):
        _0_point_1_sources = {GraphEnums.INF_EUROROAD: 27,
                              GraphEnums.CA_NETSCIENCE: 14,
                              GraphEnums.SOCFB_BOWDOIN47: 35,
                              }

        _0_point_7_sources = {GraphEnums.INF_EUROROAD: 28,
                              GraphEnums.CA_NETSCIENCE: 15,
                              }

        _1_point_3_sources = {GraphEnums.INF_EUROROAD: 30,
                              GraphEnums.CA_NETSCIENCE: 16,
                              }

        _1_point_9_sources = {GraphEnums.INF_EUROROAD: 29,
                              GraphEnums.CA_NETSCIENCE: 17,
                              GraphEnums.BIO_CELEGANS: 31,
                              }

        _2_point_5_sources = {GraphEnums.INF_EUROROAD: 26,
                              GraphEnums.CA_NETSCIENCE: 18,
                              GraphEnums.BIO_CELEGANS: 32,
                              GraphEnums.VOLES: 49,
                              GraphEnums.MULTIMAGMA: 58}

        source_dicts = [_0_point_1_sources,
                        _0_point_7_sources,
                        _1_point_3_sources,
                        _1_point_9_sources,
                        _2_point_5_sources]

        output_dir = 'mu-test'

        for source_dict in source_dicts:
            self.generate_plots(source_dict, output_dir)

    def generate_2hop_features(self):
        source_dict = {GraphEnums.INF_EUROROAD: 39,
                       GraphEnums.CA_NETSCIENCE: 38,
                       GraphEnums.BIO_CELEGANS: 37}

        output_dir = '2hop-features'

        self.generate_plots(source_dict, output_dir)

    def generate_density_test(self):
        source_dict1 = {GraphEnums.SBM: 93,
                        GraphEnums.ER: 100,
                        GraphEnums.NWS_K7: 90,
                        }

        source_dict2 = {GraphEnums.NWS_K70: 97
                        }

        source_dict3 = {#GraphEnums.NWS_K70: 113,
                        GraphEnums.NWS_P_0_point_5: 114,
                        GraphEnums.SBM_INTP_5_PERCENT: 117,
                        GraphEnums.SBM_INTP_15_PERCENT: 120,
                        }

        source_dict4 = {GraphEnums.NWS_N_K7: 116,
                        GraphEnums.NWS_N_PROP_K: 115,

        }

        source_dicts = [source_dict1, source_dict2, source_dict3, source_dict4]

        output_dir = 'density-test'

        for source_dict in source_dicts:
            self.generate_plots(source_dict, output_dir)

    def generate_centrality_combinations(self):
        source_dict = {GraphEnums.INF_EUROROAD: 153,
                       GraphEnums.CA_NETSCIENCE: 152,
                       GraphEnums.BIO_CELEGANS: 154,
                       GraphEnums.VOLES: 155,
                       GraphEnums.SOCFB_BOWDOIN47: 156,
                       GraphEnums.MULTIMAGMA: 162,
        }

        output_dir = 'centrality-combinations'

        self.generate_plots(source_dict, output_dir)

    def generate_all_plots(self,
                           yaxis: str,
                           include_mu: bool,
                           include_top_3_combinations: bool,
                           include_2hop: bool,
                           include_density: bool,
                           include_centrality_comb: bool):
        """Run the script using the virtual environment's Python."""

        self.yaxis = yaxis

        if include_mu:
            self.generate_mu_test()

        if include_top_3_combinations:
            self.generate_top_tree_combinations()

        if include_2hop:
            self.generate_2hop_features()

        if include_density:
            self.generate_density_test()

        if include_centrality_comb:
            self.generate_centrality_combinations()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--twohop',
                        action='store_true',
                        help='Using this flag generates the twohop feature plots')

    parser.add_argument('--mu',
                        action='store_true',
                        help='Using this flag generates the mu-test plots')

    parser.add_argument('--top_3_combinations',
                        action='store_true',
                        help='Using this flag generates top-3-combinations of features')

    parser.add_argument('--density',
                        action='store_true',
                        help='Using this flag generates density plots')

    parser.add_argument('--centrality_combinations',
                        action='store_true',
                        help='Using this flag generates centrality combination plots')

    parser.add_argument('--yaxis',
                        choices=['acc', 'frob'],
                        default='acc')

    args = parser.parse_args()

    pg = PlotGenerator()

    pg.generate_all_plots(args.yaxis, args.mu, args.top_3_combinations, args.twohop, args.density, args.centrality_combinations)
