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
    NWS = auto()
    SBM = auto()


def generate_all_plots(include_mu: bool, include_combinations: bool, include_2hop: bool):
    """Run the script using the virtual environment's Python."""
    venv_python = activate_venv()

    baselines = {GraphEnums.INF_EUROROAD: 20,
                 GraphEnums.CA_NETSCIENCE: 19,
                 GraphEnums.BIO_CELEGANS: 36,
                 GraphEnums.SOCFB_BOWDOIN47: 44,
                 GraphEnums.VOLES: 60,
                 GraphEnums.MULTIMAGMA: 59,
                 GraphEnums.NWS: None,  # TODO
                 GraphEnums.SBM: None,  # TODO
                 }

    if include_mu:
        generate_mu_test(venv_python, baselines)

    if include_combinations:
        generate_top_tree_combinations(venv_python, baselines)

    if include_2hop:
        generate_2hop_features(venv_python, baselines)


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


def generate_plots(venv_python, baseline_dict: dict, source_dict: dict, outputdir: str):
    for graph, source in source_dict.items():
        baseline = baseline_dict[graph]

        print(60*'-')
        print(f'Running with baseline: {baseline}, source: {source}, outputdir: {outputdir}')
        print(60*'-')
        subprocess.run(
            [venv_python, "plot.py", "--baseline", str(baseline), "--source", str(source), "--outputdir", outputdir])


def generate_top_tree_combinations(venv_python, baselines):
    top_three_comb_sources = {GraphEnums.INF_EUROROAD: 73,
                              GraphEnums.CA_NETSCIENCE: 71,
                              GraphEnums.SOCFB_BOWDOIN47: 74,
                              GraphEnums.VOLES: 72,
                              }
    outputdir = 'top-3-combinations'

    # Generate top-3-combinations plots
    generate_plots(venv_python, baselines, top_three_comb_sources, outputdir)


def generate_mu_test(venv_python, baselines):
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

    outputdir = 'mu-test'

    for source_dict in source_dicts:
        generate_plots(venv_python, baselines, source_dict, outputdir)


def generate_2hop_features(venv_ptyhon, baselines):
    source_dict = {GraphEnums.INF_EUROROAD: 39,
                   GraphEnums.CA_NETSCIENCE: 38,
                   GraphEnums.BIO_CELEGANS: 37}

    outputdir = '2hop-features'

    generate_plots(venv_ptyhon, baselines, source_dict, outputdir)


def generate_density_test(venv_python, baselines):
    source_dict = {GraphEnums.NWS: -1,
                   GraphEnums.SBM: -1}
    # TODO: get correct indices and finish implementation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--twohop',
                        action='store_true',
                        help='Using this flag generates the twohop feature plots')

    parser.add_argument('--mu',
                        action='store_true',
                        help='Using this flag generates the mu-test plots')

    parser.add_argument('--combinations',
                        action='store_true',
                        help='Using this flag generates top-3-combinations of features')

    args = parser.parse_args()

    generate_all_plots(args.mu, args.combinations, args.twohop)
