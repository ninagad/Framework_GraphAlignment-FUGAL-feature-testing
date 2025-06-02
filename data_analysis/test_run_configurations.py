# Local imports
from data_analysis.utils import (get_graph_names_from_file,
                                 strip_graph_name,
                                 get_training_graph_names,
                                 get_config_file,
                                 get_algo_args)


def test_graphs_are_training_graphs(runs: list[int]):
    names = get_graph_names_from_file(runs)
    names = [strip_graph_name(name) for name in names]

    expected_names = get_training_graph_names()

    # Check lists contain the same elements regardless of the order.
    assert sorted(names) == expected_names, f'Current graphs are: {names} from runs {runs}. Expected {expected_names}'

def test_run_has_6_noise_levels(run: int):
    config = get_config_file(run)

    noises = config['noises']

    expected_noises = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]

    assert noises == expected_noises, f'Current noise levels are {noises}. Expected {expected_noises}'


def test_run_have_expected_mu(run: int, expected_mu: float):
    args: list[dict] = get_algo_args(run)

    for args_dict in args:
        current_mu = args_dict['mu']

        assert current_mu == expected_mu, f'Current mu is {current_mu} in run {run}. Expected mu {expected_mu}'


def test_run_have_expected_nu_and_mu(run: int, expected_nu: float, expected_mu: float):
    args: list[dict] = get_algo_args(run)

    for args_dict in args:
        current_nu = args_dict['nu']
        current_mu = args_dict['mu']

        assert current_nu == expected_nu, f'Current nu is {current_nu} in run {run}. Expected nu {expected_nu}'
        assert current_mu == expected_mu, f'Current mu is {current_mu} in run {run}. Expected mu {expected_mu}'


def test_run_have_expected_frank_wolfe_iters(run: int, expected_fw: int):
    args = get_algo_args(run)

    for args_dict in args:
        current_fw = args_dict['frank_wolfe_iters']

        assert current_fw == expected_fw, (f'Current number of frank wolfe iters is {current_fw} in run {run}. '
                                           f'Expected fw iters is {expected_fw}')


def test_run_have_expected_sinkhorn_reg(run: int, expected_sinkhorn_reg: float):
    args = get_algo_args(run)

    for args_dict in args:
        current_sinkhorn = args_dict['sinkhorn_reg']

        assert current_sinkhorn == expected_sinkhorn_reg, (f'Current sinkhorn reg is {current_sinkhorn} in run {run}. '
                                                           f'Expected sinkhorn reg is {expected_sinkhorn_reg}')


def test_run_has_5_iterations(run: int):
    config = get_config_file(run)

    iters = config['iters']

    assert iters == 5, f'Got {iters} iterations in run {run}. Expected 5'


def test_runs_have_ordered_analysis_graphs(grouped_runs: dict):
    training_graphs = get_training_graph_names()

    for runs in grouped_runs.values():
        graph_names = get_graph_names_from_file(runs)
        stripped_names = [strip_graph_name(name) for name in graph_names]

        assert stripped_names == training_graphs, f'Current graphs has order {stripped_names} in runs {runs}. Expected graphs to have order {training_graphs}.'


def test_loaded_graphs(runs: list[int], expected: list[int]):
    for run, expected in zip(runs, expected):
        config = get_config_file(run)
        load_val = list(set(config['load']))
        if len(load_val) != 1:
            raise NotImplementedError

        assert load_val[0] == expected, f'Expected {expected}. Got {load_val}'


def test_configuration_graph_iters_nu_mu_sinkhorn(runs: dict, nu: float, mu: float, sinkhorn_reg: float):
    for fw, runs in runs.items():
        test_graphs_are_training_graphs(runs)

        for run in runs:
            test_run_has_5_iterations(run)
            test_run_has_6_noise_levels(run)
            test_run_have_expected_nu_and_mu(run, nu, mu)
            test_run_have_expected_sinkhorn_reg(run, sinkhorn_reg)


def test_graph_set_are_equal(run_1: int, run_2: int):
    graph1, graph2 = get_graph_names_from_file([run_1, run_2])

    assert graph1 == graph2, f'Graphs are not the same. Graph 1: {graph1}, Graph 2: {graph2}'
