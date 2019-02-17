from artemis.general.iteratorize import Iteratorize


def parameter_search(objective, space, n_calls, n_random_starts=3, acq_optimizer="auto", n_jobs=4):
    """
    :param Callable[[Any], scalar] objective: The objective function that we're trying to optimize
    :param dict[str, Dimension] space:
    :param n_calls:
    :param n_random_starts:
    :param acq_optimizer:
    :return Generator[{'names': List[str], 'x_iters': List[]:
    """  # TODO: Finish building this
    from skopt import gp_minimize  # Soft requirements are imported in here.
    from skopt.utils import use_named_args

    for k, var in space.items():
        var.name=k
    space = list(space.values())

    objective = use_named_args(space)(objective)

    iter = Iteratorize(
        func = lambda callback: gp_minimize(objective,
            dimensions=space,
            n_calls=n_calls,
            n_random_starts = n_random_starts,
            random_state=1234,
            n_jobs=n_jobs,
            verbose=False,
            callback=callback,
            acq_optimizer = acq_optimizer,
            ),
    )

    for i, iter_info in enumerate(iter):
        yield iter_info
