# ruff: noqa: N806, N815 # Upper case

"""Use the sweep_data to set config parameters."""

from engibench.problems.power_electronics.utils.config import Config


def process_sweep_data(config: Config, sweep_data: list[float]) -> Config:
    """Use the sweep_data to set config parameters.

    The following parameters are set:
    config.capacitor_val, config.inductor_val, config.switch_T1, config.switch_T2, config.switch_L1, config.switch_L2.
    """
    config.capacitor_val = sweep_data[: config.n_C]  # 6 capacitors
    config.inductor_val = sweep_data[config.n_C : config.n_C + config.n_L]  # 3 inductors
    config.switch_T1 = [sweep_data[config.n_C + config.n_L]] * config.n_S  # 5 switches (T1)
    config.switch_T2 = [1.0] * config.n_S  # 5 switches (T2), all set to 1.0 for now
    config.switch_L1 = sweep_data[
        config.n_C + config.n_L + 1 : config.n_C + config.n_L + 1 + config.n_S
    ]  # 5 switches (L1), binary values (0 or 1)
    config.switch_L2 = sweep_data[
        config.n_C + config.n_L + 1 + config.n_S : config.n_C + config.n_L + 1 + config.n_S * 2
    ]  # 5 switches (L2), binary values (0 or 1)

    return config
