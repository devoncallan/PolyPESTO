from typing import List, Dict, Optional

from polypesto.utils import ID


def parse_obs_noise(
    obs_noise: float | List[float] | Dict[ID.StrObsName, float] | None,
    obs_names: List[ID.StrObsName],
) -> Dict[ID.StrObsName, float] | None:

    n_obs = len(obs_names)
    if obs_noise is None:
        return None
    elif isinstance(obs_noise, (int, float)):
        return {obs_name: float(obs_noise) for obs_name in obs_names}
    elif isinstance(obs_noise, list):
        if len(obs_noise) != n_obs:
            raise ValueError(
                "Length of obs_noise list must match number of observables."
            )
        return {
            obs_name: float(noise)
            for obs_name, noise in zip(obs_names, obs_noise, strict=True)
        }
    elif isinstance(obs_noise, dict):
        missing_obs = set(obs_names) - set(obs_noise.keys())
        if missing_obs:
            raise ValueError(
                f"obs_noise dict is missing entries for observables: {missing_obs}"
            )
        return {obs_name: float(obs_noise[obs_name]) for obs_name in obs_names}
    else:
        raise TypeError("obs_noise must be None, a float, a list of floats, or a dict")
