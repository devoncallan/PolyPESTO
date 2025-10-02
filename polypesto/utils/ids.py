from typing import List


def obs_id(name: str) -> str:
    """Generate a standard observation ID given a name."""
    return f"obs_{name}"


def param_id(i: int) -> str:
    """Generate a standard parameter ID given an index."""
    return f"p_{i:03d}"


def make_param_ids(n: int) -> List[str]:
    """Generate a list of sequential standard parameter IDs of given length."""
    return [param_id(i) for i in range(n)]


def cond_id(name: str) -> str:
    """Generate a standard condition ID given a name."""
    return f"c_{name}"


def make_cond_ids(n: int) -> List[str]:
    """Generate a list of sequential standard condition IDs of given length."""
    return [cond_id(f"{i:03d}") for i in range(n)]


def prob_id(i: int) -> str:
    """Generate a standard problem ID given an index."""
    return f"prob_{i:03d}"


def make_prob_ids(n: int) -> List[str]:
    """Generate a list of sequential standard problem IDs of given length."""
    return [prob_id(i) for i in range(n)]
