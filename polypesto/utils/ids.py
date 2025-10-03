from typing import List, TypeAlias, Tuple
import hashlib


class ID:
    """Namespace for ID generation functions."""

    StrObsID: TypeAlias = str
    StrObsName: TypeAlias = str
    StrObsFormula: TypeAlias = str

    StrCondID: TypeAlias = str
    StrCondName: TypeAlias = str

    ObsCondKey: TypeAlias = Tuple[StrObsID, StrCondID]

    @staticmethod
    def get_hash(s: str) -> str:
        return hashlib.md5(s.encode(), usedforsecurity=False).hexdigest()

    @staticmethod
    def obs_id(name: StrObsID) -> str:
        """Generate a standard observable ID given a name."""
        return f"obs_{name}"

    @staticmethod
    def param_id(i: int) -> str:
        """Generate a standard parameter ID given an index."""
        return f"p_{i:03d}"

    @staticmethod
    def cond_id(name: StrCondName) -> str:
        """Generate a standard condition ID given a name."""
        return f"c_{name}"

    @staticmethod
    def prob_id(i: int) -> str:
        """Generate a standard problem ID given an index."""
        return f"prob_{i:03d}"

    @staticmethod
    def make_param_ids(n: int) -> List[str]:
        """Generate a list of sequential standard parameter IDs of given length."""
        return [ID.param_id(i) for i in range(n)]

    @staticmethod
    def make_cond_ids(n: int) -> List[str]:
        """Generate a list of sequential standard condition IDs of given length."""
        return [ID.cond_id(f"{i:03d}") for i in range(n)]

    @staticmethod
    def make_prob_ids(n: int) -> List[str]:
        """Generate a list of sequential standard problem IDs of given length."""
        return [ID.prob_id(i) for i in range(n)]
