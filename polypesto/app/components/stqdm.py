import contextlib

import pypesto.engine.multi_process as _pp_engine
import pypesto.util as _pp_util
import streamlit as st
import tqdm.auto as _tqdm


@contextlib.contextmanager
def streamlit_tqdm(container: st.delta_generator.DeltaGenerator | None = None):
    """
    Redirect tqdm progress bars (including pypesto's) into a Streamlit container.

    We patch:
    - `tqdm.auto.tqdm` for any direct library use.
    - `pypesto.util._tqdm` and `pypesto.util.tqdm` used by pypesto.
    - `pypesto.engine.multi_process.tqdm` used inside the multistart engine.
    """

    c = container or st

    orig_std = _tqdm.tqdm
    orig_pp = _pp_util._tqdm
    orig_pp_fn = _pp_util.tqdm
    orig_engine_fn = _pp_engine.tqdm

    def st_tqdm(*args, **kwargs):
        bar = orig_pp(*args, **kwargs)
        progress = c.progress(0.0)
        status = c.empty()

        orig_update = bar.update

        def update(n=1):
            orig_update(n)
            if bar.total:
                progress.progress(bar.n / bar.total)
            status.text(bar.desc or "")

        bar.update = update

        orig_close = bar.close

        def close():
            progress.empty()
            status.empty()
            orig_close()

        bar.close = close
        return bar

    _tqdm.tqdm = st_tqdm  # generic
    _pp_util._tqdm = st_tqdm  # pypesto backend
    _pp_util.tqdm = st_tqdm  # pypesto wrapper
    _pp_engine.tqdm = st_tqdm  # engine-level import

    try:
        yield
    finally:
        _tqdm.tqdm = orig_std
        _pp_util._tqdm = orig_pp
        _pp_util.tqdm = orig_pp_fn
        _pp_engine.tqdm = orig_engine_fn
