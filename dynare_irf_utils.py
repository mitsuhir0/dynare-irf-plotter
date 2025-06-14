"""Utility functions for loading, processing, and plotting Dynare IRF

(Impulse Response Function) data.
"""

import math
import pickle
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.io import loadmat
from scipy.io.matlab import mat_struct


def load(filename: str) -> dict[mat_struct]:
    """Load a .mat file and return the data."""
    return loadmat(filename, squeeze_me=True, struct_as_record=False)


def get_endo_names(M_: mat_struct, long: bool = False) -> list[str]:
    """Extract endogenous variable names from M_.

    If long is True, return long names; otherwise, return short names.
    """
    if long:
        if hasattr(M_, "endo_names_long"):
            return [str(name).strip() for name in np.atleast_1d(M_.endo_names_long)]
        msg = "Missing attribute: endo_names_long"
        raise AttributeError(msg)
    if hasattr(M_, "endo_names"):
        return [str(name).strip() for name in np.atleast_1d(M_.endo_names)]
    msg = "M_ does not have 'endo_names' attribute."
    raise AttributeError(msg)


def get_endo_names_all(M_: mat_struct) -> tuple[list[str], list[str]]:
    """Extract both short and long endogenous variable names from M_.

    Returns a tuple of two lists: (short_names, long_names).
    """
    short = get_endo_names(M_, long=False)
    long = get_endo_names(M_, long=True)
    return short, long


def get_exo_names(M_: mat_struct, long=False) -> list[str]:
    """Extract shock names from M_."""
    if long:
        if hasattr(M_, "exo_names_long"):
            return [str(name).strip() for name in np.atleast_1d(M_.exo_names_long)]
        msg = "M_ does not have 'exo_names_long' attribute."
        raise AttributeError(msg)
    if not hasattr(M_, "exo_names"):
        msg = "M_ does not have 'exo_names' attribute."
        raise AttributeError(msg)
    return [str(name).strip() for name in np.atleast_1d(M_.exo_names)]


def get_exo_names_all(M_: mat_struct) -> tuple[list[str], list[str]]:
    """Extract both short and long shock names from M_.

    Returns a tuple of two lists: (short_names, long_names).
    """
    short = get_exo_names(M_, long=False)
    long = get_exo_names(M_, long=True)
    return short, long


def get_irf_endo_vars(oo_: mat_struct, M_: mat_struct) -> dict[str, list[str]]:
    """Extract a list of endogenous variables used in IRFs for each shock.

    Parameters
    ----------
    oo_ : mat_struct
        The oo_ object from Dynare .mat file, containing IRF results.
    M_ : mat_struct
        The M_ object from Dynare .mat file, containing model variable names.

    Returns
    -------
    dict[str, list[str]]
        A dictionary mapping each exogenous shock name to a list of endogenous variables that have IRFs.

    """
    irfs = oo_.irfs

    # Get variable names and shock names
    endo_names = get_endo_names(M_, long=False)
    exo_names = get_exo_names(M_, long=False)

    # Convert IRF data to a dictionary
    irf_dict = {
        name: getattr(irfs, name)
        for name in dir(irfs)
        if not name.startswith("__") and isinstance(getattr(irfs, name), np.ndarray)
    }

    # Group IRFs by shock (names only, no data)
    used_vars_by_shock = defaultdict(list)
    for full_name in irf_dict:
        for shock in exo_names:
            if full_name.endswith(f"_{shock}"):
                var = full_name[: -(len(shock) + 1)]
                if var in endo_names:
                    used_vars_by_shock[shock].append(var)
                break

    # Remove duplicates and sort (optional)
    for shock in used_vars_by_shock:
        used_vars_by_shock[shock] = sorted(set(used_vars_by_shock[shock]))

    if not used_vars_by_shock:
        msg = "No IRF variable names found for the given shocks."
        raise ValueError(msg)

    return dict(used_vars_by_shock)


def get_irf(oo_: mat_struct, M_: mat_struct) -> dict[str, pd.DataFrame]:
    """Extract IRF data from the oo_ object using endo_names and exo_names from M_,

    and return a dictionary of DataFrames indexed by shock name.
    """
    irfs = oo_.irfs

    # Convert IRF data to a dictionary
    irf_dict = {
        name: getattr(irfs, name)
        for name in dir(irfs)
        if not name.startswith("__") and isinstance(getattr(irfs, name), np.ndarray)
    }

    # Get variable names used in IRFs (dependent function)
    used_vars_by_shock = get_irf_endo_vars(oo_, M_)

    # Group IRFs by shock
    shock_dfs = {}
    for shock, vars_for_shock in used_vars_by_shock.items():
        var_data = {
            var: irf_dict[f"{var}_{shock}"]
            for var in vars_for_shock
            if f"{var}_{shock}" in irf_dict
        }
        if var_data:
            shock_dfs[shock] = pd.DataFrame(var_data)

    if not shock_dfs:
        msg = "No IRF data found for the given shocks."
        raise ValueError(msg)

    return shock_dfs


def to_endo_name_long(short_name: str, M_: mat_struct) -> str:
    """Convert a short variable name to its long name using M_.endo_names and M_.endo_names_long."""
    # Extract name lists and convert to strings
    short, long = get_endo_names_all(M_)

    # Search without using a dictionary
    try:
        idx = short.index(short_name)
        return long[idx]
    except ValueError as err:
        msg = "Variable name not found in M_.endo_names."
        raise KeyError(msg) from err


def to_endo_name_short(long_name: str, M_: mat_struct) -> str:
    """Convert a long variable name to its short name

    using M_.endo_names and M_.endo_names_long.
    """
    # Extract name lists and convert to strings
    short, long = get_endo_names_all(M_)

    # Search without using a dictionary
    try:
        idx = long.index(long_name)
        return short[idx]
    except ValueError:
        msg = f"Variable name '{long_name}' was not found in M_.endo_names_long."
        raise KeyError(msg) from None


def to_exo_name_long(short_name: str, M_: mat_struct) -> str:
    """Convert a short shock name to its long name

    using M_.exo_names and M_.exo_names_long.
    """
    # Extract name lists and convert to strings
    short, long = get_exo_names_all(M_)

    try:
        idx = short.index(short_name)
        if short[idx] == long[idx]:
            return short_name
        return long[idx]

    except ValueError:
        msg = f"Shock name '{short_name}' was not found in M_.exo_names."
        raise KeyError(msg) from None


def to_exo_name_short(long_name: str, M_: mat_struct) -> str:
    """Convert a short shock name to its long name

    using M_.exo_names and M_.exo_names_long.
    """
    # Extract name lists and convert to strings
    short, long = get_exo_names_all(M_)

    try:
        idx = long.index(long_name)
        if short[idx] == long[idx]:
            return long_name
        return short[idx]

    except ValueError:
        msg = f"Shock name '{long_name}' was not found in M_.exo_names_long."
        raise KeyError(msg) from None


def convert(name: str, M_: mat_struct, vartype: str, length: str) -> str:
    """Convert endogenous or exogenous variable names to short or long names.

    Parameters
    ----------
    name : str
        The name of the variable to convert.
    M_ : mat_struct
        The MAT structure containing variable names.
    vartype : str
        The type of variable ('endo' for endogenous, 'exo' for exogenous).
    length : str
        The length of the variable name ('short' for short, 'long' for long).

    Returns
    -------
    str
        The converted variable name.

    """
    msg = "length must be 'short' or 'long'."
    if vartype == "endo":
        if length == "short":
            return to_endo_name_short(name, M_)
        if length == "long":
            return to_endo_name_long(name, M_)
        raise ValueError(msg)
    if vartype == "exo":
        if length == "short":
            return to_exo_name_short(name, M_)
        if length == "long":
            return to_exo_name_long(name, M_)
        raise ValueError(msg)
    msg = "vartype must be 'endo' for endogenous or 'exo' for exogenous."
    raise ValueError(msg)


def plot_irf_df(
    df: pd.DataFrame,
    endo_names: list[str],
    shock_name: str,
    n_cols: int = 3,
    M_: mat_struct = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    suptitle: str | None = None,
    irf_threshold: float = 1e-10,
) -> Figure:
    irf_df = df[endo_names]

    # For each column, if max(abs(series)) < irf_threshold, set all values to 0
    for col in irf_df.columns:
        arr = irf_df[col].to_numpy().copy()
        if np.nanmax(np.abs(arr)) < irf_threshold:
            arr[:] = 0
            irf_df[col] = arr

    n_series = irf_df.shape[1]  # Number of series (columns)
    n_rows = math.ceil(n_series / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    axes = np.array(axes).reshape(-1)  # Flatten axes for easier handling

    for i, col in enumerate(irf_df.columns):
        if M_ is not None:
            # If M_ is specified, convert to long name
            title = convert(col, M_, vartype="endo", length="long")
            if suptitle is None:
                # If suptitle is not specified, convert shock name to long name
                suptitle = convert(shock_name, M_, vartype="exo", length="long")
        else:
            title = col
        ax = axes[i]
        ax.plot(irf_df[col])
        ax.set_title(title)
        ax.axhline(0, color="gray", linestyle="--")
        ax.grid()
        ax.set_xlim(left=0)  # x軸の左端を0に固定

        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Impulse Responses to {suptitle}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def dump_figure(fig: Figure) -> bytes:
    """Serialize a matplotlib Figure object to bytes using pickle.

    Parameters
    ----------
    fig : Figure
        The matplotlib Figure object to serialize.

    Returns
    -------
    bytes
        The pickled bytes representation of the figure and related info.

    """
    info = {
        "figure": fig,
        "matplotlib_version": mpl.__version__,
        "pickle_protocol": pickle.HIGHEST_PROTOCOL,
    }
    return pickle.dumps(info, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    """Load sample data, extract IRFs, and plot results."""
    data = load("sample.mat")
    oo_ = data["oo_"]
    M_ = data["M_"]  # noqa: N806

    shock_dfs = get_irf(oo_, M_)
    # ✅ Check results (example: IRF for shock 'eu')

    df = shock_dfs["eps_u"]  # noqa: PD901

    plot_irf_df(df, df.columns, "eps_u", M_=M_, n_cols=2)
    plt.show()


if __name__ == "__main__":
    main()
