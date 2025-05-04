import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_fontja
import matplotlib
import pickle


from scipy.io import loadmat
from matplotlib.figure import Figure
from scipy.io.matlab import mat_struct
from collections import defaultdict


def load(filename) -> dict[mat_struct]:
    """
    Load a .mat file and return the data.
    """
    return loadmat(filename, squeeze_me=True, struct_as_record=False)


def get_endo_names(M_: mat_struct, long=False) -> list[str]:
    """
    Extract endogenous variable names from M_.
    If long is True, return long names; otherwise, return short names.
    """
    if long:
        if hasattr(M_, 'endo_names_long'):
            return [str(name).strip() for name in np.atleast_1d(M_.endo_names_long)]
        else:
            raise AttributeError("M_ does not have 'endo_names_long' attribute.")
    else:
        if hasattr(M_, 'endo_names'):
            return [str(name).strip() for name in np.atleast_1d(M_.endo_names)]
        else:
            raise AttributeError("M_ does not have 'endo_names' attribute.")


def get_endo_names_all(M_: mat_struct) -> tuple[list[str], list[str]]:
    """
    Extract both short and long endogenous variable names from M_.
    Returns a tuple of two lists: (short_names, long_names).
    """
    short = get_endo_names(M_, long=False)
    long = get_endo_names(M_, long=True)
    return short, long


def get_exo_names(M_: mat_struct, long=False) -> list[str]:
    """
    Extract shock names from M_.
    """
    if long:
        if hasattr(M_, 'exo_names_long'):
            return [str(name).strip() for name in np.atleast_1d(M_.exo_names_long)]
        else:
            raise AttributeError("M_ does not have 'exo_names_long' attribute.")
    else:
        if not hasattr(M_, 'exo_names'):
            raise AttributeError("M_ does not have 'exo_names' attribute.")
        else:
            return [str(name).strip() for name in np.atleast_1d(M_.exo_names)]


def get_exo_names_all(M_: mat_struct) -> tuple[list[str], list[str]]:
    """
    Extract both short and long shock names from M_.
    Returns a tuple of two lists: (short_names, long_names).
    """
    short = get_exo_names(M_, long=False)
    long = get_exo_names(M_, long=True)
    return short, long



def get_irf_endo_vars(oo_: mat_struct, M_: mat_struct) -> dict[str, list[str]]:
    """
    Extract a list of endogenous variables used in IRFs for each shock.

    Parameters:
        oo_ (mat_struct): The oo_ object from Dynare .mat file.
        M_ (mat_struct): The M_ object from Dynare .mat file.

    Returns:
        dict[str, list[str]]: A dictionary mapping each exogenous shock name
                              to a list of endogenous variables that have IRFs.
    """
    irfs = oo_.irfs

    # 変数名とショック名を取得
    endo_names = get_endo_names(M_, long=False)
    exo_names = get_exo_names(M_, long=False)

    # IRFデータを辞書に変換
    irf_dict = {
        name: getattr(irfs, name)
        for name in dir(irfs)
        if not name.startswith('__') and isinstance(getattr(irfs, name), np.ndarray)
    }

    # IRFをショックごとにグループ化（データ不要、名前のみ）
    used_vars_by_shock = defaultdict(list)
    for full_name in irf_dict:
        for shock in exo_names:
            if full_name.endswith(f'_{shock}'):
                var = full_name[:-(len(shock)+1)]
                if var in endo_names:
                    used_vars_by_shock[shock].append(var)
                break

    # 重複排除とソート（好みに応じて）
    for shock in used_vars_by_shock:
        used_vars_by_shock[shock] = sorted(set(used_vars_by_shock[shock]))

    if not used_vars_by_shock:
        raise ValueError("No IRF variable names found for the given shocks.")

    return dict(used_vars_by_shock)


def get_irf(oo_: mat_struct, M_: mat_struct) -> dict[str, pd.DataFrame]:
    """
    Extract IRF data from the oo_ object using endo_names and exo_names from M_,
    and return a dictionary of DataFrames indexed by shock name.
    """
    irfs = oo_.irfs
    endo_names = get_endo_names(M_, long=False)
    exo_names = get_exo_names(M_, long=False)

    # IRFデータを辞書に変換
    irf_dict = {
        name: getattr(irfs, name)
        for name in dir(irfs)
        if not name.startswith('__') and isinstance(getattr(irfs, name), np.ndarray)
    }

    # IRFに使用された変数名だけ取得（依存関数）
    used_vars_by_shock = get_irf_endo_vars(oo_, M_)

    # IRFをショックごとにグループ化
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
        raise ValueError("No IRF data found for the given shocks.")

    return shock_dfs


def to_endo_name_long(short_name: str, M_: mat_struct) -> str:
    """
    Convert a short variable name to its long name using M_.endo_names and M_.endo_names_long."""

    # 名前リストを取り出して文字列に変換
    short, long = get_endo_names_all(M_)

    # 辞書を使わずに検索
    try:
        idx = short.index(short_name)
        return long[idx]
    except ValueError:
        raise KeyError(f"変数名 '{short_name}' は M_.endo_names に見つかりませんでした。")


def to_endo_name_short(long_name: str, M_: mat_struct) -> str:
    """
    Convert a long variable name to its short name using M_.endo_names and M_.endo_names_long.
    """
    # 名前リストを取り出して文字列に変換
    short, long = get_endo_names_all(M_)

    # 辞書を使わずに検索
    try:
        idx = long.index(long_name)
        return short[idx]
    except ValueError:
        raise KeyError(f"変数名 '{long_name}' は M_.endo_names_long に見つかりませんでした。")



def to_exo_name_long(short_name: str, M_: mat_struct) -> str:
    """
    Convert a short shock name to its long name using M_.exo_names and M_.exo_names_long.
    """
    # 名前リストを取り出して文字列に変換
    short, long = get_exo_names_all(M_)

    try:
        idx = short.index(short_name)
        if short[idx] == long[idx]:
            return short_name
        return long[idx]

    except ValueError:
        raise KeyError(f"ショック名 '{short_name}' は M_.exo_names に見つかりませんでした。")


def to_exo_name_short(long_name: str, M_: mat_struct) -> str:
    """
    Convert a short shock name to its long name using M_.exo_names and M_.exo_names_long.
    """
    # 名前リストを取り出して文字列に変換
    short, long = get_exo_names_all(M_)

    try:
        idx = long.index(long_name)
        if short[idx] == long[idx]:
            return long_name
        return short[idx]

    except ValueError:
        raise KeyError(f"ショック名 '{long_name}' は M_.exo_names_long に見つかりませんでした。")



def convert(name: str, M_: mat_struct, vartype: str, length: str) -> str:
    """内生変数または外生変数をshortnameまたはlongnameに変換する関数

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
    if vartype == 'endo':
        if length == 'short':
            return to_endo_name_short(name, M_)
        elif length == 'long':
            return to_endo_name_long(name, M_)
    elif vartype == 'exo':
        if length == 'short':
            return to_exo_name_short(name, M_)
        elif length == 'long':
            return to_exo_name_long(name, M_)
    else:
        raise ValueError("vartype must be 'endo' for endogenous or 'exo' for exogenous.")


def plot_irf_df(df: pd.DataFrame, endo_names: list[str], shock_name: str, n_cols: int=3, M_: mat_struct = None, xlabel: str = None, ylabel: str = None, suptitle: str = None) -> Figure:
    irf_df = df[endo_names]

    n_series = irf_df.shape[1]  # 系列数（列の数）
    n_rows = math.ceil(n_series / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    axes = np.array(axes).reshape(-1)  # 軸を1次元にして扱いやすくする

    for i, col in enumerate(irf_df.columns):
        if M_ is not None:
            # M_が指定されている場合、長い名前に変換
            title = convert(col, M_, vartype='endo', length='long')
            if suptitle is None:
                # suptitleが指定されていない場合、ショック名を長い名前に変換
                suptitle = convert(shock_name, M_, vartype='exo', length='long')
        else:
            title = col
        ax = axes[i]
        ax.plot(irf_df[col])
        ax.set_title(title)
        ax.axhline(0, color='gray', linestyle='--')
        ax.grid(True)

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        
        if ylabel is not None:
            ax.set_ylabel(ylabel)

    # 余った subplot を非表示にする
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"IRFs to shock: {suptitle}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def dump_figure(fig: Figure) -> bytes:
    info = {
        'figure': fig,
        'matplotlib_version': matplotlib.__version__,
        'pickle_protocol': pickle.HIGHEST_PROTOCOL
    }
    return pickle.dumps(info, protocol=pickle.HIGHEST_PROTOCOL)


def main():

    data = load('result.mat')
    oo_ = data['oo_']
    M_ = data['M_']

    shock_dfs = get_irf(oo_, M_)
    # ✅ 結果の確認（例：euショックのIRF）

    df = shock_dfs['shock_a1']

    plot_irf_df(df, ['a1', 'a2', 'e', 'pi1', 'pi2', 'y1'], 'shock_a1', M_=M_)
    plt.show()


if __name__ == "__main__":
    main()

