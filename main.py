import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_fontja

from scipy.io import loadmat
from matplotlib.figure import Figure
from scipy.io.matlab import mat_struct
from collections import defaultdict


def load(filename) -> dict[mat_struct]:
    """
    Load a .mat file and return the data.
    """
    return loadmat(filename, squeeze_me=True, struct_as_record=False)


from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.io.matlab import mat_struct


def get_irf(oo_: mat_struct, M_: mat_struct) -> dict[str, pd.DataFrame]:
    """
    Extract IRF data from the oo_ object using endo_names and exo_names from M_.
    """
    irfs = oo_.irfs

    # 変数名とショック名を取得
    endo_names = [name.strip() for name in M_.endo_names.flatten()]
    exo_names = [name.strip() for name in M_.exo_names.flatten()]

    # IRFデータを辞書に変換
    irf_dict = {
        name: getattr(irfs, name)
        for name in dir(irfs)
        if not name.startswith('__') and isinstance(getattr(irfs, name), np.ndarray)
    }

    # IRFをショックごとにグループ化
    grouped_irfs = defaultdict(dict)
    for full_name, values in irf_dict.items():
        for shock in exo_names:
            if full_name.endswith(f'_{shock}'):
                var = full_name[:-(len(shock)+1)]
                if var in endo_names:
                    grouped_irfs[shock][var] = values
                break

    # DataFrame化（ショックごと）
    shock_dfs = {}
    for shock, var_dict in grouped_irfs.items():
        df = pd.DataFrame(var_dict)
        shock_dfs[shock] = df

    return shock_dfs


def get_endo_names_long(short_name: str, M_: mat_struct) -> str:
    """
    Convert a short variable name to its long name using M_.endo_names and M_.endo_names_long."""

    # 名前リストを取り出して文字列に変換
    endo_names = [str(name).strip() for name in np.atleast_1d(M_.endo_names)]
    endo_names_long = [str(name).strip() for name in np.atleast_1d(M_.endo_names_long)]

    # 辞書を使わずに検索
    try:
        idx = endo_names.index(short_name)
        return endo_names_long[idx]
    except ValueError:
        raise KeyError(f"変数名 '{short_name}' は M_.endo_names に見つかりませんでした。")


def get_endo_names_short(long_name: str, M_: mat_struct) -> str:
    """
    Convert a long variable name to its short name using M_.endo_names and M_.endo_names_long.
    """
    # 名前リストを取り出して文字列に変換
    endo_names = [str(name).strip() for name in np.atleast_1d(M_.endo_names)]
    endo_names_long = [str(name).strip() for name in np.atleast_1d(M_.endo_names_long)]

    # 辞書を使わずに検索
    try:
        idx = endo_names_long.index(long_name)
        return endo_names[idx]
    except ValueError:
        raise KeyError(f"変数名 '{long_name}' は M_.endo_names_long に見つかりませんでした。")



def get_shock_name_long(short_name: str, M_: mat_struct) -> str:
    """
    Convert a short shock name to its long name using M_.exo_names and M_.exo_names_long.
    """
    # 名前リストを取り出して文字列に変換
    exo_names = [str(name).strip() for name in np.atleast_1d(M_.exo_names)]
    exo_names_long = [str(name).strip() for name in np.atleast_1d(M_.exo_names_long)]

    try:
        idx = exo_names.index(short_name)
        if exo_names[idx] == exo_names_long[idx]:
            return short_name
        return exo_names_long[idx]

    except ValueError:
        raise KeyError(f"ショック名 '{short_name}' は M_.exo_names に見つかりませんでした。")



def plot_irf_df(df: pd.DataFrame, endo_names: list[str], shock_name: str, n_cols: int=3, M_: mat_struct = None) -> Figure:
    irf_df = df[endo_names]

    n_series = irf_df.shape[1]  # 系列数（列の数）
    n_rows = math.ceil(n_series / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    axes = np.array(axes).reshape(-1)  # 軸を1次元にして扱いやすくする

    for i, col in enumerate(irf_df.columns):
        if M_ is not None:
            # M_が指定されている場合、長い名前に変換
            title = get_endo_names_long(col, M_)
            suptitle = get_shock_name_long(shock_name, M_)
        else:
            title = col
        ax = axes[i]
        ax.plot(irf_df[col])
        ax.set_title(title)
        ax.axhline(0, color='gray', linestyle='--')
        ax.grid(True)

    # 余った subplot を非表示にする
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"IRFs to shock: {suptitle}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


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

