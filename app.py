"""Streamlit app for visualizing Dynare IRFs from MATLAB .mat files."""

import io
from pathlib import Path

import pandas as pd
import scipy.io
import streamlit as st
from matplotlib.figure import Figure

import text
from dynare_irf_utils import (
    convert,
    dump_figure,
    get_irf,
    get_irf_endo_vars,
    load,
    plot_irf_df,
)

type Mat = scipy.io.matlab.mat_struct


def remove_suptitle(fig: Figure) -> Figure:
    """Remove the suptitle from a matplotlib figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object from which to remove the suptitle.

    Returns
    -------
        matplotlib.figure.Figure: The modified figure object.

    """
    # Remove suptitle using the public API
    fig.suptitle(None)
    fig.tight_layout()
    return fig


def get_mat_file(use_sample_file, uploaded_file) -> tuple:
    """MATファイルのパスまたはアップロードファイルを返す"""
    if use_sample_file:
        sample_file_path = "sample.mat"
        if Path(sample_file_path).exists():
            return sample_file_path, None
        st.error("Sample MAT file (sample.mat) not found.")
        return None, None
    if uploaded_file is not None:
        return uploaded_file, Path(uploaded_file.name).stem
    return None, None


def fix_dynare_typo(M_: Mat, use_sample_file: bool) -> Mat:
    """Dynareファイルのtypo修正"""
    if use_sample_file:
        target = "monetary policy shock"
        replacement = "cost push shock"
        M_.exo_names_long = [
            replacement if s == target else s for s in M_.exo_names_long
        ]
    return M_


def get_endo_names_long(oo_: Mat, M_: Mat) -> list[str]:
    """エンドジェナス変数名リストの取得"""
    endo_vars_shocks = get_irf_endo_vars(oo_, M_)
    endo_vars = endo_vars_shocks[next(iter(endo_vars_shocks.keys()))]
    endo_names_long = sorted(
        [convert(name, M_, vartype="endo", length="long") for name in endo_vars],
    )
    return endo_names_long


def convert_selected_endo_names(
    selected_endo_names_long: list[str],
    M_: Mat,
) -> list[str]:
    """選択された変数名のshort name変換"""
    return [
        convert(long_name, M_, vartype="endo", length="short")
        for long_name in selected_endo_names_long
    ]


def get_shock_lists(
    shock_dfs: dict[str, pd.DataFrame],
    M_: Mat,
) -> tuple[list[str], list[str]]:
    """ショックリストの取得と変換"""
    shock_list = list(shock_dfs.keys())
    long_shock_list = [
        convert(shock, M_, vartype="exo", length="long") for shock in shock_list
    ]
    return shock_list, long_shock_list


def plot_and_download_irf(
    irfs,
    selected_endo_names_short,
    shock_name,
    n_col,
    M_,
    xlabel,
    ylabel,
    perods,
    mat_file_name,
    fig_title,
    text,
    file_format,
):
    """IRFプロットとダウンロード処理"""
    fig = plot_irf_df(
        irfs[:perods],
        selected_endo_names_short,
        shock_name,
        n_cols=n_col,
        M_=M_,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    st.pyplot(fig)
    with st.expander("Display IRF Data"):
        st.write(irfs)
    base_file_name = f"{mat_file_name}_{shock_name}"
    fig_for_save = fig if fig_title else remove_suptitle(fig)
    if file_format == "pkl":
        pkl_bytes = dump_figure(fig_for_save)
        st.download_button(
            label="Download as pkl",
            data=pkl_bytes,
            file_name=f"{base_file_name}.pkl",
            mime="application/octet-stream",
        )
        with st.expander("About the PKL File"):
            st.markdown(text.about_pkl())
    else:
        buffer = io.BytesIO()
        fig_for_save.savefig(buffer, format=file_format)
        buffer.seek(0)
        mime_type = {
            "png": "image/png",
            "eps": "application/postscript",
            "pdf": "application/pdf",
            "svg": "image/svg+xml",
        }[file_format]
        st.download_button(
            label=f"Download as {file_format}",
            data=buffer,
            file_name=f"{base_file_name}.{file_format}",
            mime=mime_type,
        )


# --- メイン処理 ---
st.set_page_config(
    page_title="IRF Plotter",
    page_icon="📈",
)

st.title("IRF Plotter for MAT Files")
st.markdown(text.tool_description())
with st.expander("How to Use"):
    st.markdown(text.instructions())

use_sample_file = st.checkbox("Try the demo with a sample MAT file")
if use_sample_file:
    with st.expander("About the sample.mat file"):
        st.markdown(text.about_sample())

uploaded_file = None
if not use_sample_file:
    uploaded_file = st.file_uploader(
        "Upload a MAT file",
        type=["mat"],
        disabled=use_sample_file,
    )

mat_file_path, mat_file_name = get_mat_file(use_sample_file, uploaded_file)

if mat_file_path is not None:
    data = load(mat_file_path)
    oo_ = data.get("oo_", None)
    M_ = data.get("M_", None)
    M_ = fix_dynare_typo(M_, use_sample_file)

    if oo_ is None:
        st.error("The uploaded MAT file does not contain 'oo_' data.")
    else:
        endo_names_long = get_endo_names_long(oo_, M_)
        selected_endo_names_long = st.multiselect(
            "Select endogenous variables to plot:",
            options=endo_names_long,
            default=endo_names_long[:5],
        )
        if selected_endo_names_long:
            selected_endo_names_short = convert_selected_endo_names(
                selected_endo_names_long,
                M_,
            )
            shock_dfs = get_irf(oo_, M_)
            shock_list, long_shock_list = get_shock_lists(shock_dfs, M_)
            selected_shocks = st.selectbox(
                "Select shocks to plot:",
                options=long_shock_list,
                index=0,
            )
            with st.expander("Plot Options"):
                n_col = st.number_input(
                    "Number of columns for the plot layout:",
                    min_value=1,
                    max_value=5,
                    value=2,
                    step=1,
                )
                plot_xlabel = st.text_input("X-axis label:", value="Time")
                plot_ylabel = st.text_input("Y-axis label:", value="Response")
                perods = st.number_input(
                    "Number of periods to plot:",
                    min_value=1,
                    max_value=100,
                    value=len(shock_dfs[shock_list[0]]),
                    step=1,
                )
            if selected_shocks:
                shock_name = convert(
                    selected_shocks,
                    M_,
                    vartype="exo",
                    length="short",
                )
                irfs = shock_dfs[shock_name]
                st.subheader(f"Shock: {selected_shocks}")
                include_title = st.checkbox(
                    "Include figure title in the exported file",
                    value=True,
                )
                file_format = st.selectbox(
                    "Select file format to download:",
                    options=["png", "pdf", "eps", "svg", "pkl"],
                )
                if not mat_file_name:
                    mat_file_name = "sample"
                plot_and_download_irf(
                    irfs,
                    selected_endo_names_short,
                    shock_name,
                    n_col,
                    M_,
                    plot_xlabel,
                    plot_ylabel,
                    perods,
                    mat_file_name,
                    include_title,
                    text,
                    file_format,
                )
            else:
                st.warning("Please select at least one shock to plot.")
        else:
            st.warning("Please select at least one endogenous variable to plot.")

st.markdown("---")
st.markdown(text.desclaimer())
st.markdown("---")
st.markdown(text.copyright())
