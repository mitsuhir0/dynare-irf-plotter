"""Streamlit app for visualizing Dynare IRFs from MATLAB .mat files."""

import base64
import io
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import streamlit as st
import yaml
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


def get_mat_file(use_sample_file: bool, uploaded_file: object) -> tuple:
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


def fix_dynare_typo(M_: Mat) -> Mat:
    """Dynareファイルのtypo修正"""
    target = "monetary policy shock"
    replacement = "cost push shock"
    M_.exo_names_long = [replacement if s == target else s for s in M_.exo_names_long]
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


plot_threshold = 1e-10


def plot_and_download_irf(
    irf_data_list: list,
    var_names: list,
    shock_names: list,
    style_options: list,
    legend_labels: list,
    legend_panel_mode: int,
    show_grid: bool,
    fig_width: float,
    fig_height: float,
    download_yaml: bool,
    save_options: dict,
) -> None:
    """IRFプロットとダウンロード処理"""
    fig = plot_irf_df(
        irf_data_list,
        var_names,
        shock_names,
        n_cols=style_options["n_col"],
        M_=style_options["M_"],
        xlabel=style_options["xlabel"],
        ylabel=style_options["ylabel"],
        irf_threshold=plot_threshold,
    )
    st.pyplot(fig)
    with st.expander("Display IRF Data"):
        st.write(irf_data_list)
    base_file_name = f"{save_options['mat_file_name']}_{shock_names}"
    fig_for_save = fig if save_options["fig_title"] else remove_suptitle(fig)
    if save_options["file_format"] == "pkl":
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
        fig_for_save.savefig(buffer, format=save_options["file_format"])
        buffer.seek(0)
        mime_type = {
            "png": "image/png",
            "eps": "application/postscript",
            "pdf": "application/pdf",
            "svg": "image/svg+xml",
        }[save_options["file_format"]]
        st.download_button(
            label=f"Download as {save_options['file_format']}",
            data=buffer,
            file_name=f"{base_file_name}.{save_options['file_format']}",
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

# --- MAT file upload UI ---
use_sample_file = st.checkbox("Try the demo with a sample MAT file")
if use_sample_file:
    with st.expander("About the sample.mat file"):
        st.markdown(text.about_sample())

uploaded_files = []
if not use_sample_file:
    uploaded_files = st.file_uploader(
        "Upload one or more MAT files",
        type=["mat"],
        disabled=use_sample_file,
        accept_multiple_files=True,
    )

mat_file_paths = []
mat_file_names = []
if use_sample_file:
    mat_file_path, mat_file_name = get_mat_file(use_sample_file, None)
    if mat_file_path:
        mat_file_paths = [mat_file_path]
        mat_file_names = [mat_file_name or "sample"]
elif uploaded_files:
    for f in uploaded_files:
        path, name = get_mat_file(use_sample_file, f)
        if path:
            mat_file_paths.append(path)
            mat_file_names.append(name or Path(f.name).stem)

if mat_file_paths:
    datas = [load(path) for path in mat_file_paths]
    oo_list = [d.get("oo_", None) for d in datas]
    M_list = [d.get("M_", None) for d in datas]
    if use_sample_file:
        M_list = [fix_dynare_typo(M_) for M_ in M_list]

    if any(oo is None for oo in oo_list):
        st.error("At least one MAT file does not contain 'oo_' data.")
    else:
        # Variable and shock selection
        endo_names_long_sets = [
            set(get_endo_names_long(oo, M))
            for oo, M in zip(oo_list, M_list, strict=False)
        ]
        common_endo_names_long = sorted(set.intersection(*endo_names_long_sets))
        selected_endo_names_long = st.multiselect(
            "Select endogenous variables to plot:",
            options=common_endo_names_long,
            default=st.session_state.get(
                "selected_endo_names_long",
                common_endo_names_long[:],
            ),
        )
        if selected_endo_names_long:
            selected_endo_names_short_list = [
                convert_selected_endo_names(selected_endo_names_long, M) for M in M_list
            ]
            shock_dfs_list = [
                get_irf(oo, M) for oo, M in zip(oo_list, M_list, strict=False)
            ]
            shock_list_sets = [set(dfs.keys()) for dfs in shock_dfs_list]
            common_shocks = sorted(set.intersection(*shock_list_sets))
            long_shock_list = [
                convert(shock, M_list[0], vartype="exo", length="long")
                for shock in common_shocks
            ]
            selected_shock_long = st.selectbox(
                "Select shocks to plot:",
                options=long_shock_list,
                index=(
                    long_shock_list.index(st.session_state["selected_shock_long"])
                    if "selected_shock_long" in st.session_state
                    and st.session_state["selected_shock_long"] in long_shock_list
                    else 0
                ),
            )
            with st.expander("Plot Options"):
                n_col = st.number_input(
                    "Number of columns for the plot layout:",
                    min_value=1,
                    max_value=5,
                    value=st.session_state.get("n_col", 2),
                    step=1,
                )
                plot_xlabel = st.text_input(
                    "X-axis label:",
                    value=st.session_state.get("plot_xlabel", "Time"),
                )
                plot_ylabel = st.text_input(
                    "Y-axis label:",
                    value=st.session_state.get("plot_ylabel", "Response"),
                )
                periods = st.number_input(
                    "Number of periods to plot:",
                    min_value=1,
                    max_value=100,
                    value=st.session_state.get(
                        "periods",
                        len(shock_dfs_list[0][common_shocks[0]]),
                    ),
                    step=1,
                )
            # --- Per-MAT file plot style options ---
            marker_choices = ["o", "s", "^", "D", "v", "x", "*", "+", ".", "None"]
            linestyle_choices = ["-", "--", "-.", ":"]
            color_choices = [
                "blue",
                "orange",
                "green",
                "red",
                "purple",
                "brown",
                "pink",
                "gray",
                "olive",
                "cyan",
            ]
            file_plot_options = []
            st.markdown("### Plot Style for Each MAT File")
            loaded_file_opts = st.session_state.get(
                "file_plot_options",
                [{} for _ in mat_file_names],
            )
            if isinstance(loaded_file_opts, tuple):
                loaded_file_opts = loaded_file_opts[0]
            for idx, name in enumerate(mat_file_names):
                with st.expander(f"Style for {name}"):
                    marker = st.selectbox(
                        f"Marker for {name}",
                        marker_choices,
                        index=(
                            marker_choices.index(loaded_file_opts[idx]["marker"])
                            if idx < len(loaded_file_opts)
                            and loaded_file_opts[idx].get("marker") in marker_choices
                            else idx % len(marker_choices)
                        ),
                        key=f"marker_{idx}",
                    )
                    linestyle = st.selectbox(
                        f"Line style for {name}",
                        linestyle_choices,
                        index=(
                            linestyle_choices.index(loaded_file_opts[idx]["linestyle"])
                            if idx < len(loaded_file_opts)
                            and loaded_file_opts[idx].get("linestyle")
                            in linestyle_choices
                            else 0
                        ),
                        key=f"linestyle_{idx}",
                    )
                    color = st.selectbox(
                        f"Color for {name}",
                        color_choices,
                        index=(
                            color_choices.index(loaded_file_opts[idx]["color"])
                            if idx < len(loaded_file_opts)
                            and loaded_file_opts[idx].get("color") in color_choices
                            else idx % len(color_choices)
                        ),
                        key=f"color_{idx}",
                    )
                    legend_label = st.text_input(
                        f"Legend label for {name}",
                        value=(
                            loaded_file_opts[idx]["legend_label"]
                            if idx < len(loaded_file_opts)
                            and loaded_file_opts[idx].get("legend_label")
                            else name
                        ),
                        key=f"legend_{idx}",
                    )
                    file_plot_options.append(
                        {
                            "marker": marker if marker != "None" else None,
                            "linestyle": linestyle,
                            "color": color,
                            "legend_label": legend_label,
                        },
                    )
            show_legend = st.checkbox(
                "Show legend",
                value=st.session_state.get("show_legend", True),
            )
            legend_panel_mode = st.selectbox(
                "Legend display mode:",
                options=[0, 1],
                index=st.session_state.get("legend_panel_mode", 0),
                format_func=lambda x: "all panels" if x == 0 else "first panel only",
            )
            n_vars = len(selected_endo_names_long)
            n_rows = math.ceil(n_vars / n_col)
            show_grid = st.checkbox(
                "Show grid",
                value=st.session_state.get("show_grid", False),
            )
            fig_width = st.number_input(
                "Figure width (inches)",
                min_value=4,
                max_value=24,
                value=st.session_state.get("fig_width", 5 * n_col),
                step=1,
            )
            fig_height = st.number_input(
                "Figure height (inches)",
                min_value=3,
                max_value=20,
                value=st.session_state.get("fig_height", 3 * n_rows),
                step=1,
            )
            # --- Save/Load Plot Options (YAML) UI ---
            st.markdown("#### Save/Load Plot Options (YAML)")
            # --- Load (YAML upload) UI ---
            load_yaml_file = st.file_uploader(
                "Upload a YAML file to load plot/UI options",
                type=["yaml", "yml"],
                key="yaml_upload",
            )
            yaml_load_error = False
            rerun_key = "yaml_rerun_done"
            rerun_flag = "yaml_needs_rerun"
            if load_yaml_file is not None:
                try:
                    loaded = yaml.safe_load(load_yaml_file)
                    legend_mode = loaded.get("legend_panel_mode", 0)
                    if not isinstance(legend_mode, int) or legend_mode not in (0, 1):
                        legend_mode = 0
                    for k, v in loaded.items():
                        if k in ("endo_names_long", "shock_names"):
                            continue
                        if k == "legend_panel_mode":
                            st.session_state[k] = legend_mode
                        else:
                            st.session_state[k] = v
                    st.session_state[rerun_flag] = True
                except yaml.YAMLError as e:
                    yaml_load_error = True
                    st.error(
                        f"YAML file is invalid and was not loaded. Reason: {e}",
                    )
                    st.info("Please select a valid YAML file.")
                    st.session_state[rerun_key] = False  # Reset flag on error
                    st.session_state[rerun_flag] = False
            else:
                st.session_state[rerun_key] = False
                st.session_state[rerun_flag] = False

            # Show redraw button after YAML load
            if st.session_state.get(rerun_flag, False) and not yaml_load_error:
                st.warning(
                    "To apply the loaded YAML settings, "
                    "please click the redraw button below.",
                )
                if st.button("Redraw"):
                    st.session_state[rerun_flag] = False
                    st.rerun()
            # --- Save (YAML download) UI ---
            st.markdown("#### Save Plot Options (YAML Download)")
            plot_options = {}
            save_vars = [
                # "selected_endo_names_long",  # do not save
                # "selected_shock_long",      # do not save
                "n_col",
                "plot_xlabel",
                "plot_ylabel",
                "periods",
                "file_plot_options",
                "show_legend",
                "legend_panel_mode",
                "show_grid",
                "fig_width",
                "fig_height",
            ]
            for var in save_vars:
                # Exclude endo/shock names and selected_endo_names_long,
                # selected_shock_long from YAML save/load
                if var in (
                    "endo_names_long",
                    "shock_names",
                    "common_endo_names_long",
                    "common_shocks",
                    "long_shock_list",
                    "selected_endo_names_long",
                    "selected_shock_long",
                ):
                    continue
                if var == "legend_panel_mode":
                    val = 0 if legend_panel_mode in (0,) else 1
                    plot_options[var] = val
                elif var in locals():
                    plot_options[var] = locals()[var]
            yaml_str = yaml.dump(plot_options, allow_unicode=True, sort_keys=False)
            b64 = base64.b64encode(yaml_str.encode()).decode()
            download_link = (
                f'<a href="data:text/yaml;base64,{b64}" '
                'download="irf_plot_options.yaml">Download YAML</a>'
            )
            st.markdown(download_link, unsafe_allow_html=True)

            # --- Drawing and saving ---
            if selected_shock_long:
                shock_name = convert(
                    selected_shock_long,
                    M_list[0],
                    vartype="exo",
                    length="short",
                )
                st.subheader(f"Shock: {selected_shock_long}")
                include_title = st.checkbox(
                    "Include figure title in the exported file",
                    value=True,
                )
                file_format = st.selectbox(
                    "Select file format to download:",
                    options=["png", "pdf", "eps", "svg", "pkl"],
                )
                # --- IRF thresholding for direct matplotlib plotting ---
                for irf_dfs in shock_dfs_list:
                    df = irf_dfs[shock_name]
                    for col in df.columns:
                        arr = df[col].to_numpy().copy()
                        if np.nanmax(np.abs(arr)) < plot_threshold:
                            arr[:] = 0
                            df[col] = arr
                fig, axes = plt.subplots(n_rows, n_col, figsize=(fig_width, fig_height))
                if n_vars == 1:
                    axes = [[axes]]
                elif n_rows == 1:
                    axes = [axes]
                elif n_col == 1:
                    axes = [[ax] for ax in axes]
                else:
                    axes = axes.reshape((n_rows, n_col))
                for idx_var, var_long in enumerate(selected_endo_names_long):
                    row = idx_var // n_col
                    col = idx_var % n_col
                    ax = axes[row][col]
                    for irf_dfs, endo_short, plot_opt in zip(
                        shock_dfs_list,
                        selected_endo_names_short_list,
                        file_plot_options,
                        strict=False,
                    ):
                        var_short = endo_short[idx_var]
                        irfs = irf_dfs[shock_name][:periods]
                        ax.plot(
                            irfs.index,
                            irfs[var_short],
                            label=(
                                plot_opt["legend_label"]
                                if show_legend
                                and (
                                    legend_panel_mode == 0
                                    or (legend_panel_mode == 1 and idx_var == 0)
                                )
                                else None
                            ),
                            marker=plot_opt["marker"],
                            linestyle=plot_opt["linestyle"],
                            color=plot_opt["color"],
                        )
                    if show_grid:
                        ax.grid(visible=True)
                    ax.set_title(var_long)
                    ax.set_xlabel(plot_xlabel)
                    ax.set_ylabel(plot_ylabel)
                    if show_legend and (
                        legend_panel_mode == 0
                        or (legend_panel_mode == 1 and idx_var == 0)
                    ):
                        ax.legend()
                # Hide unused subplots
                for idx in range(n_vars, n_rows * n_col):
                    row = idx // n_col
                    col = idx % n_col
                    axes[row][col].axis("off")
                if include_title:
                    fig.suptitle(f"IRFs for shock: {selected_shock_long}")
                fig.tight_layout(rect=[0, 0.03, 1, 0.95] if include_title else None)
                st.pyplot(fig)
                with st.expander("Display IRF Data"):
                    for mat_name, irf_dfs in zip(
                        mat_file_names,
                        shock_dfs_list,
                        strict=False,
                    ):
                        st.write(f"{mat_name}")
                        st.write(irf_dfs[shock_name][:periods])
                base_file_name = f"{'_'.join(mat_file_names)}_{shock_name}"
                fig_for_save = fig if include_title else remove_suptitle(fig)
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
            else:
                st.warning("Please select at least one shock to plot.")
        else:
            st.warning("Please select at least one endogenous variable to plot.")

# --- Multi-page support ---
try:
    import streamlit

    if hasattr(streamlit, "sidebar"):
        st.sidebar.page_link("app.py", label="IRF Plotter", icon="📈")
        st.sidebar.page_link("documentation.py", label="Documentation", icon="📖")
except Exception:
    pass

st.markdown("---")
st.markdown(text.desclaimer())
st.markdown("---")
st.markdown(text.copyright())
