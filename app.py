"""Streamlit app for visualizing Dynare IRFs from MATLAB .mat files."""

import io
from pathlib import Path

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

mat_file_path = None
uploaded_file = None

if not use_sample_file:
    uploaded_file = st.file_uploader(
        "Upload a MAT file",
        type=["mat"],
        disabled=use_sample_file,
    )

    sample_file_path = "sample.mat"
    if Path(sample_file_path).exists():
        mat_file_path = sample_file_path
    else:
        st.error("Sample MAT file (sample.mat) not found.")
        st.error("Sample MAT file (sample.mat) not found.")

elif uploaded_file is not None:
    mat_file_name = Path(uploaded_file.name).stem

    mat_file_path = uploaded_file

if mat_file_path is not None:
    if isinstance(mat_file_path, str):  # Path to the sample file
        data = load(mat_file_path)
    else:  # Uploaded file (BytesIO)
        data = load(mat_file_path)

    oo_ = data.get("oo_", None)
    M_ = data.get("M_", None)

    # fix typos in orginal Dynare file
    if use_sample_file:
        target = "monetary policy shock"
        replacement = "cost push shock"
        M_.exo_names_long = [
            replacement if s == target else s for s in M_.exo_names_long
        ]

    if oo_ is None:
        st.error("The uploaded MAT file does not contain 'oo_' data.")
    else:
        # Retrieve the list of endo_names_long
        endo_vars_shocks = get_irf_endo_vars(oo_, M_)
        endo_vars = endo_vars_shocks[next(iter(endo_vars_shocks.keys()))]

        endo_names_long = sorted(
            [convert(name, M_, vartype="endo", length="long") for name in endo_vars],
        )

        # Allow the user to select endogenous variables
        selected_endo_names_long = st.multiselect(
            "Select endogenous variables to plot:",
            options=endo_names_long,
            default=endo_names_long[:5],  # Select the first 5 by default
        )

        if selected_endo_names_long:
            # Convert selected variables to short names
            selected_endo_names_short = [
                convert(long_name, M_, vartype="endo", length="short")
                for long_name in selected_endo_names_long
            ]

            # Retrieve IRF dataframes
            shock_dfs = get_irf(oo_, M_)

            # Retrieve the list of shocks
            shock_list = list(shock_dfs.keys())
            long_shock_list = [
                convert(shock, M_, vartype="exo", length="long") for shock in shock_list
            ]

            # Allow the user to select shocks
            selected_shocks = st.selectbox(
                "Select shocks to plot:",
                options=long_shock_list,
                index=0,  # Select the first one by default
            )

            # Plot options
            with st.expander("Plot Options"):
                n_col = st.number_input(
                    "Number of columns for the plot layout:",
                    min_value=1,
                    max_value=5,
                    value=2,  # Default value
                    step=1,
                )
                plot_xlabel = st.text_input("X-axis label:", value="Time")
                plot_ylabel = st.text_input("Y-axis label:", value="Response")
                perods = st.number_input(
                    "Number of periods to plot:",
                    min_value=1,
                    max_value=100,
                    value=len(shock_dfs[shock_list[0]]),  # Default value
                    step=1,
                )

            if selected_shocks:
                # Plot for each selected shock
                for long_shock_name in [selected_shocks]:
                    shock_name = convert(
                        long_shock_name,
                        M_,
                        vartype="exo",
                        length="short",
                    )
                    irfs = shock_dfs[shock_name]
                    st.subheader(f"Shock: {long_shock_name}")

                    # Plot
                    fig = plot_irf_df(
                        irfs[:perods],
                        selected_endo_names_short,
                        shock_name,
                        n_cols=n_col,
                        M_=M_,
                        xlabel=plot_xlabel,
                        ylabel=plot_ylabel,
                    )
                    st.pyplot(fig)

                    # Display the dataframe
                    with st.expander("Display IRF Data"):
                        st.write(irfs)

                    # Generate file name
                    if uploaded_file is not None:
                        mat_file_name = Path(uploaded_file.name).stem
                    else:
                        mat_file_name = "sample"
                    base_file_name = f"{mat_file_name}_{shock_name}"

                    # Select file format and download immediately
                    include_title = st.checkbox(
                        "Include figure title in the exported file",
                        value=True,
                    )
                    fig_for_save = fig if include_title else remove_suptitle(fig)
                    file_format = st.selectbox(
                        "Select file format to download:",
                        options=["png", "pdf", "eps", "svg", "pkl"],
                    )

                    if file_format == "pkl":
                        pkl_bytes = dump_figure(fig_for_save)
                        st.download_button(
                            label="Download as pkl",
                            data=pkl_bytes,
                            file_name=f"{base_file_name}.pkl",
                            mime="application/octet-stream",
                        )
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

                    if file_format == "pkl":
                        with st.expander("About the PKL File"):
                            st.markdown(text.about_pkl())
            else:
                st.warning("Please select at least one shock to plot.")
        else:
            st.warning("Please select at least one endogenous variable to plot.")


st.markdown("---")
st.markdown(text.desclaimer())
st.markdown("---")
st.markdown(text.copyright())
