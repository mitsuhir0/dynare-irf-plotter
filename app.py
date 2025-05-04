import streamlit as st
from main import load, get_irf, plot_irf_df, convert, get_irf_endo_vars, dump_figure
import os
import io


st.set_page_config(
    page_title="IRF Plotter",
    page_icon="📈",
)
st.title("IRF Plotter for MAT Files")

# MAT ファイルのアップロード
uploaded_file = st.file_uploader("Upload a MAT file", type=["mat"])

if uploaded_file is not None:
    # MAT ファイルの読み込み
    data = load(uploaded_file)
    oo_ = data.get('oo_', None)
    M_ = data.get('M_', None)

    if oo_ is None:
        st.error("The uploaded MAT file does not contain 'oo_' data.")
    else:
        # `endo_names_long` の一覧を取得
        endo_vars_shocks = get_irf_endo_vars(oo_, M_)
        endo_vars = endo_vars_shocks[list(endo_vars_shocks.keys())[0]]

        endo_names_long = sorted([convert(name, M_, vartype='endo', length='long') for name in endo_vars])

        # ユーザーに内生変数を選択させる
        selected_endo_names_long = st.multiselect(
            "Select endogenous variables to plot:",
            options=endo_names_long,
            default=endo_names_long[:3]  # デフォルトで最初の3つを選択
        )

        # ユーザーに列数を選択させる
        n_col = st.number_input(
            "Select the number of columns for the plot layout:",
            min_value=1,
            max_value=5,
            value=2,  # デフォルト値
            step=1
        )

        # ユーザーにタイトルやラベルを入力させる
        plot_xlabel = st.text_input("Enter x-axis label:", value="Time")
        plot_ylabel = st.text_input("Enter y-axis label:", value="Response")

        if selected_endo_names_long:
            # 選択された変数を short name に変換
            selected_endo_names_short = [
                convert(long_name, M_, vartype='endo', length='short') for long_name in selected_endo_names_long
            ]

            # IRF データフレームの取得
            shock_dfs = get_irf(oo_, M_)

            # ショックリストを取得
            shock_list = list(shock_dfs.keys())
            long_shock_list = [convert(shock, M_, vartype='exo', length='long') for shock in shock_list]

            # ユーザーにショックを選択させる
            selected_shocks = st.multiselect(
                "Select shocks to plot:",
                options=long_shock_list,
                default=long_shock_list[:1]  # デフォルトで最初の1つを選択
            )

            if selected_shocks:
                # 選択されたショックごとにプロット
                for long_shock_name in selected_shocks:
                    shock_name = convert(long_shock_name, M_, vartype='exo', length='short')
                    df = shock_dfs[shock_name]
                    st.subheader(f"Shock: {long_shock_name}")

                    # プロット
                    fig = plot_irf_df(
                        df,
                        selected_endo_names_short,
                        shock_name,
                        n_cols=n_col,
                        M_=M_,
                        xlabel=plot_xlabel,
                        ylabel=plot_ylabel,
                    )
                    st.pyplot(fig)

                    # ファイル名の生成
                    mat_file_name = os.path.splitext(uploaded_file.name)[0]
                    base_file_name = f"{mat_file_name}_{shock_name}"

                    # ダウンロード用のバイトデータを生成
                    # 1. pkl
                    pkl_bytes = dump_figure(fig)
                    st.download_button(
                        label=f"Download as .pkl for {long_shock_name}",
                        data=pkl_bytes,
                        file_name=f"{base_file_name}.pkl",
                        mime="application/octet-stream"
                    )

                    # 2. png
                    png_buffer = io.BytesIO()
                    fig.savefig(png_buffer, format="png")
                    png_buffer.seek(0)
                    st.download_button(
                        label=f"Download as .png for {long_shock_name}",
                        data=png_buffer,
                        file_name=f"{base_file_name}.png",
                        mime="image/png"
                    )

                    # 3. eps
                    eps_buffer = io.BytesIO()
                    fig.savefig(eps_buffer, format="eps")
                    eps_buffer.seek(0)
                    st.download_button(
                        label=f"Download as .eps for {long_shock_name}",
                        data=eps_buffer,
                        file_name=f"{base_file_name}.eps",
                        mime="application/postscript"
                    )

                    # 4. pdf
                    pdf_buffer = io.BytesIO()
                    fig.savefig(pdf_buffer, format="pdf")
                    pdf_buffer.seek(0)
                    st.download_button(
                        label=f"Download as .pdf for {long_shock_name}",
                        data=pdf_buffer,
                        file_name=f"{base_file_name}.pdf",
                        mime="application/pdf"
                    )

                    # データフレームを表示
                    st.write(df)

            else:
                st.warning("Please select at least one shock to plot.")
        else:
            st.warning("Please select at least one endogenous variable to plot.")
