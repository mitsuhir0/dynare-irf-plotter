import streamlit as st
from main import load, get_irf, plot_irf_df, convert, get_irf_endo_vars, dump_figure
from datetime import datetime
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
            default=endo_names_long[:5]  # デフォルトで最初の5つを選択
        )

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

            # オプション設定
            with st.expander("Plot Options"):
                n_col = st.number_input(
                    "Number of columns for the plot layout:",
                    min_value=1,
                    max_value=5,
                    value=2,  # デフォルト値
                    step=1
                )
                plot_xlabel = st.text_input("X-axis label:", value="Time")
                plot_ylabel = st.text_input("Y-axis label:", value="Response")
                perods = st.number_input(
                    "Number of periods to plot:",
                    min_value=1,
                    max_value=100,
                    value=len(shock_dfs[shock_list[0]]),  # デフォルト値
                    step=1
                )

            if selected_shocks:
                # 選択されたショックごとにプロット
                for long_shock_name in selected_shocks:
                    shock_name = convert(long_shock_name, M_, vartype='exo', length='short')
                    df = shock_dfs[shock_name]
                    st.subheader(f"Shock: {long_shock_name}")

                    # プロット
                    fig = plot_irf_df(
                        df[:perods],
                        selected_endo_names_short,
                        shock_name,
                        n_cols=n_col,
                        M_=M_,
                        xlabel=plot_xlabel,
                        ylabel=plot_ylabel,
                    )
                    st.pyplot(fig)

                    # データフレームを表示
                    with st.expander("DataFrame Display"):
                        st.write(df)

                    # ファイル名の生成
                    mat_file_name = os.path.splitext(uploaded_file.name)[0]
                    today_date = datetime.now().strftime("%Y-%m-%d")
                    base_file_name = f"{mat_file_name}_{long_shock_name}_{today_date}"

                    # 保存形式を選択して即時ダウンロード
                    file_format = st.selectbox(
                        "Select file format to download:",
                        options=["pkl", "png", "eps", "pdf"]
                    )

                    if file_format == "pkl":
                        pkl_bytes = dump_figure(fig)
                        st.download_button(
                            label="Download as pkl",
                            data=pkl_bytes,
                            file_name=f"{base_file_name}.pkl",
                            mime="application/octet-stream"
                        )
                    else:
                        buffer = io.BytesIO()
                        fig.savefig(buffer, format=file_format)
                        buffer.seek(0)
                        mime_type = {
                            "png": "image/png",
                            "eps": "application/postscript",
                            "pdf": "application/pdf"
                        }[file_format]
                        st.download_button(
                            label=f"Download as {file_format}",
                            data=buffer,
                            file_name=f"{base_file_name}.{file_format}",
                            mime=mime_type
                        )
            else:
                st.warning("Please select at least one shock to plot.")
        else:
            st.warning("Please select at least one endogenous variable to plot.")
