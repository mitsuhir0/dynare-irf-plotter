import streamlit as st
from main import load, get_irf, plot_irf_df, get_endo_names_short


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
        endo_names_long = sorted([str(name).strip() for name in M_.endo_names_long])

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

        if selected_endo_names_long:
            # 選択された変数を short name に変換
            selected_endo_names_short = [
                get_endo_names_short(long_name, M_) for long_name in selected_endo_names_long
            ]

            # IRF データフレームの取得
            shock_dfs = get_irf(oo_, M_)

            # ショックリストを取得
            shock_list = list(shock_dfs.keys())

            # ユーザーにショックを選択させる
            selected_shocks = st.multiselect(
                "Select shocks to plot:",
                options=shock_list,
                default=shock_list[:1]  # デフォルトで最初の1つを選択
            )

            if selected_shocks:
                # 選択されたショックごとにプロット
                for shock_name in selected_shocks:
                    df = shock_dfs[shock_name]
                    st.subheader(f"Shock: {shock_name}")

                    # プロット
                    st.pyplot(plot_irf_df(df, selected_endo_names_short, shock_name, n_cols=n_col, M_=M_))

                    # データフレームを表示
                    st.write(df)
            else:
                st.warning("Please select at least one shock to plot.")
        else:
            st.warning("Please select at least one endogenous variable to plot.")
