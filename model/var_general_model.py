"""
Pipeline chính cho VAR: var_pipeline.py
"""

import os
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from testing_models import (
    perform_adf_test,
    perform_granger_causality_test,
    perform_ljungbox_test,
    calculate_vif,
    perform_jarque_bera_test,
    perform_pca_on_group
)
import warnings
warnings.filterwarnings("ignore")


def load_and_merge_csv(folder, files=None, parse_dates=["time"]):
    """
    Đọc nhiều file CSV trong folder, chỉ lấy time & close,
    đổi tên close = tên file (bỏ .csv), rồi merge theo time.
    """
    folder = os.path.normpath(folder)
    if files is None:
        files = [f for f in os.listdir(folder) if f.endswith(".csv")]

    df_final = None
    for file in files:
        path = os.path.join(folder, file)
        if not os.path.exists(path):
            print(f"Warning: không tìm thấy file {path}, bỏ qua.")
            continue
        try:
            df = pd.read_csv(path, usecols=["time", "close"], parse_dates=parse_dates)
        except Exception as e:
            print(f"Lỗi đọc {path}: {e}. Bỏ qua file.")
            continue

        col_name = os.path.splitext(file)[0]
        df = df.rename(columns={"close": col_name})
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')

        if df_final is None:
            df_final = df
        else:
            df_final = pd.merge(df_final, df, on="time", how="outer")

    if df_final is None:
        return pd.DataFrame(columns=['time'])

    df_final = df_final.sort_values("time").reset_index(drop=True)
    return df_final


def select_exogenous_by_granger(df_stationary, endogenous_vars, candidate_exogs, max_lag=4):
    """
    Dùng Granger test để chọn ra những candidate_exogs nào Granger-cause ít nhất
    một biến trong endogenous_vars. Trả về list các exog được chọn.
    """
    selected = []
    for exog in candidate_exogs:
        for endog in endogenous_vars:
            if exog not in df_stationary.columns or endog not in df_stationary.columns:
                continue
            _, any_sig = perform_granger_causality_test(df_stationary, response_var=endog, predictor_var=exog, max_lag=max_lag)
            if any_sig:
                selected.append(exog)
                break
    selected = list(dict.fromkeys(selected))
    return selected


def var_model(VNIndex=False, use_granger_selection=False, pca_variance_threshold=0.90, vif_threshold=5.0):
    """
    Chạy pipeline: đọc dữ liệu, gộp, kiểm định ADF, sai phân, VIF theo nhóm, PCA khi cần,
    (tùy chọn) chọn exog bằng Granger, fit VAR, kiểm tra residuals.
    """
    # --- đọc dữ liệu ---
    internal_variables = pd.read_csv(r"data\silver\internal_data\Internal_Data_Financial_Report.csv", parse_dates=['time'])
    internal_variables['time'] = pd.to_datetime(internal_variables['time'], errors='coerce')

    ecb = pd.read_csv(r"data\silver\macro_economic_data\policy_interest_rate\ECB_INTEREST_RATE_FRED.csv",
                      usecols=["time", "close"], parse_dates=['time']).rename(columns={"close": "ECB_RATE"})
    ecb['time'] = pd.to_datetime(ecb['time'], errors='coerce')

    fed_funds = pd.read_csv(r"data\silver\macro_economic_data\policy_interest_rate\FED_FUNDS.csv",
                            usecols=["time", "close"], parse_dates=['time']).rename(columns={"close": "FED_FUNDS"})
    fed_funds['time'] = pd.to_datetime(fed_funds['time'], errors='coerce')

    folder_macro = r"data\silver\macro_economic_data\growth_and_inflation"
    growth_inflation = load_and_merge_csv(folder_macro)

    folder_market = r"data\silver\market_data"
    files_market = [
        "CBOE_Volatility_Index_FRED.csv",
        "CDS_5Y_CS_1D.csv",
        "PRICE_CS_1D.csv",
        "SX7E_STOXX_Banks_EUR_Price.csv",
        "VNINDEX_1D.csv"
    ]
    market_data = load_and_merge_csv(folder_market, files_market)

    news_path = r'data\silver\news\news00.csv'
    sentiment_data = pd.read_csv(news_path)
    sentiment_data.rename(columns={'date': 'time'}, inplace=True)
    sentiment_data['time'] = pd.to_datetime(sentiment_data['time'], errors = 'coerce')
    # --- Gom tất cả về df_full ---
    components = [internal_variables, ecb, fed_funds, growth_inflation, market_data, sentiment_data]
    df_full = None
    for comp in components:
        if df_full is None:
            df_full = comp.copy()
        else:
            if comp is None or comp.shape[0] == 0:
                continue
            if 'time' not in comp.columns:
                continue
            df_full = pd.merge(df_full, comp, on="time", how="outer")

    if df_full is None:
        raise ValueError("Không có dữ liệu hợp lệ để gộp thành df_full.")

    df_full['time'] = pd.to_datetime(df_full['time'], errors='coerce')
    df_full = df_full.sort_values('time').reset_index(drop=True)

    # --- Xác định biến nội/ngoại sinh ---
    endog_cols = [c for c in market_data.columns if c != 'time']
    if not VNIndex and 'VNINDEX_1D' in endog_cols:
        endog_cols = [c for c in endog_cols if c != 'VNINDEX_1D']

    candidate_exogs = [c for c in df_full.columns if c not in endog_cols and c != 'time']
    candidate_exogs = [c for c in candidate_exogs if pd.api.types.is_numeric_dtype(df_full[c])]

    print("\n=== Tổng quan biến ===")
    print(f"Endogenous (market) variables: {endog_cols}")
    print(f"Candidate exogenous variables: {candidate_exogs}")

    # --- 1.1 Kiểm định tính dừng ---
    print("\n*** 1.1. Kiểm định tính dừng trên dữ liệu gốc ***")
    stationary_vars = []
    non_stationary_vars = []
    numeric_cols = [c for c in df_full.columns if c != 'time' and pd.api.types.is_numeric_dtype(df_full[c])]

    for col in numeric_cols:
        p_val = perform_adf_test(df_full[col], col)
        if not np.isnan(p_val) and p_val <= 0.05:
            stationary_vars.append(col)
        else:
            non_stationary_vars.append(col)

    print("\nDừng:", stationary_vars)
    print("\nKhông dừng:", non_stationary_vars)

    # --- 1.2 Sai phân bậc 1 cho các biến không dừng ---
    print("\n*** 1.2. Lấy sai phân bậc 1 và kiểm định lại các biến không dừng ***")
    df_diff = df_full.copy()
    for col in non_stationary_vars:
        if pd.api.types.is_numeric_dtype(df_diff[col]):
            df_diff[col] = df_diff[col].diff()
            _ = perform_adf_test(df_diff[col], f"{col}_diff1")
        else:
            print(f"Bỏ diff vì {col} không numeric.")
    df_diff = df_diff.dropna().reset_index(drop=True)
    print(f"\nKích thước df_diff sau dropna: {df_diff.shape}")

    # --- 1.4 Kiểm định đa cộng tuyến theo nhóm & PCA nếu cần ---
    print("\n*** 1.4. Kiểm định đa cộng tuyến theo nhóm và PCA khi VIF >= threshold ***")
    # Xác định nhóm: market_data, internal_variables, others
    groups = {}
    market_cols = [c for c in market_data.columns if c != 'time']
    internal_cols = [c for c in internal_variables.columns if c != 'time']

    for col in candidate_exogs:
        if col in market_cols:
            groups.setdefault('market_data', []).append(col)
        elif col in internal_cols:
            groups.setdefault('internal_variables', []).append(col)
        else:
            groups.setdefault('others', []).append(col)

    final_exogs = []  # sẽ chứa columns (hoặc tên PCA components) dùng làm exog
    # tiến hành kiểm tra nhóm từng nhóm
    for grp_name, grp_cols in groups.items():
        print(f"\nNhóm: {grp_name} (số biến = {len(grp_cols)})")
        # lọc những biến thực sự có trong df_diff
        grp_cols = [c for c in grp_cols if c in df_diff.columns]
        if len(grp_cols) == 0:
            print("Không có biến hợp lệ trong df_diff cho nhóm này. Bỏ qua.")
            continue

        # tính VIF trên nhóm
        vif_df = calculate_vif(df_diff[grp_cols])
        # check max VIF
        max_vif = vif_df['VIF'].max() if not vif_df.empty else 0
        print(f"Max VIF của nhóm {grp_name} = {max_vif:.3f}")

        if not vif_df.empty and (max_vif >= vif_threshold) and (len(grp_cols) >= 2):
            # thực hiện PCA trên nhóm
            print(f"Thực hiện PCA cho nhóm {grp_name} vì max VIF >= {vif_threshold}")
            pca_df, pca_obj, scaler = perform_pca_on_group(df_diff[grp_cols], variance_threshold=pca_variance_threshold, prefix=grp_name + "_PCA")
            if pca_df is not None and not pca_df.empty:
                # Align index: pca_df index = subset index after dropna inside PCA; chúng ta cần nối pca_df vào df_diff theo index
                # Vì df_diff đã dropna toàn bộ, pca_df index phải phù hợp (row counts equal). Safer: reset index and concat.
                pca_df = pca_df.reset_index(drop=True)
                # drop original columns from df_diff and append pca columns (we operate on copy)
                df_diff = df_diff.reset_index(drop=True)
                df_diff = df_diff.drop(columns=grp_cols, errors='ignore')
                # concat pca components to df_diff
                df_diff = pd.concat([df_diff, pca_df], axis=1)
                # add pca columns to final_exogs
                final_exogs.extend(list(pca_df.columns))
            else:
                # fallback: nếu PCA thất bại, giữ nguyên biến
                print(f"PCA không tạo được components cho nhóm {grp_name}, giữ nguyên biến gốc.")
                final_exogs.extend(grp_cols)
        else:
            # không cần PCA, dùng nguyên nhóm
            print(f"Không cần PCA cho nhóm {grp_name}.")
            final_exogs.extend(grp_cols)

    # Điều chỉnh loại bỏ VNINDEX_1D nếu cần
    if not VNIndex and 'VNINDEX_1D' in final_exogs:
        final_exogs = [c for c in final_exogs if c != 'VNINDEX_1D']

    # ensure uniqueness and only keep existing columns in df_diff
    final_exogs = [c for c in dict.fromkeys(final_exogs) if c in df_diff.columns]

    print(f"\nFinal exogenous variables after grouping/PCA: {final_exogs}")

    # --- (Tùy chọn) chọn bằng Granger trên df_diff (đã dừng) ---
    if use_granger_selection:
        print("\n*** 1.5. Chọn biến ngoại sinh bằng Granger causality (tùy chọn) ***")
        chosen_by_granger = select_exogenous_by_granger(df_diff, endog_cols, final_exogs, max_lag=4)
        if len(chosen_by_granger) > 0:
            print("Biến chọn bởi Granger (ghi đè):", chosen_by_granger)
            final_exogs = chosen_by_granger
        else:
            print("Không tìm thấy biến ngoại sinh nào có quan hệ Granger -> giữ final_exogs hiện tại.")

    # -----------------------
    # PHA 2: Xây dựng mô hình VAR
    # -----------------------
    endog_data_for_var = df_diff[endog_cols] if len(endog_cols) > 0 else pd.DataFrame()
    exog_data_for_var = df_diff[final_exogs] if len(final_exogs) > 0 else None

    if endog_data_for_var.shape[0] < 5:
        raise ValueError("Dữ liệu nội sinh sau xử lý quá ít để ước lượng VAR.")

    print("\n*** 2.1. Lựa chọn độ trễ p tối ưu ***")
    model = VAR(endog_data_for_var, exog=exog_data_for_var)
    lag_selection = model.select_order(maxlags=4)
    print("Order selection:\n", lag_selection.summary())
    try:
        optimal_lag = int(lag_selection.aic)
    except Exception:
        optimal_lag = 1
    print(f"=> Độ trễ tối ưu (theo AIC) = {optimal_lag}")

    print("\n*** 2.2. Fit mô hình VAR ***")
    results = model.fit(optimal_lag)
    print(results.summary())

    # -----------------------
    # PHA 3: Diagnostics
    # -----------------------
    print("\n*** 3.1. Kiểm định tự tương quan phần dư (Ljung-Box) ***")
    residuals = results.resid
    ljung_results = perform_ljungbox_test(residuals, lags=optimal_lag + 1)

    print("\n*** 3.2. Kiểm định phân phối chuẩn phần dư (Jarque-Bera) ***")
    jb_results = perform_jarque_bera_test(residuals)

    print("\n=== Hoàn tất pipeline VAR ===")
    try:
        irf = results.irf(10)
        irf.plot(orth=False)
        print("IRF plotted (nếu môi trường hỗ trợ đồ họa).")
    except Exception as e:
        print(f"Không thể vẽ IRF: {e}")

    return df_full, df_diff, results


if __name__ == "__main__":
    df_full, df_diff, results = var_model(VNIndex=False, use_granger_selection=True, pca_variance_threshold=0.90, vif_threshold=5.0)
