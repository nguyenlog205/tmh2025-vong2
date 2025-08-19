"""
Pipeline chính cho VAR: var_pipeline.py
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR, VECM # Thêm VECM
from testing_models import (
    perform_adf_test,
    perform_granger_causality_test,
    perform_ljungbox_test,
    calculate_vif,
    perform_jarque_bera_test,
    perform_pca_on_group,
    perform_johansen_cointegration_test # Thêm hàm mới
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


def var_pipeline(
    data_root,
    VNIndex=False,
    use_granger_selection=False,
    pca_variance_threshold=0.90,
    vif_threshold=5.0,
    forecast_steps=30,
    market_files=None
):
    # --- load datasets ---
    data_root = Path(data_root) # Đảm bảo data_root là đối tượng Path
    silver_path = data_root / 'data' / 'silver'

    # --- load datasets ---
    internal_path = silver_path / 'internal_data' / 'Internal_Data_Financial_Report.csv'
    internal_variables = pd.read_csv(internal_path, parse_dates=['time'])
    internal_variables['time'] = pd.to_datetime(internal_variables['time'], errors='coerce')

    ecb_path = silver_path / 'macro_economic_data' / 'policy_interest_rate' / 'ECB_INTEREST_RATE_FRED.csv'
    ecb = pd.read_csv(ecb_path, usecols=['time', 'close'], parse_dates=['time']).rename(columns={'close': 'ECB_RATE'})
    ecb['time'] = pd.to_datetime(ecb['time'], errors='coerce')
    
    fed_funds_path = silver_path / 'macro_economic_data' / 'policy_interest_rate' / 'FED_FUNDS.csv'
    fed_funds = pd.read_csv(fed_funds_path, usecols=['time', 'close'], parse_dates=['time']).rename(columns={'close': 'FED_FUNDS'})
    fed_funds['time'] = pd.to_datetime(fed_funds['time'], errors='coerce')

    folder_macro = silver_path / 'macro_economic_data' / 'growth_and_inflation'
    growth_inflation = load_and_merge_csv(folder_macro)

    folder_market = silver_path / 'market_data'
    if market_files is None:
        market_files = [
            "CBOE_Volatility_Index_FRED.csv",
            "CDS_5Y_CS_1D.csv",
            "PRICE_CS_1D.csv",
            "SX7E_STOXX_Banks_EUR_Price.csv",
            "VNINDEX_1D.csv"
        ]
    market_data = load_and_merge_csv(folder_market, market_files)

    # *** THÊM LOGIC ĐỌC DỮ LIỆU SENTIMENT ***
    news_path = silver_path / 'news' / 'news00.csv'
    sentiment_data = pd.read_csv(news_path)
    sentiment_data.rename(columns={'date': 'time'}, inplace=True)
    sentiment_data['time'] = pd.to_datetime(sentiment_data['time'], errors='coerce')


    if not VNIndex and 'VNINDEX_1D' in market_data.columns:
        market_data = market_data.drop(columns=['VNINDEX_1D'], errors='ignore')

    # TẠO BẢN SAO CỦA DỮ LIỆU GỐC TRƯỚC KHI BIẾN ĐỔI
    market_data_original_levels = market_data.copy()

    # --- merge all into df_full ---
    # SỬ DỤNG market_data_original_levels ĐỂ MERGE, SAU ĐÓ SẼ XỬ LÝ LOG-RETURN/DIFF SAU
    components = [internal_variables, ecb, fed_funds, growth_inflation, market_data_original_levels, sentiment_data]
    df_full = None
    for comp in components:
        if comp is None or (isinstance(comp, pd.DataFrame) and comp.shape[0] == 0):
            continue
        if df_full is None:
            df_full = comp.copy()
        else:
            if 'time' not in comp.columns:
                continue
            df_full = pd.merge(df_full, comp, on='time', how='outer')

    if df_full is None:
        raise ValueError('Không có dữ liệu hợp lệ để gộp df_full')

    df_full['time'] = pd.to_datetime(df_full['time'], errors='coerce')
    df_full = df_full.sort_values('time').reset_index(drop=True)

    # --- define endogenous (market) and candidate exogenous ---
    market_cols = [c for c in market_data_original_levels.columns if c != 'time']
    endog_cols = market_cols.copy()
    if not VNIndex and 'VNINDEX_1D' in endog_cols:
        endog_cols = [c for c in endog_cols if c != 'VNINDEX_1D']

    candidate_exogs = [c for c in df_full.columns if c not in endog_cols and c != 'time']
    candidate_exogs = [c for c in candidate_exogs if pd.api.types.is_numeric_dtype(df_full[c])]

    print('\n=== Tổng quan biến ===')
    print('Endogenous (market, original levels):', endog_cols)
    print('Candidate exogenous variables:', candidate_exogs)

    # --- 1.1 ADF trên tất cả numeric ---
    stationary_vars = []
    non_stationary_vars = []
    numeric_cols = [c for c in df_full.columns if c != 'time' and pd.api.types.is_numeric_dtype(df_full[c])]
    for col in numeric_cols:
        p = perform_adf_test(df_full[col], col)
        if not np.isnan(p) and p <= 0.05:
            stationary_vars.append(col)
        else:
            non_stationary_vars.append(col)
    print('\nDừng:', stationary_vars)
    print('\nKhông dừng:', non_stationary_vars)

    # *** LOGIC MỚI: KIỂM TRA ĐỒNG TÍCH HỢP TRƯỚC KHI LẤY SAI PHÂN ***
    non_stationary_endogs = [c for c in endog_cols if c in non_stationary_vars]
    coint_rank = 0
    if len(non_stationary_endogs) >= 2:
        coint_rank = perform_johansen_cointegration_test(df_full[non_stationary_endogs])
    
    use_vecm = coint_rank > 0

    if use_vecm:
        print(f"\n>>> LỰA CHỌN MÔ HÌNH: VECM (vì có {coint_rank} quan hệ đồng tích hợp)")
        # Với VECM, ta dùng dữ liệu gốc cho biến nội sinh
        # Nhưng các biến ngoại sinh không dừng vẫn cần được lấy sai phân
        df_model_data = df_full.copy()
        exogs_to_diff = [c for c in candidate_exogs if c in non_stationary_vars]
        for col in exogs_to_diff:
             if pd.api.types.is_numeric_dtype(df_model_data[col]):
                df_model_data[col] = df_model_data[col].diff()
        
        # Biến market (nội sinh) đã được chuyển thành log-return (dừng) sẽ được dùng làm biến ngoại sinh nếu chúng không nằm trong nhóm đồng tích hợp
        stationary_endogs = [c for c in endog_cols if c in stationary_vars]
        for col in stationary_endogs:
            df_model_data[col] = np.log(df_model_data[col]).diff()

    else:
        print("\n>>> LỰA CHỌN MÔ HÌNH: VAR (vì không có hoặc không đủ biến để kiểm định đồng tích hợp)")
        # Quy trình cũ: lấy sai phân tất cả các biến không dừng
        df_model_data = df_full.copy()
        # Biến market đã được chuyển thành log-return, giờ xử lý các biến còn lại
        other_non_stationary = [c for c in non_stationary_vars if c not in market_cols]
        for col in market_cols: # Chuyển market data thành log returns
            df_model_data[col] = np.log(df_model_data[col]).diff()
        for col in other_non_stationary:
            if pd.api.types.is_numeric_dtype(df_model_data[col]):
                df_model_data[col] = df_model_data[col].diff()
        
    df_diff = df_model_data.dropna().reset_index(drop=True)
    print(f"\nKích thước dữ liệu sau khi xử lý và dropna: {df_diff.shape}")


    # --- 1.4 VIF & PCA cho EXOG GROUPS ONLY ---
    groups = {}
    internal_cols = [c for c in internal_variables.columns if c != 'time']
    # Thêm các cột sentiment vào nhóm 'others'
    sentiment_cols = [c for c in sentiment_data.columns if c != 'time']
    
    for col in candidate_exogs:
        if col in internal_cols:
            groups.setdefault('internal_variables', []).append(col)
        elif col in sentiment_cols:
             groups.setdefault('sentiment_data', []).append(col)
        else:
            groups.setdefault('others', []).append(col)

    final_exogs = []
    for grp_name, grp_cols in groups.items():
        print(f"\nNhóm: {grp_name} (số biến = {len(grp_cols)})")
        grp_cols_in_df = [c for c in grp_cols if c in df_diff.columns]
        if len(grp_cols_in_df) == 0:
            print('Không có biến hợp lệ trong df_diff cho nhóm này. Bỏ qua.')
            continue
        vif_df = calculate_vif(df_diff[grp_cols_in_df])
        max_vif = vif_df['VIF'].max() if not vif_df.empty else 0
        print(f"Max VIF của nhóm {grp_name} = {max_vif:.3f}")
        if not vif_df.empty and (max_vif >= vif_threshold) and (len(grp_cols_in_df) >= 2):
            print(f"Thực hiện PCA cho {grp_name} vì max VIF >= {vif_threshold}")
            pca_df, pca_obj, scaler = perform_pca_on_group(df_diff[grp_cols_in_df], variance_threshold=pca_variance_threshold, prefix=grp_name + '_PCA')
            if pca_df is not None and not pca_df.empty:
                pca_df = pca_df.reset_index(drop=True)
                df_diff = df_diff.reset_index(drop=True)
                df_diff = df_diff.drop(columns=grp_cols_in_df, errors='ignore')
                df_diff = pd.concat([df_diff, pca_df], axis=1)
                final_exogs.extend(list(pca_df.columns))
            else:
                print('PCA thất bại -> giữ nguyên biến gốc')
                final_exogs.extend(grp_cols_in_df)
        else:
            print(f"Không cần PCA cho nhóm {grp_name}.")
            final_exogs.extend(grp_cols_in_df)

    if not VNIndex and 'VNINDEX_1D' in final_exogs:
        final_exogs = [c for c in final_exogs if c != 'VNINDEX_1D']

    final_exogs = [c for c in dict.fromkeys(final_exogs) if c in df_diff.columns]
    print(f"\nFinal exogenous variables after grouping/PCA: {final_exogs}")

    if use_granger_selection and len(final_exogs) > 0:
        print('\n*** Chọn exog bằng Granger causality (tùy chọn) ***')
        # Granger chỉ nên chạy trên dữ liệu dừng
        chosen = select_exogenous_by_granger(df_diff, endog_cols, final_exogs)
        if len(chosen) > 0:
            print('Biến chọn bởi Granger (ghi đè):', chosen)
            final_exogs = chosen
        else:
            print('Không tìm thấy exog đáng kể theo Granger -> giữ nguyên final_exogs')

    # -----------------------
    # PHA 2: Xây VAR hoặc VECM
    # -----------------------
    if use_vecm:
        # Dữ liệu cho VECM
        endog_data_for_var = df_diff[non_stationary_endogs]
    else:
        # Dữ liệu cho VAR
        endog_data_for_var = df_diff[endog_cols]

    exog_data_for_var = df_diff[final_exogs] if len(final_exogs) > 0 else None

    if endog_data_for_var.shape[0] < 10:
        raise ValueError('Dữ liệu nội sinh sau xử lý quá ít (ít hơn 10 hàng).')

    print('\n*** Lựa chọn độ trễ p tối ưu ***')
    # Chọn độ trễ dựa trên mô hình VAR trên dữ liệu đã dừng
    temp_endog_for_lag = df_diff[endog_cols] if not use_vecm else df_diff[non_stationary_endogs].diff().dropna()
    temp_model_for_lag_selection = VAR(temp_endog_for_lag, exog=exog_data_for_var.loc[temp_endog_for_lag.index] if exog_data_for_var is not None else None)

    lag_selection = temp_model_for_lag_selection.select_order(maxlags=8)
    try:
        optimal_lag = int(lag_selection.aic)
    except Exception:
        optimal_lag = 1
    optimal_lag = max(1, optimal_lag)
    print('Order selection summary:\n', lag_selection.summary())
    print(f'=> Độ trễ tối ưu (cho VAR tương ứng) = {optimal_lag}')

    # Fit model
    if use_vecm:
        print('\n*** Fit VECM ***')
        # k_ar_diff cho VECM là p-1 của VAR tương ứng
        model = VECM(endog_data_for_var, exog=exog_data_for_var, k_ar_diff=optimal_lag - 1, coint_rank=coint_rank)
        results = model.fit()
    else:
        print('\n*** Fit VAR ***')
        model = VAR(endog_data_for_var, exog=exog_data_for_var)
        results = model.fit(optimal_lag)
    
    print(results.summary())

    # -----------------------
    # Diagnostics
    # -----------------------
    print('\n*** Diagnostics: Ljung-Box (resid) ***')
    residuals = results.resid
    perform_ljungbox_test(residuals, lags=optimal_lag + 1)
    
    print('\n*** Diagnostics: Jarque-Bera (resid) ***')
    perform_jarque_bera_test(residuals)

    try:
        if not use_vecm: # VECM không có hàm is_stable() trực tiếp
            stability = results.is_stable()
            print('\nModel stability (is_stable):', stability)
        roots = results.roots
        print('\nCharacteristic roots (abs):', np.abs(roots))
    except Exception:
        pass

    # IRF & FEVD
    try:
        irf = results.irf(forecast_steps)
        print('\nIRF object created (steps=', forecast_steps, ')')
    except Exception as e:
        print('Không thể tạo IRF:', e)
        irf = None
    try:
        fevd = results.fevd(forecast_steps)
        print('FEVD object created (steps=', forecast_steps, ')')
    except Exception as e:
        print('Không thể tạo FEVD:', e)
        fevd = None

    # Forecast
    steps = forecast_steps
    try:
        exog_future = None
        if exog_data_for_var is not None:
            last_exog = exog_data_for_var.values[-1]
            exog_future = np.tile(last_exog.reshape(1, -1), (steps, 1))
        
        if use_vecm:
            # Hàm forecast của VECM không cần tham số 'y'
            fc = results.forecast(steps=steps, exog_future=exog_future)
        else:
            # Hàm forecast của VAR yêu cầu tham số 'y' (các quan sát cuối)
            last_obs = endog_data_for_var.values[-results.k_ar:]
            fc = results.forecast(y=last_obs, steps=steps, exog_future=exog_future)

        last_time = df_diff['time'].iloc[-1]
        future_index = pd.date_range(last_time + pd.Timedelta(days=1), periods=steps, freq='D')
        
        # Lấy đúng tên cột cho forecast_df
        forecast_cols = non_stationary_endogs if use_vecm else endog_cols
        forecast_df = pd.DataFrame(fc, columns=forecast_cols, index=future_index)
        print(f"\nForecast (next {steps} days) created.\n")
    except Exception as e:
        print('Không thể dự báo:', e)
        forecast_df = pd.DataFrame()

    return {
        'df_full': df_full,
        'df_diff': df_diff,
        'results': results,
        'forecast': forecast_df,
        'irf': irf,
        'fevd': fevd,
        'final_exogs': final_exogs,
        'model_type': 'VECM' if use_vecm else 'VAR'
    }
