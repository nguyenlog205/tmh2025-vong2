"""
Pipeline chính cho VAR: var_pipeline.py
"""

import os
from pathlib import Path
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
        # Chuyển đổi data_root (có thể là string) thành đối tượng Path
    data_root = data_root / 'data' / 'sliver'

    # --- load datasets ---
    # Sử dụng toán tử / để nối đường dẫn một cách an toàn
    internal_path = data_root / 'internal_data' / 'Internal_Data_Financial_Report.csv'
    internal_variables = pd.read_csv(internal_path, parse_dates=['time'])
    internal_variables['time'] = pd.to_datetime(internal_variables['time'], errors='coerce')

    ecb_path = data_root / 'macro_economic_data' / 'policy_interest_rate' / 'ECB_INTEREST_RATE_FRED.csv'
    ecb = pd.read_csv(ecb_path, usecols=['time', 'close'], parse_dates=['time']).rename(columns={'close': 'ECB_RATE'})
    ecb['time'] = pd.to_datetime(ecb['time'], errors='coerce')
    
    fed_funds_path = data_root / 'macro_economic_data' / 'policy_interest_rate' / 'FED_FUNDS.csv'
    fed_funds = pd.read_csv(fed_funds_path, usecols=['time', 'close'], parse_dates=['time']).rename(columns={'close': 'FED_FUNDS'})
    fed_funds['time'] = pd.to_datetime(fed_funds['time'], errors='coerce')

    folder_macro = data_root / 'macro_economic_data' / 'growth_and_inflation'
    growth_inflation = load_and_merge_csv(folder_macro)

    folder_market = data_root / 'market_data'

    folder_market = os.path.join(data_root, 'market_data')
    if market_files is None:
        market_files = [
            "CBOE_Volatility_Index_FRED.csv",
            "CDS_5Y_CS_1D.csv",
            "PRICE_CS_1D.csv",
            "SX7E_STOXX_Banks_EUR_Price.csv",
            "VNINDEX_1D.csv"
        ]
    market_data = load_and_merge_csv(folder_market, market_files)

    # If VNIndex=False -> drop VNINDEX_1D everywhere (including market_data)
    if not VNIndex and 'VNINDEX_1D' in market_data.columns:
        market_data = market_data.drop(columns=['VNINDEX_1D'], errors='ignore')

    # Convert market_data price columns -> log returns
    market_price_cols = [c for c in market_data.columns if c != 'time']
    for col in market_price_cols:
        market_data[col] = np.log(market_data[col]).diff()

    # --- merge all into df_full ---
    components = [internal_variables, ecb, fed_funds, growth_inflation, market_data]
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
    market_cols = [c for c in market_data.columns if c != 'time']
    endog_cols = market_cols.copy()  # market_data luôn nội sinh
    if not VNIndex and 'VNINDEX_1D' in endog_cols:
        endog_cols = [c for c in endog_cols if c != 'VNINDEX_1D']

    candidate_exogs = [c for c in df_full.columns if c not in endog_cols and c != 'time']
    candidate_exogs = [c for c in candidate_exogs if pd.api.types.is_numeric_dtype(df_full[c])]

    print('\n=== Tổng quan biến ===')
    print('Endogenous (market, log-returns):', endog_cols)
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

    # --- 1.2 Diff non-stationary (first difference) ---
    df_diff = df_full.copy()
    for col in non_stationary_vars:
        if pd.api.types.is_numeric_dtype(df_diff[col]):
            df_diff[col] = df_diff[col].diff()
    df_diff = df_diff.dropna().reset_index(drop=True)
    print(f"\nKích thước df_diff sau dropna: {df_diff.shape}")

    # --- 1.4 VIF & PCA cho EXOG GROUPS ONLY (không chạm biến market/endog) ---
    groups = {}
    internal_cols = [c for c in internal_variables.columns if c != 'time']
    for col in candidate_exogs:
        if col in internal_cols:
            groups.setdefault('internal_variables', []).append(col)
        else:
            groups.setdefault('others', []).append(col)

    final_exogs = []
    for grp_name, grp_cols in groups.items():
        print(f"\nNhóm: {grp_name} (số biến = {len(grp_cols)})")
        grp_cols = [c for c in grp_cols if c in df_diff.columns]
        if len(grp_cols) == 0:
            print('Không có biến hợp lệ trong df_diff cho nhóm này. Bỏ qua.')
            continue
        vif_df = calculate_vif(df_diff[grp_cols])
        max_vif = vif_df['VIF'].max() if not vif_df.empty else 0
        print(f"Max VIF của nhóm {grp_name} = {max_vif:.3f}")
        if not vif_df.empty and (max_vif >= vif_threshold) and (len(grp_cols) >= 2):
            print(f"Thực hiện PCA cho {grp_name} vì max VIF >= {vif_threshold}")
            pca_df, pca_obj, scaler = perform_pca_on_group(df_diff[grp_cols], variance_threshold=pca_variance_threshold, prefix=grp_name + '_PCA')
            if pca_df is not None and not pca_df.empty:
                pca_df = pca_df.reset_index(drop=True)
                df_diff = df_diff.reset_index(drop=True)
                df_diff = df_diff.drop(columns=grp_cols, errors='ignore')
                df_diff = pd.concat([df_diff, pca_df], axis=1)
                final_exogs.extend(list(pca_df.columns))
            else:
                print('PCA thất bại -> giữ nguyên biến gốc')
                final_exogs.extend(grp_cols)
        else:
            print(f"Không cần PCA cho nhóm {grp_name}.")
            final_exogs.extend(grp_cols)

    # ensure VNINDEX removed from exogs if VNIndex=False
    if not VNIndex and 'VNINDEX_1D' in final_exogs:
        final_exogs = [c for c in final_exogs if c != 'VNINDEX_1D']

    final_exogs = [c for c in dict.fromkeys(final_exogs) if c in df_diff.columns]
    print(f"\nFinal exogenous variables after grouping/PCA: {final_exogs}")

    # optional Granger selection (on df_diff)
    if use_granger_selection and len(final_exogs) > 0:
        print('\n*** Chọn exog bằng Granger causality (tùy chọn) ***')
        chosen = []
        for ex in final_exogs:
            for end in endog_cols:
                if ex not in df_diff.columns or end not in df_diff.columns:
                    continue
                _, any_sig = perform_granger_causality_test(df_diff, response_var=end, predictor_var=ex, max_lag=4)
                if any_sig:
                    chosen.append(ex)
                    break
        chosen = list(dict.fromkeys(chosen))
        if len(chosen) > 0:
            print('Biến chọn bởi Granger (ghi đè):', chosen)
            final_exogs = chosen
        else:
            print('Không tìm thấy exog đáng kể theo Granger -> giữ nguyên final_exogs')

    # -----------------------
    # PHA 2: Xây VAR
    # -----------------------
    endog_data_for_var = df_diff[endog_cols] if len(endog_cols) > 0 else pd.DataFrame()
    exog_data_for_var = df_diff[final_exogs] if len(final_exogs) > 0 else None

    if endog_data_for_var.shape[0] < 10:
        raise ValueError('Dữ liệu nội sinh sau xử lý quá ít (ít hơn 10 hàng).')

    print('\n*** Lựa chọn độ trễ p tối ưu ***')
    model = VAR(endog_data_for_var, exog=exog_data_for_var)
    lag_selection = model.select_order(maxlags=8)
    try:
        optimal_lag = int(lag_selection.aic)
    except Exception:
        try:
            optimal_lag = int(getattr(lag_selection, 'aic') if hasattr(lag_selection, 'aic') else 1)
        except Exception:
            optimal_lag = 1
    optimal_lag = max(1, optimal_lag)
    print('Order selection summary:\n', lag_selection.summary())
    print(f'=> Độ trễ tối ưu = {optimal_lag}')

    print('\n*** Fit VAR ***')
    results = model.fit(optimal_lag)
    print(results.summary())

    # -----------------------
    # Diagnostics
    # -----------------------
    print('\n*** Diagnostics: Ljung-Box (resid) ***')
    residuals = results.resid
    ljung = perform_ljungbox_test(residuals, lags=optimal_lag + 1)
    print('\n*** Diagnostics: Jarque-Bera (resid) ***')
    jb = perform_jarque_bera_test(residuals)

    # Stability check
    try:
        stability = results.is_stable()
        print('\nModel stability (is_stable):', stability)
    except Exception:
        try:
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

    # Forecast (30 days) - if exog present, create exog_future by repeating last observed exog row
    steps = forecast_steps
    try:
        last_obs = endog_data_for_var.values[-results.k_ar:]
    except Exception:
        last_obs = endog_data_for_var.values[-1:]
    forecast_vals = None
    future_index = None
    try:
        if exog_data_for_var is None:
            fc = results.forecast(y=last_obs, steps=steps)
        else:
            last_exog = exog_data_for_var.values[-1]
            exog_future = np.tile(last_exog.reshape(1, -1), (steps, 1))
            try:
                fc = results.forecast(y=last_obs, steps=steps, exog_future=exog_future)
            except TypeError:
                # older/newer statsmodels interface fallback
                fc = results.forecast(y=last_obs, steps=steps)
        forecast_vals = fc
        # build future dates as daily sequence after last df_diff time
        last_time = df_diff['time'].iloc[-1]
        future_index = pd.date_range(last_time + pd.Timedelta(days=1), periods=steps, freq='D')
        forecast_df = pd.DataFrame(forecast_vals, columns=endog_cols, index=future_index)
        print(f"\nForecast (next {steps} days) created.\n")
    except Exception as e:
        print('Không thể dự báo:', e)
        forecast_df = pd.DataFrame()

    # return everything useful
    return {
        'df_full': df_full,
        'df_diff': df_diff,
        'results': results,
        'forecast': forecast_df,
        'irf': irf,
        'fevd': fevd,
        'final_exogs': final_exogs
    }
