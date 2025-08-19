import os
from pathlib import Path
import pandas as pd
import numpy as np
import numpy.linalg as npl
from statsmodels.tsa.api import VAR, VECM
from statsmodels.tsa.vector_ar.var_model import VARProcess
from testing_models import (
    perform_adf_test,
    perform_granger_causality_test,
    perform_ljungbox_test,
    calculate_vif,
    perform_jarque_bera_test,
    perform_pca_on_group,
    perform_johansen_cointegration_test
)
import warnings
warnings.filterwarnings("ignore")


def load_and_merge_csv(folder, files=None, parse_dates=["time"]):
    folder = os.path.normpath(str(folder))
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


# ----------------- helper functions for IRF/FEVD -----------------
def _ensure_coefs_shape(coefs):
    coefs = np.asarray(coefs)
    if coefs.ndim == 3:
        # detect if shape is (n, n, p) -> transpose to (p, n, n)
        if coefs.shape[2] != coefs.shape[1] and coefs.shape[0] == coefs.shape[1]:
            coefs = np.transpose(coefs, (2, 0, 1))
    elif coefs.ndim == 2:
        n = coefs.shape[0]
        coefs = coefs.reshape(1, n, n)
    else:
        raise ValueError(f"Unexpected coefs shape: {coefs.shape}")
    return coefs


def compute_irf_from_var_coefs(coefs, sigma_u, steps):
    coefs = _ensure_coefs_shape(coefs)
    p, n, _ = coefs.shape
    psi = [np.eye(n)]
    for h in range(1, steps + 1):
        psi_h = np.zeros((n, n))
        for i in range(1, min(p, h) + 1):
            psi_h += coefs[i - 1] @ psi[h - i]
        psi.append(psi_h)
    psi = np.stack(psi, axis=0)  # (steps+1, n, n)
    try:
        chol = npl.cholesky(sigma_u)
    except Exception:
        vals, vecs = npl.eigh(sigma_u)
        vals[vals < 0] = 0.0
        chol = vecs @ np.diag(np.sqrt(vals))
    ortho_irf = np.einsum('hij,jk->hik', psi, chol)  # (steps+1, n, n)
    return psi, ortho_irf


def compute_fevd_from_irf(ortho_irf, steps):
    H = steps
    _, n, _ = ortho_irf.shape
    fevd = np.zeros((H + 1, n, n))
    for h in range(1, H + 1):
        num = np.zeros((n, n))
        den = np.zeros((n,))
        for s in range(0, h):
            num += (ortho_irf[s] ** 2)
            den += np.sum(ortho_irf[s] ** 2, axis=1)
        for i in range(n):
            if den[i] == 0:
                fevd[h, i, :] = np.nan
            else:
                fevd[h, i, :] = num[i, :] / den[i]
    return fevd
# ----------------------------------------------------------------


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
    data_root = Path(data_root)
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

    news_path = silver_path / 'news' / 'news00.csv'
    sentiment_data = pd.read_csv(news_path)
    sentiment_data.rename(columns={'date': 'time'}, inplace=True)
    sentiment_data['time'] = pd.to_datetime(sentiment_data['time'], errors='coerce')


    if not VNIndex and 'VNINDEX_1D' in market_data.columns:
        market_data = market_data.drop(columns=['VNINDEX_1D'], errors='ignore')

    market_data_original_levels = market_data.copy()

    # --- merge all into df_full ---
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

    # *** CHECK COINTEGRATION BEFORE DIFFERENCING ***
    non_stationary_endogs = [c for c in endog_cols if c in non_stationary_vars]
    coint_rank = 0
    if len(non_stationary_endogs) >= 2:
        coint_rank = perform_johansen_cointegration_test(df_full[non_stationary_endogs])
    
    use_vecm = coint_rank > 0

    if use_vecm:
        print(f"\n>>> LỰA CHỌN MÔ HÌNH: VECM (vì có {coint_rank} quan hệ đồng tích hợp)")
        df_model_data = df_full.copy()
        exogs_to_diff = [c for c in candidate_exogs if c in non_stationary_vars]
        for col in exogs_to_diff:
             if pd.api.types.is_numeric_dtype(df_model_data[col]):
                df_model_data[col] = df_model_data[col].diff()
        
        stationary_endogs = [c for c in endog_cols if c in stationary_vars]
        for col in stationary_endogs:
            df_model_data[col] = np.log(df_model_data[col]).diff()

    else:
        print("\n>>> LỰA CHỌN MÔ HÌNH: VAR (vì không có hoặc không đủ biến để kiểm định đồng tích hợp)")
        df_model_data = df_full.copy()
        other_non_stationary = [c for c in non_stationary_vars if c not in market_cols]
        for col in market_cols:
            df_model_data[col] = np.log(df_model_data[col]).diff()
        for col in other_non_stationary:
            if pd.api.types.is_numeric_dtype(df_model_data[col]):
                df_model_data[col] = df_model_data[col].diff()
            
    df_diff = df_model_data.dropna().reset_index(drop=True)
    print(f"\nKích thước dữ liệu sau khi xử lý và dropna: {df_diff.shape}")

    # --- 1.4 VIF & PCA cho EXOG GROUPS ONLY ---
    groups = {}
    internal_cols = [c for c in internal_variables.columns if c != 'time']
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
        chosen = select_exogenous_by_granger(df_diff, endog_cols, final_exogs)
        if len(chosen) > 0:
            print('Biến chọn bởi Granger (ghi đè):', chosen)
            final_exogs = chosen
        else:
            print('Không tìm thấy exog đáng kể theo Granger -> giữ nguyên final_exogs')

    # PHA 2: Xây VAR hoặc VECM
    if use_vecm:
        endog_data_for_var = df_diff[non_stationary_endogs]
    else:
        endog_data_for_var = df_diff[endog_cols]

    exog_data_for_var = df_diff[final_exogs] if len(final_exogs) > 0 else None

    if endog_data_for_var.shape[0] < 10:
        raise ValueError('Dữ liệu nội sinh sau xử lý quá ít (ít hơn 10 hàng).')

    print('\n*** Lựa chọn độ trễ p tối ưu ***')
    temp_endog_for_lag = df_diff[endog_cols] if not use_vecm else df_diff[non_stationary_endogs].diff().dropna()
    temp_exog_for_lag = exog_data_for_var.loc[temp_endog_for_lag.index] if exog_data_for_var is not None else None
    temp_model_for_lag_selection = VAR(temp_endog_for_lag, exog=temp_exog_for_lag)

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
        model = VECM(endog_data_for_var, exog=exog_data_for_var, k_ar_diff=optimal_lag - 1, coint_rank=coint_rank)
        results = model.fit()
    else:
        print('\n*** Fit VAR ***')
        model = VAR(endog_data_for_var, exog=exog_data_for_var)
        results = model.fit(optimal_lag)
    
    print(results.summary())

    # Diagnostics
    print('\n*** Diagnostics: Ljung-Box (resid) ***')
    residuals = results.resid
    if isinstance(residuals, np.ndarray):
        endog_names_for_residuals = non_stationary_endogs if use_vecm else endog_cols
        residuals = pd.DataFrame(residuals, columns=endog_names_for_residuals)

    perform_ljungbox_test(residuals, lags=optimal_lag + 1)
    
    print('\n*** Diagnostics: Jarque-Bera (resid) ***')
    perform_jarque_bera_test(residuals)

    try:
        if not use_vecm:
            stability = results.is_stable()
            print('\nModel stability (is_stable):', stability)
        roots = results.roots
        print('\nCharacteristic roots (abs):', np.abs(roots))
    except Exception:
        pass

    # ---------- SỬA LỖI TẠI ĐÂY (IRF & FEVD) ----------
    irf = None
    fevd = None
    try:
        if use_vecm:
            var_rep_coefficients = results.var_rep()  # thường trả về (p, n, n) hoặc (n, n, p)
            sigma_u = results.sigma_u

            # Try to build VARProcess AND call its irf/fevd if available
            try:
                var_process = VARProcess(coefs=var_rep_coefficients, sigma_u=sigma_u)
                # robust: lấy attribute, kiểm tra kiểu trước khi gọi
                varproc_irf_attr = getattr(var_process, 'irf', None)
                varproc_fevd_attr = getattr(var_process, 'fevd', None)

                # IRF
                if isinstance(varproc_irf_attr, np.ndarray):
                    irf = varproc_irf_attr
                    print('IRF obtained as numpy.ndarray from VARProcess.irf attribute (no call).')
                elif callable(varproc_irf_attr):
                    try:
                        res_irf = varproc_irf_attr(forecast_steps)
                        irf = res_irf
                        print('IRF created by calling VARProcess.irf(...)')
                    except Exception as e_call:
                        print('Calling VARProcess.irf(...) failed:', e_call)
                else:
                    # not present or unknown type -> fallback manual
                    pass

                # FEVD
                if isinstance(varproc_fevd_attr, np.ndarray):
                    fevd = varproc_fevd_attr
                    print('FEVD obtained as numpy.ndarray from VARProcess.fevd attribute (no call).')
                elif callable(varproc_fevd_attr):
                    try:
                        res_fevd = varproc_fevd_attr(forecast_steps)
                        fevd = res_fevd
                        print('FEVD created by calling VARProcess.fevd(...)')
                    except Exception as e_call:
                        print('Calling VARProcess.fevd(...) failed:', e_call)

                # If either missing -> manual compute
                if irf is None or fevd is None:
                    psi, ortho_irf = compute_irf_from_var_coefs(var_rep_coefficients, sigma_u, forecast_steps)
                    irf = irf if irf is not None else ortho_irf
                    fevd = fevd if fevd is not None else compute_fevd_from_irf(ortho_irf, forecast_steps)
                    print('\nIRF/FEVD computed manually from VECM var_rep().')

            except Exception as e_varproc:
                # Fallback: compute manually from var_rep coefficients
                try:
                    psi, ortho_irf = compute_irf_from_var_coefs(var_rep_coefficients, sigma_u, forecast_steps)
                    irf = ortho_irf
                    fevd = compute_fevd_from_irf(ortho_irf, forecast_steps)
                    print('\nIRF/FEVD computed manually from VECM var_rep() (fallback).')
                except Exception as e_manual:
                    raise RuntimeError(f'Không thể tạo IRF/FEVD từ var_rep: {e_manual}') from e_manual

        else:
            # VAR results: statsmodels VARResults usually cung cấp .irf() và .fevd()
            try:
                res_irf_attr = getattr(results, 'irf', None)
                res_fevd_attr = getattr(results, 'fevd', None)

                if isinstance(res_irf_attr, np.ndarray):
                    irf = res_irf_attr
                    print('IRF obtained as numpy.ndarray from results.irf attribute (no call).')
                elif callable(res_irf_attr):
                    try:
                        irf_obj = res_irf_attr(forecast_steps)
                        irf = irf_obj
                        print('IRF object created by calling results.irf(...)')
                    except Exception as e_call:
                        print('Calling results.irf(...) failed:', e_call)

                if isinstance(res_fevd_attr, np.ndarray):
                    fevd = res_fevd_attr
                    print('FEVD obtained as numpy.ndarray from results.fevd attribute (no call).')
                elif callable(res_fevd_attr):
                    try:
                        fevd_obj = res_fevd_attr(forecast_steps)
                        fevd = fevd_obj
                        print('FEVD object created by calling results.fevd(...)')
                    except Exception as e_call:
                        print('Calling results.fevd(...) failed:', e_call)

                if irf is None or fevd is None:
                    var_coefs = getattr(results, 'coefs', None)
                    sigma_u = getattr(results, 'sigma_u', None)
                    if var_coefs is not None and sigma_u is not None:
                        psi, ortho_irf = compute_irf_from_var_coefs(var_coefs, sigma_u, forecast_steps)
                        irf = irf if irf is not None else ortho_irf
                        fevd = fevd if fevd is not None else compute_fevd_from_irf(ortho_irf, forecast_steps)
                        print('\nIRF/FEVD computed manually from VARResults.coefs.')
                    else:
                        print('Không có coefs/sigma_u để tính IRF/FEVD thủ công.')
            except Exception as e:
                raise RuntimeError('Lỗi khi tạo IRF/FEVD cho VAR: {}'.format(e))
    except Exception as e:
        print(f'Không thể tạo IRF/FEVD: {e}')
    # ----------------------------------------------------

    # ---------- SỬA LỖI TẠI ĐÂY (Forecast) -------------
    steps = forecast_steps
    forecast_df = pd.DataFrame()
    try:
        if use_vecm:
            exog_future = None
            if exog_data_for_var is not None and len(exog_data_for_var) > 0:
                last_exog = exog_data_for_var.values[-1]
                # create exog_future by repeating last observed exog (shape: steps x k)
                exog_future = np.tile(last_exog.reshape(1, -1), (steps, 1))

            # VECMResults.predict expects exog_fc for future exog in many statsmodels versions
            fc = None
            tried_signatures = []
            # try exog_fc keyword first (explicit)
            try:
                fc = results.predict(steps=steps, exog_fc=exog_future)
                tried_signatures.append('predict(steps=..., exog_fc=...)')
            except Exception as e1:
                # try exog keyword
                try:
                    fc = results.predict(steps=steps, exog=exog_future)
                    tried_signatures.append('predict(steps=..., exog=...)')
                except Exception as e2:
                    # try positional
                    try:
                        fc = results.predict(steps, exog_fc=exog_future)
                        tried_signatures.append('predict(steps, exog_fc=...)')
                    except Exception as e3:
                        try:
                            fc = results.predict(steps, exog=exog_future)
                            tried_signatures.append('predict(steps, exog=...)')
                        except Exception as e4:
                            # try without exog if exog_future is None
                            if exog_future is None:
                                try:
                                    fc = results.predict(steps=steps)
                                    tried_signatures.append('predict(steps=...)')
                                except Exception as e5:
                                    raise RuntimeError(f'Không thể gọi VECMResults.predict với bất kỳ signature nào: {e5}')
                            else:
                                # exog_future provided but no signature worked
                                raise RuntimeError(f'Không thể gọi VECMResults.predict với exog_fc/exog; thử các signatures: {[(s) for s in tried_signatures]}')

            if fc is None:
                raise RuntimeError('VECM predict returned None')

            # fc shape should be (steps, n_endog)
            last_time = df_diff['time'].iloc[-1]
            future_index = pd.date_range(last_time + pd.Timedelta(days=1), periods=steps, freq='D')

            forecast_cols = non_stationary_endogs if use_vecm else endog_cols
            forecast_df = pd.DataFrame(fc, columns=forecast_cols, index=future_index)
            print(f"\nForecast (next {steps} days) created (VECM).\n")

        else:
            # VAR case (robust)
            exog_future = None
            if exog_data_for_var is not None and len(exog_data_for_var) > 0:
                last_exog = exog_data_for_var.values[-1]
                exog_future = np.tile(last_exog.reshape(1, -1), (steps, 1))

            last_obs = endog_data_for_var.values[-results.k_ar:]
            try:
                fc = results.forecast(y=last_obs, steps=steps, exog_future=exog_future)
            except TypeError:
                try:
                    fc = results.forecast(y=last_obs, steps=steps, exog=exog_future)
                except Exception:
                    fc = results.forecast(y=last_obs, steps=steps)

            last_time = df_diff['time'].iloc[-1]
            future_index = pd.date_range(last_time + pd.Timedelta(days=1), periods=steps, freq='D')

            forecast_cols = non_stationary_endogs if use_vecm else endog_cols
            forecast_df = pd.DataFrame(fc, columns=forecast_cols, index=future_index)
            print(f"\nForecast (next {steps} days) created (VAR).\n")

    except Exception as e:
        print('Không thể dự báo:', e)
    # ----------------------------------------------------

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
