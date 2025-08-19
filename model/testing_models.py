"""
Bộ công cụ các Hàm Kiểm định Thống kê (testing_models.py)
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.vector_ar.vecm import coint_johansen # Thêm import cho Johansen test
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


def perform_adf_test(series: pd.Series, series_name: str):
    """
    Thực hiện kiểm định ADF. Trả về p-value.
    In ra kết quả tóm tắt.
    """
    try:
        adf_test = adfuller(series.dropna(), autolag='AIC')
    except Exception as e:
        print(f"Lỗi khi chạy ADF cho {series_name}: {e}")
        return np.nan

    adf_output = pd.Series(adf_test[0:4],
                           index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in adf_test[4].items():
        adf_output[f'Critical Value ({key})'] = value
    p_value = adf_test[1]
    return p_value


def perform_granger_causality_test(data: pd.DataFrame, response_var: str, predictor_var: str, max_lag: int = 4):
    """
    Thực hiện kiểm định Granger causality giữa predictor_var -> response_var.
    Trả về dict: {lag: p-value}, và boolean (True nếu có ít nhất 1 lag p<=0.05).
    Note: data phải đã dừng (hoặc ở trạng thái đã chuẩn bị để test).
    """
    print(f"\n--- Kiểm định Granger: '{predictor_var}' -> '{response_var}' (max_lag={max_lag}) ---")
    test_data = data[[response_var, predictor_var]].dropna()
    if test_data.shape[0] < max_lag + 2:
        print("Dữ liệu không đủ dài để kiểm định Granger.")
        return {}, False

    try:
        gc_test = grangercausalitytests(test_data[[response_var, predictor_var]], maxlag=max_lag, verbose=False)
    except Exception as e:
        print(f"Lỗi khi chạy Granger: {e}")
        return {}, False

    lag_pvals = {}
    any_sig = False
    for lag in range(1, max_lag + 1):
        try:
            p_value = gc_test[lag][0]['ssr_ftest'][1]
        except Exception:
            try:
                p_value = gc_test[lag][0]['ssr_chi2test'][1]
            except Exception:
                p_value = np.nan
        lag_pvals[lag] = p_value
        if (not np.isnan(p_value)) and (p_value <= 0.05):
            any_sig = True

    for lag, p in lag_pvals.items():
        conclusion = "Có quan hệ nhân quả Granger (p<=0.05)" if (not np.isnan(p) and p <= 0.05) else "Không có (p>0.05 hoặc NaN)"
        print(f"Lag {lag}: p-value = {p} => {conclusion}")

    return lag_pvals, any_sig


def perform_ljungbox_test(residuals, lags: int = 10):
    """
    Kiểm định Ljung-Box. Hỗ trợ residuals là Series hoặc DataFrame.
    In kết quả và trả dict {col: pvalue}.
    """
    print(f"\n--- Kiểm định tự tương quan Ljung-Box (lags={lags}) ---")
    results = {}
    if isinstance(residuals, pd.Series):
        lb = acorr_ljungbox(residuals.dropna(), lags=[lags], return_df=True)
        p = lb['lb_pvalue'].iloc[0]
        print(lb)
        results[residuals.name if residuals.name is not None else 'resid'] = p
    else:
        for col in residuals.columns:
            series = residuals[col].dropna()
            if len(series) < lags + 2:
                print(f"Column {col}: dữ liệu không đủ cho Ljung-Box (bỏ qua).")
                results[col] = np.nan
                continue
            lb = acorr_ljungbox(series, lags=[lags], return_df=True)
            p = lb['lb_pvalue'].iloc[0]
            print(f"{col}:\n{lb}")
            results[col] = p

    for k, p in results.items():
        if not np.isnan(p) and p <= 0.05:
            print(f"=> {k}: Bác bỏ H0 (phần dư có tự tương quan), p={p:.4f}")
        else:
            print(f"=> {k}: Không bác bỏ H0 (không tìm thấy tự tương quan rõ), p={p if not np.isnan(p) else 'NaN'}")

    return results


def calculate_vif(X: pd.DataFrame):
    """
    Tính VIF cho các biến (X) - X không chứa NaN & phải là numeric.
    Trả DataFrame gồm feature và VIF.
    """
    print("\n--- Hệ số Phóng đại Phương sai (VIF) ---")
    X_num = X.select_dtypes(include=[np.number]).dropna()
    if X_num.shape[1] == 0:
        print("Không có biến numeric để tính VIF.")
        return pd.DataFrame(columns=['feature', 'VIF'])

    X_with_const = X_num.copy()
    X_with_const['const'] = 1.0

    vif_vals = []
    for i in range(len(X_with_const.columns)):
        try:
            vif_vals.append(variance_inflation_factor(X_with_const.values, i))
        except Exception:
            vif_vals.append(np.nan)

    vif_df = pd.DataFrame({'feature': X_with_const.columns, 'VIF': vif_vals})
    vif_df = vif_df[vif_df['feature'] != 'const'].reset_index(drop=True)
    print(vif_df.round(3))
    return vif_df


def perform_jarque_bera_test(residuals: pd.DataFrame):
    """
    Kiểm định Jarque-Bera cho residuals (DataFrame).
    Trả dict {col: (stat, p)}.
    """
    print("\n--- Kiểm định phân phối chuẩn Jarque-Bera ---")
    results = {}
    if isinstance(residuals, pd.Series):
        residuals = residuals.to_frame()

    for col in residuals.columns:
        try:
            stat, p_value, _, _ = jarque_bera(residuals[col].dropna())
            results[col] = (stat, p_value)
            conclusion = "Không tuân theo phân phối chuẩn (Bác bỏ H0)" if p_value <= 0.05 else "Tuân theo phân phối chuẩn"
            print(f"{col}: JB stat = {stat:.3f}, p-value = {p_value:.4f} => {conclusion}")
        except Exception as e:
            print(f"Lỗi JB cho {col}: {e}")
            results[col] = (np.nan, np.nan)

    return results


def perform_pca_on_group(X: pd.DataFrame, variance_threshold: float = 0.90, prefix: str = "PCA"):
    """
    Thực hiện PCA trên DataFrame X (các biến numeric, không chứa NaN).
    - Chuẩn hoá (StandardScaler) trước PCA.
    - Chọn số component tối thiểu sao cho explained_variance_ratio_.cumsum() >= variance_threshold.
    Trả về:
      - pca_df: DataFrame các thành phần chính (index giữ nguyên) với tên cột prefix_1, prefix_2, ...
      - pca_object: đối tượng sklearn PCA đã fit
      - scaler: StandardScaler đã fit (để có thể inverse nếu cần)
    """
    X_num = X.select_dtypes(include=[np.number]).copy()
    if X_num.shape[1] < 2:
        # Nếu chỉ 0 or 1 biến, PCA không cần thiết/không thực hiện
        print("PCA bỏ qua: nhóm có <2 biến numeric.")
        return X_num.copy(), None, None

    # Drop NaN rows (PCA yêu cầu dữ liệu hoàn chỉnh)
    X_num = X_num.dropna()
    if X_num.shape[0] == 0:
        print("PCA bỏ qua: không có hàng hợp lệ sau dropna.")
        return pd.DataFrame(index=X.index), None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num.values)

    pca = PCA()
    pca.fit(X_scaled)

    explained = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(explained, variance_threshold) + 1)
    n_components = max(1, min(n_components, X_num.shape[1]))

    # refit with chosen n_components
    pca = PCA(n_components=n_components)
    comps = pca.fit_transform(X_scaled)

    col_names = [f"{prefix}_{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(comps, index=X_num.index, columns=col_names)

    print(f"PCA: giảm {X_num.shape[1]} -> {n_components} components, explained variance = {explained[n_components-1]:.3f}")

    return pca_df, pca, scaler

# *** HÀM MỚI ĐƯỢC THÊM VÀO ***
def perform_johansen_cointegration_test(df: pd.DataFrame, det_order=0, k_ar_diff=1):
    """
    Thực hiện kiểm định đồng tích hợp Johansen.
    df: DataFrame chứa các biến không dừng (ở dạng gốc, chưa sai phân).
    Trả về số lượng quan hệ đồng tích hợp (cointegration rank).
    """
    print("\n--- Kiểm định Đồng tích hợp Johansen ---")
    if df.shape[1] < 2:
        print("Cần ít nhất 2 biến để thực hiện kiểm định Johansen. Bỏ qua.")
        return 0
    
    # Đảm bảo không có giá trị NaN
    df_test = df.dropna()
    if df_test.shape[0] < 10: # Cần đủ dữ liệu
        print("Không đủ dữ liệu sau khi loại bỏ NaN để kiểm định Johansen. Bỏ qua.")
        return 0

    try:
        johansen_result = coint_johansen(df_test, det_order, k_ar_diff)
    except Exception as e:
        print(f"Lỗi khi chạy kiểm định Johansen: {e}")
        return 0

    trace_stat = johansen_result.lr1
    crit_vals = johansen_result.cvt  # Critical values (90%, 95%, 99%)

    print("Trace Statistic:", trace_stat)
    print("Critical Values (90%, 95%, 99%):")
    print(crit_vals)

    # Xác định rank (số quan hệ đồng tích hợp)
    rank = 0
    for i in range(len(trace_stat)):
        # So sánh với mức ý nghĩa 95% (cột thứ 2)
        if trace_stat[i] > crit_vals[i, 1]:
            rank += 1
        else:
            break
            
    print(f"=> Kết luận: Tìm thấy {rank} quan hệ đồng tích hợp (ở mức ý nghĩa 5%).")
    return rank
