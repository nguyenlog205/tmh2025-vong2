import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# ===================================================================================================
# Dữ liệu cho phần 

# --- Hàm 1: Sinh dữ liệu NLP (Đã sửa để không bắt buộc lưu file) ---
def generate_nlp_scores(start_date: str, num_days: int, prob_news_event: float, mean_score: float, std_dev_score: float):
    dates = pd.date_range(start=start_date, periods=num_days, freq='D')
    data = []
    for date in dates:
        score = 0
        if np.random.rand() < prob_news_event:
            score = np.random.normal(loc=mean_score, scale=std_dev_score)
        data.append({'date': date, 'nlp_score': score})
    df = pd.DataFrame(data)
    df['nlp_score'] = df['nlp_score'].round().astype(int)
    return df

# --- Hàm 2: Sinh dữ liệu CDS (Giữ nguyên) ---
def generate_cds_data(nlp_scores_df: pd.DataFrame, omega: float, phi: float, beta: float, sigma: float, start_cds_value: float):
    nlp_scores = nlp_scores_df['nlp_score'].values
    num_days = len(nlp_scores)
    cds_series = np.zeros(num_days)
    cds_series[0] = start_cds_value
    for t in range(1, num_days):
        cds_yesterday = cds_series[t-1]
        nlp_today = nlp_scores[t]
        random_shock = np.random.normal(0, sigma)
        cds_today = omega + (phi * cds_yesterday) - (beta * nlp_today) + random_shock
        cds_series[t] = max(1, cds_today)
    result_df = nlp_scores_df.copy()
    result_df['cds_spread'] = cds_series.round(2)
    return result_df