import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===================================================================================================
# Dữ liệu cho phần Credit Suisse
# Kịch bản 03: Cú sốc niềm tin đột ngột
# ===================================================================================================

# --- Hàm 1B: Sinh dữ liệu NLP với một cú sốc (SHOCK) ---
def generate_nlp_scores_with_shock(start_date: str, num_days: int, prob_news_event: float, 
                                   shock_day: int,
                                   mean_score_before: float, std_dev_before: float,
                                   mean_score_after: float, std_dev_after: float):
    """
    Sinh dữ liệu sentiment với một cú sốc xảy ra vào ngày 'shock_day'.
    """
    dates = pd.date_range(start=start_date, periods=num_days, freq='D')
    data = []
    for day in range(num_days):
        score = 0
        if np.random.rand() < prob_news_event:
            # Kiểm tra xem có phải giai đoạn sau cú sốc hay không
            if day < shock_day:
                # Giai đoạn trước khủng hoảng: tâm lý tích cực
                score = np.random.normal(loc=mean_score_before, scale=std_dev_before)
            else:
                # Giai đoạn sau khủng hoảng: tâm lý sụp đổ
                score = np.random.normal(loc=mean_score_after, scale=std_dev_after)
        data.append({'date': dates[day], 'nlp_score': score})
        
    df = pd.DataFrame(data)
    df['nlp_score'] = df['nlp_score'].round().astype(int)
    return df

# --- Hàm 2: Sinh dữ liệu CDS ---
def generate_cds_data(nlp_scores_df: pd.DataFrame, omega: float, phi: float, beta: float, sigma: float, start_cds_value: float):
    """Sinh dữ liệu CDS spread dựa trên điểm NLP."""
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

# --- Hàm 3: Trực quan hóa ---
def visualize_combined_data(df: pd.DataFrame, title: str, shock_day: int = None):
    """Hàm trực quan hóa kết quả, có thể vẽ thêm đường chỉ báo cú sốc."""
    print("\nĐang tạo biểu đồ kết quả...")
    sns.set_theme(style="whitegrid")
    df['date'] = pd.to_datetime(df['date'])
    
    fig, ax1 = plt.subplots(figsize=(15, 7))
    
    colors = ['g' if x >= 0 else 'r' for x in df['nlp_score']]
    ax1.bar(df['date'], df['nlp_score'], color=colors, alpha=0.5, width=1.2, label='Điểm NLP (Trục trái)')
    ax1.set_xlabel('Ngày')
    ax1.set_ylabel('Điểm Sentiment NLP', color='dimgray')
    ax1.tick_params(axis='y', labelcolor='dimgray')
    ax1.grid(False)

    ax2 = ax1.twinx()
    ax2.plot(df['date'], df['cds_spread'], color='red', linewidth=2.5, label='CDS Spread (Trục phải)')
    ax2.set_ylabel('CDS Spread (Basis Points)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Vẽ đường chỉ báo ngày xảy ra cú sốc
    if shock_day is not None:
        shock_date = df['date'].iloc[0] + pd.Timedelta(days=shock_day)
        ax1.axvline(x=shock_date, color='black', linestyle='--', linewidth=2, label=f'Ngày xảy ra khủng hoảng (Ngày {shock_day})')
        ax1.legend(loc='upper left')

    plt.title(title, fontsize=16, fontweight='bold')
    fig.tight_layout()
    plt.show()

# =======================================================================
# == HÀM CHÍNH CHO KỊCH BẢN 03: CÚ SỐC NIỀM TIN ĐỘT NGỘT               ==
# ========================================================================
def run_sudden_shock_scenario():
    """
    Chạy mô phỏng cho Kịch bản 03:
    Một sự kiện lớn gây sụp đổ niềm tin tức thời.
    - Điểm sentiment: Giảm mạnh đột ngột.
    - Phí CDS: Tăng vọt tức thời.
    """
    # --- Bước 1: Set tham số NLP để tạo một cú sốc tại ngày 500 ---
    print("Bước 1: Sinh dữ liệu NLP cho kịch bản cú sốc...")
    shock_day = 500
    nlp_params_shock = {
        "start_date": '2021-01-01',
        "num_days": 365 * 2,
        "prob_news_event": 0.45,
        "shock_day": shock_day,
        "mean_score_before": 2.0,       # Trước khủng hoảng, mọi thứ khá tốt
        "std_dev_before": 2.0,
        "mean_score_after": -4.0,       # QUAN TRỌNG: Sau khủng hoảng, niềm tin sụp đổ
        "std_dev_after": 2.5            # Sự hoảng loạn làm tăng biến động
    }
    nlp_df = generate_nlp_scores_with_shock(**nlp_params_shock)
    
    # --- Bước 2: Set tham số CDS để phản ứng mạnh với cú sốc ---
    print("Bước 2: Sinh dữ liệu CDS tương ứng...")
    cds_params_shock = {
        "omega": 1.5,
        "phi": 0.96,
        "beta": 1.5,                  # QUAN TRỌNG: Tác động của tin tức cực mạnh để tạo ra cú spike
        "sigma": 3.5,                 # QUAN TRỌNG: Thị trường cực kỳ bất ổn và hoảng loạn
        "start_cds_value": 55         # Bắt đầu từ mức rủi ro an toàn
    }
    final_dataset = generate_cds_data(nlp_df, **cds_params_shock)
    
    # --- Bước 3: Hiển thị kết quả ---
    print("\nMô phỏng hoàn tất! Xem trước 5 dòng dữ liệu cuối cùng:")
    print(final_dataset.tail())
    
    # Trực quan hóa kết quả
    title = 'Kịch Bản 03: Cú Sốc Niềm Tin Đột Ngột'
    visualize_combined_data(final_dataset, title, shock_day=shock_day)

# === Chạy Kịch bản 03 ===
if __name__ == "__main__":
    run_sudden_shock_scenario()