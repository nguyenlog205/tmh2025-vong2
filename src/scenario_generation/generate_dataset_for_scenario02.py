import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===================================================================================================
# Dữ liệu cho phần Credit Suisse
# Kịch bản 02: Xói mòn niềm tin từ từ
# ===================================================================================================

# --- Hàm 1A: Sinh dữ liệu NLP với xu hướng (TREND) ---
def generate_nlp_scores_with_trend(start_date: str, num_days: int, prob_news_event: float, 
                                   mean_score: float, std_dev_score: float, trend: float):
    """
    Sinh dữ liệu điểm sentiment có xu hướng thay đổi theo thời gian.
    'trend' là mức thay đổi của mean_score mỗi ngày.
    """
    dates = pd.date_range(start=start_date, periods=num_days, freq='D')
    data = []
    current_mean = mean_score
    for day in range(num_days):
        score = 0
        if np.random.rand() < prob_news_event:
            # Điểm score được sinh ra từ mean đã được điều chỉnh theo trend
            score = np.random.normal(loc=current_mean, scale=std_dev_score)
        
        data.append({'date': dates[day], 'nlp_score': score})
        
        # Cập nhật mean cho ngày tiếp theo
        current_mean += trend
        
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
def visualize_combined_data(df: pd.DataFrame, title: str):
    """Hàm trực quan hóa kết quả với tiêu đề tùy chỉnh."""
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
    ax2.plot(df['date'], df['cds_spread'], color='darkorange', linewidth=2.5, label='CDS Spread (Trục phải)')
    ax2.set_ylabel('CDS Spread (Basis Points)', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    
    plt.title(title, fontsize=16, fontweight='bold')
    fig.tight_layout()
    plt.show()

# =======================================================================
# == HÀM CHÍNH CHO KỊCH BẢN 02: XÓI MÒN NIỀM TIN TỪ TỪ                ==
# ========================================================================
def run_gradual_erosion_scenario():
    """
    Chạy mô phỏng cho Kịch bản 02:
    Niềm tin suy giảm từ từ do tin tức tiêu cực nhỏ, xuất hiện liên tiếp.
    - Điểm sentiment: Có xu hướng giảm dần theo thời gian.
    - Phí CDS: Có xu hướng tăng dần theo thời gian.
    """
    # --- Bước 1: Set tham số NLP để tạo điểm sentiment có xu hướng GIẢM DẦN ---
    print("Bước 1: Sinh dữ liệu NLP cho kịch bản xói mòn niềm tin...")
    nlp_params_erosion = {
        "start_date": '2021-01-01',
        "num_days": 365 * 2,
        "prob_news_event": 0.40,      # Tin tức tiêu cực xuất hiện khá thường xuyên
        "mean_score": 1.5,            # Điểm trung bình ban đầu vẫn còn dương
        "std_dev_score": 2.5,         # Biến động ở mức vừa phải
        "trend": -0.008               # QUAN TRỌNG: Mỗi ngày, mean_score giảm 0.008 điểm
    }
    nlp_df = generate_nlp_scores_with_trend(**nlp_params_erosion)
    
    # --- Bước 2: Set tham số CDS để tạo chỉ số có xu hướng TĂNG DẦN ---
    print("Bước 2: Sinh dữ liệu CDS tương ứng...")
    cds_params_erosion = {
        "omega": 2,                   # Rủi ro cơ sở cao hơn một chút
        "phi": 0.97,                  # Quán tính vẫn cao nhưng cho phép sự thay đổi diễn ra
        "beta": 1.8,                  # Tác động của tin tức (cả tốt và xấu) rất mạnh
        "sigma": 3.5,                 # Nhiễu thị trường cao hơn, bất ổn hơn
        "start_cds_value": 60         # Bắt đầu từ mức rủi ro an toàn
    }
    final_dataset = generate_cds_data(nlp_df, **cds_params_erosion)
    
    # --- Bước 3: Hiển thị kết quả ---
    print("\nMô phỏng hoàn tất! Xem trước 5 dòng dữ liệu cuối cùng:")
    print(final_dataset.tail())
    
    # Trực quan hóa kết quả
    title = 'Kịch Bản 02: Xói Mòn Niềm Tin Từ Từ'
    visualize_combined_data(final_dataset, title)

# === Chạy Kịch bản 02 ===
if __name__ == "__main__":
    run_gradual_erosion_scenario()