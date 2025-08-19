import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===================================================================================================
# Dữ liệu cho phần Credit Suisse
# Kịch bản 01: Cơ sở (Niềm tin cao và ổn định)
# ===================================================================================================

# --- Hàm 1: Sinh dữ liệu NLP ---
def generate_nlp_scores(start_date: str, num_days: int, prob_news_event: float, mean_score: float, std_dev_score: float):
    """Sinh dữ liệu điểm sentiment từ tin tức."""
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
        cds_series[t] = max(1, cds_today) # Đảm bảo CDS không âm
    result_df = nlp_scores_df.copy()
    result_df['cds_spread'] = cds_series.round(2)
    return result_df

# --- Hàm 3: Trực quan hóa (Đã cập nhật để nhận tiêu đề) ---
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
    ax2.plot(df['date'], df['cds_spread'], color='royalblue', linewidth=2.5, label='CDS Spread (Trục phải)')
    ax2.set_ylabel('CDS Spread (Basis Points)', color='royalblue')
    ax2.tick_params(axis='y', labelcolor='royalblue')
    
    plt.title(title, fontsize=16, fontweight='bold')
    fig.tight_layout()
    plt.show()

# =======================================================================
# == HÀM CHÍNH CHO KỊCH BẢN 01: CƠ SỞ (BASELINE)                       ==
# ========================================================================
def run_baseline_scenario():
    """
    Chạy mô phỏng cho Kịch bản 01:
    Giả định Credit Suisse duy trì vị thế và niềm tin cao, ổn định.
    - Điểm sentiment: Cao và ít biến động.
    - Phí CDS: Thấp và ổn định.
    """
    # --- Bước 1: Set tham số NLP để tạo điểm sentiment CAO và ỔN ĐỊNH ---
    print("Bước 1: Sinh dữ liệu NLP cho kịch bản cơ sở...")
    nlp_params_baseline = {
        "start_date": '2021-01-01',
        "num_days": 365 * 2,
        "prob_news_event": 0.25,      # Ít tin tức gây nhiễu
        "mean_score": 3.0,            # QUAN TRỌNG: Điểm trung bình rất tích cực
        "std_dev_score": 1.5          # QUAN TRỌNG: Biến động rất thấp, đảm bảo tính ổn định
    }
    nlp_df = generate_nlp_scores(**nlp_params_baseline)
    
    # --- Bước 2: Set tham số CDS để tạo chỉ số THẤP và ỔN ĐỊNH ---
    print("Bước 2: Sinh dữ liệu CDS tương ứng...")
    cds_params_baseline = {
        "omega": 1,                   # Rủi ro cơ sở không đáng kể
        "phi": 0.98,                  # QUAN TRỌNG: Quán tính rất cao, giúp CDS đi ngang và ổn định
        "beta": 1.0,                  # Tin tức có tác động vừa phải
        "sigma": 1.0,                 # QUAN TRỌNG: Nhiễu thị trường rất thấp
        "start_cds_value": 50         # Bắt đầu từ mức rủi ro rất thấp
    }
    final_dataset = generate_cds_data(nlp_df, **cds_params_baseline)
    
    # --- Bước 3: Hiển thị kết quả ---
    print("\nMô phỏng hoàn tất! Xem trước 5 dòng dữ liệu cuối cùng:")
    print(final_dataset.tail())
    
    # Trực quan hóa kết quả
    title = 'Kịch Bản 01: Cơ Sở - Niềm Tin Cao & Ổn Định'
    visualize_combined_data(final_dataset, title)

# === Chạy Kịch bản 01 ===
if __name__ == "__main__":
    run_baseline_scenario()