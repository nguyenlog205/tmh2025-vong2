import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===================================================================================================
# Dữ liệu cho phần Credit Suisse
# Kịch bản 01: Chỉ sinh điểm NLP, xuất file và visualize
# ===================================================================================================

# --- Hàm 1: Sinh dữ liệu NLP (Không đổi) ---
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
    # Giữ lại số thực để khớp với format yêu cầu (ví dụ: 2.0, -1.0)
    df['nlp_score'] = df['nlp_score'].round(1) 
    return df

# ==================================================================
# == HÀM 2 (MỚI): Trực quan hóa chỉ điểm NLP                       ==
# ==================================================================
def visualize_nlp_scores(df: pd.DataFrame, title: str, shock_day: int = None, window: int = 20):
    """
    Trực quan hóa điểm NLP bằng đường trung bình trượt để làm rõ xu hướng.
    """
    print(f"\nĐang tạo biểu đồ với đường trung bình trượt (cửa sổ {window} ngày)...")
    sns.set_theme(style="whitegrid")
    df['date'] = pd.to_datetime(df['date'])
    
    # Tính toán đường trung bình trượt
    df['rolling_avg'] = df['nlp_score'].rolling(window=window, center=True, min_periods=1).mean()
    
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Vẽ các điểm dữ liệu gốc dưới dạng cột mờ
    # colors = ['g' if x >= 0 else 'r' for x in df['nlp_score']]
    # ax.bar(df['date'], df['nlp_score'], color=colors, alpha=0.2, width=1.0, label='Điểm NLP Gốc (hàng ngày)')
    
    # Vẽ đường trung bình trượt
    ax.plot(df['date'], df['rolling_avg'], color='blue', linewidth=2.5, label=f'Trung bình trượt {window} ngày')
    
    # Thêm các đường và nhãn
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    if shock_day is not None:
        shock_date = df['date'].iloc[0] + pd.Timedelta(days=shock_day)
        ax.axvline(x=shock_date, color='black', linestyle='--', linewidth=2, label=f'Ngày xảy ra khủng hoảng (Ngày {shock_day})')

    ax.set_xlabel('Ngày')
    ax.set_ylabel('Điểm Sentiment NLP')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend()
    
    fig.tight_layout()
    plt.show()


# =======================================================================
# == HÀM CHÍNH CHO KỊCH BẢN 01 (CHỈ TÍNH NLP)                          ==
# ========================================================================
def run_nlp_only_scenario():
    """
    Chạy mô phỏng chỉ để sinh điểm NLP, xuất file CSV và trực quan hóa.
    """
    np.random.seed(101)
    # --- Bước 1: Set tham số NLP để tạo điểm sentiment ---
    print("Bước 1: Sinh dữ liệu NLP cho kịch bản cơ sở...")
    nlp_params_baseline = {
        "start_date": '2020-10-01',
        "num_days": 1003,
        "prob_news_event": 0.05,
        "mean_score": 2,
        "std_dev_score": 3
    }
    nlp_df = generate_nlp_scores(**nlp_params_baseline)
    
    # --- Bước 2: Xử lý và xuất điểm NLP ra file CSV ---
    print("\nBước 2: Chuẩn bị và xuất dữ liệu NLP ra file CSV...")
    output_df = nlp_df.copy()
    output_df.rename(columns={'nlp_score': 'total_mark'}, inplace=True)
    output_df['date'] = output_df['date'].dt.strftime('%Y-%m-%d')
    
    file_name = r'src\scenario_generation\scenario_01_nlp.csv'
    output_df.to_csv(file_name, index=False)
    
    print(f"Đã xuất thành công dữ liệu ra file: {file_name}")
    print("Xem trước 5 dòng dữ liệu trong file CSV:")
    print(output_df.head())

    # ==================================================================
    # == BƯỚC 3 (MỚI): Trực quan hóa kết quả NLP                      ==
    # ==================================================================
    visualize_nlp_scores(nlp_df, title='Biểu đồ Điểm Sentiment NLP (Kịch bản Cơ sở)')


# === Chạy Kịch bản chỉ sinh NLP ===
if __name__ == "__main__":
    run_nlp_only_scenario()