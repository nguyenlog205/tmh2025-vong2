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
def visualize_nlp_scores(df: pd.DataFrame, title: str):
    """Hàm trực quan hóa chỉ riêng điểm NLP."""
    print("\nĐang tạo biểu đồ kết quả NLP...")
    sns.set_theme(style="whitegrid")
    df['date'] = pd.to_datetime(df['date'])
    
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Tạo màu sắc dựa trên giá trị dương hoặc âm
    colors = ['g' if x >= 0 else 'r' for x in df['nlp_score']]
    
    # Vẽ biểu đồ cột
    ax.bar(df['date'], df['nlp_score'], color=colors, alpha=0.7, width=1.0, label='Điểm NLP')
    
    # Thêm đường tham chiếu tại y=0
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    # Đặt nhãn và tiêu đề
    ax.set_xlabel('Ngày')
    ax.set_ylabel('Điểm Sentiment NLP')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
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