import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===================================================================================================
# Dữ liệu cho phần Credit Suisse
# Kịch bản 02: Chỉ sinh điểm NLP, xuất file và visualize
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
            score = np.random.normal(loc=current_mean, scale=std_dev_score)
        
        data.append({'date': dates[day], 'nlp_score': score})
        
        # Cập nhật mean cho ngày tiếp theo
        current_mean += trend
        
    df = pd.DataFrame(data)
    df['nlp_score'] = df['nlp_score'].round(1)
    return df

# --- Hàm 2 (MỚI): Trực quan hóa chỉ điểm NLP ---
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
# == HÀM CHÍNH CHO KỊCH BẢN 02 (CHỈ TÍNH NLP)                          ==
# ========================================================================
def run_gradual_erosion_nlp_only():
    """
    Chạy mô phỏng chỉ để sinh điểm NLP có xu hướng giảm, xuất file và trực quan hóa.
    """
    np.random.seed(202)
    # --- Bước 1: Set tham số NLP để tạo điểm sentiment có xu hướng GIẢM DẦN ---
    print("Bước 1: Sinh dữ liệu NLP cho kịch bản xói mòn niềm tin...")
    nlp_params_erosion = {
        "start_date": '2020-10-01',
        "num_days": 1003,
        "prob_news_event": 0.05,
        "mean_score": 1.5,
        "std_dev_score": 2.5,
        "trend": -0.008
    }
    nlp_df = generate_nlp_scores_with_trend(**nlp_params_erosion)
    
    # --- Bước 2: Xử lý và xuất điểm NLP ra file CSV ---
    print("\nBước 2: Chuẩn bị và xuất dữ liệu NLP ra file CSV...")
    output_df = nlp_df.copy()
    output_df.rename(columns={'nlp_score': 'total_mark'}, inplace=True)
    output_df['date'] = output_df['date'].dt.strftime('%Y-%m-%d')
    
    file_name = r'src\scenario_generation\scenario_02_nlp.csv'
    output_df.to_csv(file_name, index=False)
    
    print(f"Đã xuất thành công dữ liệu ra file: {file_name}")
    print("Xem trước 5 dòng dữ liệu trong file CSV:")
    print(output_df.head())

    # --- Bước 3: Trực quan hóa kết quả NLP ---
    visualize_nlp_scores(nlp_df, title='Biểu đồ Điểm Sentiment NLP (Kịch bản Xói mòn từ từ)')


# === Chạy Kịch bản 02 chỉ sinh NLP ===
if __name__ == "__main__":
    run_gradual_erosion_nlp_only()