import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'Base_dir: {BASE_DIR}')

news_path = os.path.join(BASE_DIR, 'data', 'bronze', 'news', 'news00.json')
output_path = os.path.join(BASE_DIR, 'data', 'silver', 'news', 'news00.csv')

def read_file(file_path: str):
    """
    Đọc tệp JSON từ đường dẫn được cung cấp và trả về DataFrame.
    """
    try:
        df = pd.read_json(file_path)
        return df
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp tại {file_path}")
        return None

def process_dataset(dataset: pd.DataFrame):
    """
    Tổng hợp dữ liệu theo ngày và điền các ngày còn thiếu bằng giá trị của ngày trước đó.
    """
    # Chuyển đổi cột 'date' sang định dạng datetime
    dataset['date'] = pd.to_datetime(dataset['date'], format='%d-%m-%Y')

    # Nhóm theo 'date' và tính tổng 'mark'
    aggregated_df = dataset.groupby('date')['mark'].sum().reset_index()

    # Tạo một chuỗi ngày đầy đủ từ ngày đầu đến ngày cuối trong dữ liệu
    full_date_range = pd.date_range(start=aggregated_df['date'].min(), 
                                    end=aggregated_df['date'].max())
    
    # Đặt cột 'date' làm index để dễ dàng reindex
    aggregated_df.set_index('date', inplace=True)
    
    # Reindex DataFrame để thêm các ngày còn thiếu
    processed_df = aggregated_df.reindex(full_date_range)
    
    # Điền giá trị còn thiếu bằng giá trị từ ngày gần nhất trong quá khứ (ffill)
    processed_df.fillna(0, inplace=True)
    
    # Đổi tên cột index thành 'date' và reset index để đưa 'date' thành cột thông thường
    processed_df.index.name = 'date'
    processed_df.reset_index(inplace=True)
    
    # Đổi tên cột 'mark' thành 'total_mark' cho rõ ràng
    processed_df.rename(columns={'mark': 'total_mark'}, inplace=True)

    return processed_df

def main(start_date: str = None, end_date: str = None):
    """
    Chức năng chính để đọc dữ liệu, xử lý, lọc và in kết quả.
    """
    df = read_file(news_path)
    if df is None:
        return
    
    processed_df = process_dataset(df)
    print("\nDataFrame sau khi xử lý (tổng hợp, làm mịn):")
    print(processed_df)

    # Lọc dữ liệu theo ngày nếu có tham số
    if start_date:
        start_date_dt = pd.to_datetime(start_date)
        processed_df = processed_df[pd.to_datetime(processed_df['date'], format='%d-%m-%Y') >= start_date_dt]
    
    if end_date:
        end_date_dt = pd.to_datetime(end_date)
        processed_df = processed_df[pd.to_datetime(processed_df['date'], format='%d-%m-%Y') <= end_date_dt]
    
    print("\nDataFrame sau khi lọc:")
    print(processed_df)
    
    if processed_df.empty:
        # print("Không có dữ liệu nào trong khoảng thời gian đã chọn.")
        return None
    
    return processed_df

if __name__ == "__main__":
    # Ví dụ cách gọi hàm main() với các tùy chọn:
    # 1. Chạy mà không lọc ngày

    # 2020-10-01 đến 2023-06-30
    final_df = main('2020-10-01', '2023-06-30')

    # 2. Chạy với lọc ngày bắt đầu
    # final_df = main(start_date='10-08-2025')
    
    # 3. Chạy với lọc ngày kết thúc
    # final_df = main(end_date='15-08-2025')
    
    # 4. Chạy với cả hai ngày bắt đầu và kết thúc
    # final_df = main(start_date='10-08-2025', end_date='15-08-2025')
    
    if final_df is not None:
        final_df.to_csv(output_path, index=False)
        print(f"\nĐã lưu DataFrame vào {output_path}")