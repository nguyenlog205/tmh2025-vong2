import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)

news_path = os.path.join(BASE_DIR, 'data', 'bronze', 'news', 'news00.json')
output_path = os.path.join(BASE_DIR, 'data', 'silver', 'news', 'news00.csv')

def read_file(file_path: str):
    """
    Đọc tệp JSON từ đường dẫn được cung cấp và trả về DataFrame.
    """
    return pd.read_json(file_path)

def process_dataset(dataset: pd.DataFrame):
    """
    Tổng hợp dữ liệu theo ngày và tính tổng cột 'mark'.

    Args:
        dataset (pd.DataFrame): DataFrame đầu vào chứa các cột 'date' và 'mark'.

    Returns:
        pd.DataFrame: DataFrame mới với mỗi ngày xuất hiện một lần duy nhất
                      và tổng 'mark' cho ngày đó.
    """
    # Chuyển đổi cột 'date' sang định dạng datetime
    dataset['date'] = pd.to_datetime(dataset['date'], format='%d-%m-%Y')

    # Nhóm theo 'date' và tính tổng 'mark'
    aggregated_df = dataset.groupby('date')['mark'].sum().reset_index()

    # Đặt tên lại cột 'date' nếu cần thiết (mặc định đã là 'date')
    # aggregated_df.columns = ['date', 'total_mark']

    return aggregated_df

def main():
    """
    Chức năng chính để đọc dữ liệu, xử lý và in kết quả.
    """
    df = read_file(news_path)
    print("DataFrame ban đầu:")
    print(df)

    processed_df = process_dataset(df)
    print("\nDataFrame sau khi xử lý (tổng hợp theo ngày):")
    print(processed_df)
    return processed_df

if __name__ == "__main__":
    df = main()
    df.to_csv(output_path, index=False)