import pandas as pd
from bs4 import BeautifulSoup

# Tên file HTML đầu vào và file CSV đầu ra
html_input_filename = 'financial_report.html'
csv_output_filename = 'financial_report.csv'

# --- Bước 1: Đọc và tải nội dung của file HTML ---
try:
    with open(html_input_filename, 'r', encoding='utf-8') as f:
        html_content = f.read()
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{html_input_filename}'. Vui lòng đảm bảo file này tồn tại trong cùng thư mục.")
    exit()

# --- Bước 2: Dùng BeautifulSoup để phân tích (parse) HTML ---
soup = BeautifulSoup(html_content, 'html.parser')

# --- Bước 3: Trích xuất tiêu đề các cột (là các kỳ báo cáo) ---
# Tìm vùng chứa các tiêu đề cột dựa trên cấu trúc HTML
header_container = soup.find('div', class_='stickyContainer-MJytD_Lf')
time_periods = header_container.find_all('div', class_='container-OxVAcLqi')

# Duyệt qua từng phần tử để lấy tên cột
columns = []
for period in time_periods:
    quarter_div = period.find('div', class_='value-OxVAcLqi')
    month_year_div = period.find('div', class_='subvalue-OxVAcLqi')
    
    quarter = quarter_div.text.strip() if quarter_div else ''
    month_year = month_year_div.text.strip() if month_year_div else ''
    
    # Kết hợp Quý và Tháng/Năm để tạo tiêu đề cột rõ ràng
    if month_year:
        columns.append(f"{quarter} ({month_year})")
    elif quarter:
        columns.append(quarter)

# --- Bước 4: Trích xuất dữ liệu của từng hàng ---
# Tìm tất cả các hàng dữ liệu dựa trên thuộc tính 'data-name'
data_rows = soup.find_all('div', attrs={'data-name': True})

financial_data = []
for row in data_rows:
    # Lấy tên của chỉ số tài chính từ thuộc tính 'data-name'
    metric_name = row['data-name']
    
    # Tìm vùng chứa các giá trị trong hàng đó
    values_container = row.find('div', class_='values-C9MdAMrq')
    values = values_container.find_all('div', class_='value-OxVAcLqi')
    
    # Lấy giá trị text và làm sạch các ký tự unicode không mong muốn
    row_values = [v.text.strip().replace('\u202a', '').replace('\u202c', '') for v in values]
    
    # Thêm tên chỉ số vào đầu danh sách các giá trị của hàng
    financial_data.append([metric_name] + row_values)

# --- Bước 5: Sử dụng Pandas để tạo và cấu trúc bảng dữ liệu ---
# Tạo danh sách tên cột cho DataFrame, cột đầu tiên là 'Financial Metric'
df_columns = ['Financial Metric'] + columns

# Tạo DataFrame từ dữ liệu đã trích xuất
df = pd.DataFrame(financial_data, columns=df_columns)

# --- Bước 6: Xuất DataFrame ra file CSV ---
# Sử dụng to_csv để lưu DataFrame. 
# index=False để không ghi chỉ số của DataFrame vào file.
# encoding='utf-8-sig' để đảm bảo các ký tự tiếng Việt hiển thị đúng trong Excel.
df.to_csv(csv_output_filename, index=False, encoding='utf-8-sig')

print(f"Đã xử lý thành công! Dữ liệu đã được lưu vào file '{csv_output_filename}'.")