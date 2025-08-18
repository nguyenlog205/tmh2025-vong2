
import os as os
import pandas as pd
import time
# =========================================================
# Accessing news with Google GenAI
# =========================================================
import google.generativeai as genai
import logging

logging.basicConfig(
    level=logging.INFO,  # Chỉ ghi log từ cấp INFO trở lên (INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s' # Định dạng của log message
)

genai.configure(api_key="AIzaSyCPcIm450Gq_PG4IQSkex6Q-YI0sKsyLXA")

import itertools
from typing import List

def process_file_in_chunks(file_path: str, chunk_size: int = 4) -> List[List[str]]:
    """
    Đọc một file văn bản theo từng khối và trả về một list chứa các khối đó.

    Hàm này rất hiệu quả về bộ nhớ vì nó không đọc toàn bộ file cùng một lúc.

    Args:
        file_path (str): Đường dẫn đến file cần đọc.
        chunk_size (int, optional): Số dòng trong mỗi khối. Mặc định là 4.

    Returns:
        List[List[str]]: Một list, trong đó mỗi phần tử là một list 
                         chứa các dòng đã được xử lý (đã xóa ký tự xuống dòng).
                         Trả về một list rỗng nếu file không tồn tại hoặc có lỗi.
    """
    all_chunks = []  # List để lưu tất cả các khối tin
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            while True:
                # Dùng itertools.islice để lấy chunk_size dòng tiếp theo
                chunk = list(itertools.islice(f, chunk_size))
                
                # Nếu chunk rỗng, nghĩa là đã đọc hết file thì dừng lại
                if not chunk:
                    break
                
                # Xử lý: làm sạch ký tự xuống dòng '\n' ở cuối mỗi dòng
                processed_lines = [line.strip() for line in chunk]
                
                # Thêm khối đã xử lý vào list kết quả
                all_chunks.append(processed_lines)
                
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại đường dẫn '{file_path}'")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        
    return all_chunks

def clean_json_string(text):
    """
    Loại bỏ các chuỗi '```json' và '```' khỏi văn bản.

    Args:
        text (str): Chuỗi văn bản cần làm sạch.

    Returns:
        str: Chuỗi văn bản đã được làm sạch.
    """
    if isinstance(text, str):
        # Loại bỏ '```json'
        cleaned_text = text.replace('```json', '').replace('```', '')
        return cleaned_text.strip() # Dùng .strip() để loại bỏ khoảng trắng thừa
    return text

def access_news(text: str) -> int:
    template = """
        Bạn hãy đóng vai là một nhà phân tích kinh tế - tài chính. 
        Nhiệm vụ: dựa vào đoạn văn được cung cấp, hãy đánh giá mức độ tích cực hoặc tiêu cực cho uy tín của ngân hàng Credit Suisse tại một thời gian xác định. 
        Chỉ trả về **một JSON object** theo format:

        {
            "title": "",
            "date": "",
            "description": "",
            "mark": 0
        }

        Trong đó:
        - "title": tiêu đề ngắn gọn, súc tích (tóm tắt nội dung chính)
        - "date": thời gian sự kiện, định dạng dd-mm-yyyy. 
            - Nếu không có ngày thì điền "00" ở vị trí ngày.  
            - Nếu không có tháng thì điền "00" ở vị trí tháng.  
            - Nếu không có năm thì để trống chuỗi "".
        - "description": mô tả tình huống
        - "mark": là số nguyên trong khoảng [-5; -4; -3; -2; -1; 0; 1; 2; 3; 4; 5]. Càng âm càng tiêu cực, càng dương càng tích cực.
        - Vui lòng trả về kết quả dưới dạng JSON thuần túy, không bao gồm các ký tự định dạng markdown như ```. Nếu như không liên quan tới credit Suisse thì trả về điểm 0.
        Bây giờ hãy phân tích đoạn văn sau:
        Đoạn văn
        """
    
    prompt = template + text + """
    Output: 
    """
    

    response = genai.GenerativeModel("gemini-2.5-flash").generate_content(
       prompt
    )
    # print(response.text)
    time.sleep(2)
    
    return clean_json_string(response.text)

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f'Base_dir: {BASE_DIR}')

    try:
        logging.info('Start reading input file!')
        news_dataset_path = os.path.join(BASE_DIR, 'data', 'bronze', 'news', 'news00.txt')
        list_of_news = process_file_in_chunks(file_path=news_dataset_path, chunk_size=4)
        logging.info('Imported file successfully!')
    except FileNotFoundError:
        logging.error(f'File not found at {news_dataset_path}!')
        return # Dừng chương trình nếu không tìm thấy file
    except Exception as e:
        logging.error(f'An unexpected error occurred: {e}')
        return

    logging.info(f'Total news pieces to process: {len(list_of_news)}')
    logging.info('--- Start processing news ---')

    # Danh sách tạm thời để lưu 50 tin
    temp_dataset = []
    # Biến đếm số file đã được lưu
    file_count = 0
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    output_dir = os.path.join(BASE_DIR, 'data', 'silver', 'news')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, item in enumerate(list_of_news):
        # Lấy nội dung tin tức
        news = item[0] + item[1] + item[2]
        
        # Gọi API để xử lý tin tức
        # Thêm time.sleep để tránh lỗi ResourceExhausted
        response = access_news(news) 
        time.sleep(2) # Đợi 2 giây giữa mỗi lần gọi API
        
        # Nếu response không hợp lệ, bỏ qua
        if response == 'None' or response is None:
            logging.warning(f'Skipping news item {i} due to invalid response.')
            continue

        # Thêm kết quả vào danh sách tạm thời
        temp_dataset.append(response)

        # Cứ mỗi 50 tin, lưu và reset danh sách
        if len(temp_dataset) >= 50:
            file_name = f'news{str(file_count).zfill(2)}.csv'
            processed_path = os.path.join(output_dir, file_name)
            
            df = pd.DataFrame(temp_dataset)
            df.to_csv(processed_path, index=False)
            
            logging.info(f'Saved {len(temp_dataset)} pieces of news to {processed_path}')
            
            # Xóa danh sách tạm thời và tăng biến đếm
            temp_dataset = []
            file_count += 1
    
    # Xử lý những tin tức còn sót lại (nếu có)
    if temp_dataset:
        file_name = f'news{str(file_count).zfill(2)}.csv'
        processed_path = os.path.join(output_dir, file_name)
        
        df = pd.DataFrame(temp_dataset)
        df.to_csv(processed_path, index=False)
        
        logging.info(f'Saved remaining {len(temp_dataset)} pieces of news to {processed_path}')
    
    logging.info('--- Finished processing all news ---')
    logging.info(f'Total CSV files created: {file_count + (1 if temp_dataset else 0)}')
    return None

# main()

def check():
    example01 = """
    ```json
    hello
    ```
    """
    print(example01)
    print(clean_json_string(example01))
    print('================================')

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f'Base_dir: {BASE_DIR}')
    processed_path = os.path.join(BASE_DIR, 'data', 'silver', 'news')
    if os.path.exists(processed_path):
        print(f"Đường dẫn '{processed_path}' đã tồn tại.")
    else:
        print(f"Đường dẫn '{processed_path}' chưa tồn tại.")
    
    return None

if __name__ == '__main__':
    main()