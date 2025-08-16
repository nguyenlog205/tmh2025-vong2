


# =========================================================
# Accessing news with Google GenAI
# =========================================================
import google.generativeai as genai

genai.configure(api_key="AIzaSyCsWZjclPxavDFcwdZEPz1HJad9u9pzdSk")

with open("example.txt", "r", encoding="utf-8") as f:
    example = f.read()

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
        - "mark": -1 nếu tiêu cực, 0 nếu trung lập, +1 nếu tích cực

        Dưới đây là vài ví dụ (few-shot):

        [Example 1]
        Đoạn văn:
        "Tháng 3/2023, Credit Suisse phải nhận gói cứu trợ khẩn cấp từ Ngân hàng Quốc gia Thụy Sĩ nhằm duy trì thanh khoản."
        Kết quả:
        {
        "title": "Credit Suisse nhận gói cứu trợ khẩn cấp",
        "date": "03/2023",
        "description": "Ngân hàng buộc phải nhờ đến hỗ trợ từ ngân hàng trung ương để duy trì thanh khoản.",
        "mark": -1
        }

        [Example 2]
        Đoạn văn:
        "Credit Suisse công bố lợi nhuận quý IV/2021 vượt kỳ vọng của giới phân tích, củng cố niềm tin vào khả năng tái cấu trúc."
        Kết quả:
        {
        "title": "Lợi nhuận vượt kỳ vọng quý IV/2021",
        "date": "Q4/2021",
        "description": "Kết quả kinh doanh khả quan giúp củng cố niềm tin vào quá trình tái cấu trúc ngân hàng.",
        "mark": 1
        }

        [Example 3]
        Đoạn văn:
        "Credit Suisse thông báo thay đổi ban lãnh đạo cấp cao nhưng không đưa ra các chi tiết cụ thể về chiến lược mới."
        Kết quả:
        {
        "title": "Thay đổi lãnh đạo cấp cao",
        "date": "",
        "description": "Ngân hàng thay đổi nhân sự cấp cao nhưng chưa làm rõ chiến lược mới.",
        "mark": 0
        }

        --- 
        Bây giờ hãy phân tích đoạn văn sau:
        Đoạn văn
        """
    
    prompt = template + text + """
    Output: 
    """
    

    response = genai.GenerativeModel("gemini-2.5-flash").generate_content(
       prompt
    )
    print(response.text)

    return response.text

access_news(example)