


# =========================================================
# Accessing news with Google GenAI
# =========================================================
import google.generativeai as genai

# Khởi tạo client
genai.configure(api_key="AIzaSyCsWZjclPxavDFcwdZEPz1HJad9u9pzdSk")

# Gọi model
response = genai.GenerativeModel("gemini-2.5-flash").generate_content(
    "Explain how AI works in a few words"
)

print(response.text)


def access_news(text: str) -> int:

    return {
        'title': '',
        'date': '',
        'description': '',
        'mark': 0
    }
    return None
