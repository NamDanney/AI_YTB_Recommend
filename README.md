Flask==2.0.1
google-api-python-client==2.15.0
numpy==1.21.0
scikit-learn==0.24.2
tensorflow==2.5.0

# Hệ thống gợi ý video YouTube

Dự án này là một hệ thống gợi ý video YouTube sử dụng lọc nội dung dựa trên mạng nơ-ron tích chập (CNN) để gợi ý video cho người dùng dựa trên lịch sử xem và nội dung video.

## Cấu trúc dự án
.
├── App/
│   ├── __init__.py
│   ├── routes.py
│   ├── static/
│   │   └── styles.css
│   └── templates/
│       ├── index.html
│       ├── search_results.html
│       └── watch.html
├── Data/
│   ├── recommended_videos.txt
│   ├── user_data.txt
│   └── video_data.txt
├── scripts/
│   ├── check_model.py
│   ├── data_collection.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── predict.py
│   ├── train_model.py
│   └── utils.py
├── best_model.keras
├── cnn_content_filtering_model.keras
├── error.log
├── run.py
├── scaler.pkl
├── vectorizer.pkl
└── README.md


## Yêu cầu

- Python 3.8 hoặc cao hơn
- pip (trình quản lý gói Python)

## Cài đặt

1. **Clone repository:**

    ```sh
    git clone https://github.com/yourusername/ytb_recommend.git
    cd ytb_recommend
    ```

2. **Tạo môi trường ảo:**

    ```sh
    python -m venv venv
    ```

3. **Kích hoạt môi trường ảo:**

    - Trên Windows:

        ```sh
        venv\Scripts\activate
        ```

    - Trên macOS/Linux:

        ```sh
        source venv/bin/activate
        ```

4. **Cài đặt các gói cần thiết:**

    ```sh
    pip install -r requirements.txt
    ```

## Cấu hình

1. **Thiết lập YouTube Data API:**

    - Lấy API key từ [Google Developers Console](https://console.developers.google/).
    - Thay thế `'YOUR_API_KEY'` trong file [routes.py](http://_vscodecontentref_/#%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22c%3A%5C%5CUsers%5C%5Cadmin%5C%5CDocuments%5C%5CYTB_Recommend%5C%5CApp%5C%5Croutes.py%22%2C%22_sep%22%3A1%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fadmin%2FDocuments%2FYTB_Recommend%2FApp%2Froutes.py%22%2C%22scheme%22%3A%22file%22%7D%7D) bằng API key thực tế của bạn.

## Chạy dự án

1. **Chạy ứng dụng Flask:**

    ```sh
    python run.py
    ```

2. **Truy cập ứng dụng:**

    Mở trình duyệt web và truy cập `http://127.0.0.1:5000`.

## Tính năng của dự án

1. **Tìm kiếm video:**

    - Sử dụng thanh tìm kiếm để tìm kiếm video bằng YouTube Data API.
    - Kết quả tìm kiếm sẽ được hiển thị trên trang [`search_results.html`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fadmin%2FDocuments%2FYTB_Recommend%2FApp%2Froutes.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A31%2C%22character%22%3A4%7D%7D%5D%2C%2211dc2302-46f1-44fa-9373-a4ae6d5a5f74%22%5D "Go to definition").

2. **Xem video:**

    - Nhấp vào một video từ kết quả tìm kiếm để xem.
    - Chi tiết video sẽ được hiển thị trên trang [`watch.html`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fadmin%2FDocuments%2FYTB_Recommend%2FApp%2Froutes.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A68%2C%22character%22%3A4%7D%7D%5D%2C%2211dc2302-46f1-44fa-9373-a4ae6d5a5f74%22%5D "Go to definition").

3. **Ghi lại thời gian xem:**

    - Ứng dụng ghi lại thời gian xem của mỗi video cho mỗi người dùng.

4. **Gợi ý video:**

    - Ứng dụng gợi ý video dựa trên lịch sử xem của người dùng và nội dung video.
    - Các gợi ý sẽ được hiển thị trên trang [`index.html`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fadmin%2FDocuments%2FYTB_Recommend%2FApp%2Froutes.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A41%2C%22character%22%3A4%7D%7D%5D%2C%2211dc2302-46f1-44fa-9373-a4ae6d5a5f74%22%5D "Go to definition").

## Thu thập dữ liệu

- Ứng dụng thu thập dữ liệu video (tiêu đề, ID, tags, lượt xem, lượt thích, thời lượng) bằng YouTube Data API.
- Dữ liệu hành vi người dùng (ID người dùng, ID video, thời gian xem) được ghi lại và lưu trữ trong [user_data.txt](http://_vscodecontentref_/#%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22c%3A%5C%5CUsers%5C%5Cadmin%5C%5CDocuments%5C%5CYTB_Recommend%5C%5CData%5C%5Cuser_data.txt%22%2C%22_sep%22%3A1%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fadmin%2FDocuments%2FYTB_Recommend%2FData%2Fuser_data.txt%22%2C%22scheme%22%3A%22file%22%7D%7D).

## Tiền xử lý dữ liệu

- Dữ liệu thu thập được tiền xử lý để chuyển đổi dữ liệu văn bản thành dạng số và chuẩn hóa các đặc trưng số.
- Dữ liệu được chia thành tập huấn luyện và tập kiểm tra.

## Huấn luyện mô hình

- Dự án sử dụng mạng nơ-ron tích chập (CNN) cho lọc nội dung dựa trên.
- Mô hình được huấn luyện bằng dữ liệu đã tiền xử lý và lưu trữ dưới dạng [cnn_content_filtering_model.keras](http://_vscodecontentref_/#%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22c%3A%5C%5CUsers%5C%5Cadmin%5C%5CDocuments%5C%5CYTB_Recommend%5C%5Ccnn_content_filtering_model.keras%22%2C%22_sep%22%3A1%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fadmin%2FDocuments%2FYTB_Recommend%2Fcnn_content_filtering_model.keras%22%2C%22scheme%22%3A%22file%22%7D%7D).

## Đánh giá

- Mô hình được đánh giá bằng các chỉ số như độ chính xác, độ chính xác, độ nhạy, và F1-score.
- Kết quả đánh giá được ghi lại và có thể xem trong [error.log](http://_vscodecontentref_/#%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22c%3A%5C%5CUsers%5C%5Cadmin%5C%5CDocuments%5C%5CYTB_Recommend%5C%5Cerror.log%22%2C%22_sep%22%3A1%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fadmin%2FDocuments%2FYTB_Recommend%2Ferror.log%22%2C%22scheme%22%3A%22file%22%7D%7D).

## Đóng góp

Nếu bạn muốn đóng góp cho dự án này, vui lòng fork repository và gửi pull request.

## Giấy phép

Dự án này được cấp phép theo giấy phép MIT. Xem tệp `LICENSE` để biết thêm chi tiết.
=========================================================================================
Đây là kế hoạch cập nhật:

1 Thu thập dữ liệu:
a. Sử dụng YouTube Data API để thu thập thông tin về video:

Tiêu đề video
ID video
Tags
Số lượt xem
Số lượt thích
Thời lượng video
b. Thu thập dữ liệu hành vi người dùng (có thể cần xây dựng ứng dụng riêng):
ID người dùng
ID video đã xem
Thời gian xem của mỗi video


2 Lưu trữ dữ liệu:
Lưu trữ dữ liệu thu thập được vào các file .txt:


3 Tiền xử lý dữ liệu:
a. Đọc dữ liệu từ các file .txt
b. Chuyển đổi dữ liệu văn bản (tiêu đề, tags) thành dạng số:
Tokenization
Tạo từ điển (vocabulary)
Chuyển đổi từ thành số nguyên
Loại bỏ dữ liệu nhiễu, Phân tích liên quan SVD
c. Chuẩn hóa các đặc trưng số (lượt xem, lượt thích, thời lượng)
d. Tách dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%)


4 Xây dựng mô hình:
Sử dụng CNN cho Content-based Filtering như đã mô tả trước đó:
a. Sử dụng mô hình CNN đã định nghĩa trong artifact "cnn-content-filtering"
b. Mở rộng mô hình để xử lý thêm các đặc trưng số:

Thêm một nhánh Dense cho các đặc trưng số (lượt xem, lượt thích, thời lượng)
Kết hợp đầu ra của nhánh này với đầu ra của CNN


5 Huấn luyện mô hình:
a. Chuẩn bị dữ liệu đầu vào cho mô hình từ tập huấn luyện
b. Định nghĩa hàm mất mát và metrics (ví dụ: binary crossentropy và accuracy)
c. Sử dụng mini-batch gradient descent để huấn luyện mô hình
d. Theo dõi hiệu suất trên tập validation và điều chỉnh hyperparameters

6 Đánh giá mô hình:
a. Sử dụng tập kiểm tra để đánh giá hiệu suất cuối cùng của mô hình
b. Tính toán các metrics như Accuracy, Precision, Recall, F1-score
c. Đảm bảo độ chính xác (ACC) >= 70% như yêu cầu thì gán nhãn

7 Xây dựng hệ thống gợi ý:
a. Tạo hàm để xử lý đầu vào mới (thông tin video và lịch sử xem của người dùng)
b. Sử dụng mô hình đã huấn luyện để dự đoán khả năng người dùng thích một video
c. Sắp xếp video theo điểm số dự đoán và chọn top N video để gợi ý
8 Tạo giao diện người dùng:
a. Xây dựng một giao diện web đơn giản để demo hệ thống

