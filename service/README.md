# Esmart AI Image Generator Service

Service FastAPI và Gradio để tạo hình ảnh bằng Kandinsky 2.2 cho ứng dụng Esmart AI Content Creator.

## Cài đặt

1. Đảm bảo bạn đã cài đặt Python 3.8+ và pip.

2. Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

3. Tạo file `.env` từ `.env.example`:

```bash
cp .env.example .env
```

4. Cập nhật các biến môi trường trong file `.env`, bao gồm API key cho OpenRouter (để sử dụng DeepSeek API) và cấu hình cho dịch vụ lưu trữ đám mây (S3 hoặc tương tự).

## Cài đặt tối ưu cho Apple Silicon (Mac M1/M2)

Để tăng tốc độ xử lý trên chip Apple Silicon, chúng tôi đã tối ưu hóa service để sử dụng MPS (Metal Performance Shaders):

1. Sử dụng script cài đặt tự động:

```bash
./setup_m1.sh
```

Script này sẽ:

- Tạo môi trường ảo (venv) nếu chưa tồn tại
- Cài đặt PyTorch phiên bản tối ưu cho Apple Silicon
- Cài đặt các thư viện cần thiết với cấu hình đúng

2. Các tối ưu hóa đã được áp dụng:
   - Sử dụng MPS backend của PyTorch
   - Float16 quantization giảm bộ nhớ
   - Giảm số inference steps xuống 20
   - Sử dụng safetensors để tăng tốc độ tải model

### Benchmark dự kiến

| Thiết lập    | Thời gian/ảnh |
| ------------ | ------------- |
| M1 CPU       | ~3-5 phút     |
| M1 GPU (mps) | 45-90 giây    |
| Đã tối ưu    | 20-40 giây    |

## Chạy service

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 7860
```

Sau khi chạy, service sẽ khả dụng tại: http://localhost:7860

## Triển khai trên Hugging Face Spaces

Service này được thiết kế để dễ dàng triển khai trên Hugging Face Spaces. Đã sử dụng Gradio làm giao diện người dùng chính:

### Các bước triển khai

1. Tạo không gian mới trên Hugging Face
2. Upload các file sau lên repository của không gian:
   - `app.py`: Ứng dụng chính (sử dụng Gradio UI)
   - `requirements.txt`: Các phụ thuộc cần thiết
   - `README.md`: File README với cấu hình (phải có YAML frontmatter)

### Lưu trữ ảnh lâu dài

Service hỗ trợ lưu trữ ảnh tạo ra trên dịch vụ lưu trữ đám mây (S3) để giữ ảnh khi Spaces khởi động lại. Để kích hoạt tính năng này:

1. Cấu hình các biến môi trường trong thiết lập Spaces:
   - `USE_CLOUD_STORAGE=True`
   - `S3_BUCKET`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_REGION`

## Giao diện Gradio

Service cung cấp giao diện Gradio trực quan với các tính năng:

1. **Text to Image**: Tạo ảnh từ mô tả văn bản
2. **Topic to Image**: Tạo ảnh từ một chủ đề
3. **API Information**: Thông tin về các API endpoints

## API Endpoints

Service cũng cung cấp các API endpoints:

- **GET /api**: Kiểm tra trạng thái service
- **GET /api/health**: Endpoint kiểm tra sức khỏe cho HF Spaces monitoring
- **GET /api/status**: Kiểm tra trạng thái chi tiết của service và các models
- **POST /api/generate-image**: Tạo ảnh từ prompt
- **POST /api/topic-to-image**: Tạo ảnh từ chủ đề
- **GET /api/images/{filename}**: Lấy ảnh đã tạo theo filename

## Tối ưu hóa cho Hugging Face Spaces

Service đã được tối ưu hóa để chạy hiệu quả trên Hugging Face Spaces:

1. **Quản lý bộ nhớ thông minh**:

   - Lazy loading models khi cần
   - Sử dụng float16 để giảm footprint bộ nhớ
   - Giải phóng bộ nhớ sau khi xử lý
   - Attention slicing cho decoder model

2. **Hiệu suất cao**:

   - Cấu hình tối ưu cho GPU A10G
   - Giảm số bước inference mặc định
   - Giới hạn kích thước ảnh tạo ra

3. **Độ bền cao**:
   - Lưu trữ ảnh trên cloud storage
   - Logging toàn diện
   - Xử lý lỗi mạnh mẽ

## Lưu ý

- Service sử dụng Kandinsky 2.2 cho việc tạo ảnh từ văn bản
- Mô hình sẽ được tải khi cần thiết để tiết kiệm bộ nhớ
- Nếu OpenRouter API key được cung cấp, service sẽ sử dụng DeepSeek để cải thiện prompt trước khi tạo ảnh
