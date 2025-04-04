# Hướng dẫn triển khai Esmart Solutions

Dự án này bao gồm 3 phần chính cần triển khai riêng biệt:

1. Frontend (Vue.js)
2. Backend API (Node.js + Express)
3. AI Service (Python + FastAPI)

## 1. Triển khai Frontend trên Netlify

Frontend đã được triển khai lên Netlify thông qua kho lưu trữ GitHub. Tệp cấu hình `netlify.toml` đã được thiết lập để xây dựng và triển khai ứng dụng Vue.js.

## 2. Triển khai Backend trên Render.com

### Backend API (Node.js)

1. Đăng nhập vào [Render.com](https://render.com)
2. Chọn "New" > "Web Service"
3. Liên kết với kho lưu trữ GitHub của bạn
4. Thiết lập như sau:
   - Name: esmart-api
   - Root Directory: server
   - Environment: Node.js
   - Build Command: npm install
   - Start Command: npm start
5. Ở phần "Environment Variables", thêm các biến môi trường từ tệp `.env` của bạn
6. Tạo một cơ sở dữ liệu PostgreSQL:
   - Chọn "New" > "PostgreSQL"
   - Đặt tên: esmart-postgres
   - Sau khi tạo, lấy connection string và thêm vào biến môi trường DATABASE_URL của service Node.js

### AI Service (Python + FastAPI)

1. Trong Render.com, chọn "New" > "Web Service"
2. Liên kết với kho lưu trữ GitHub của bạn
3. Thiết lập như sau:
   - Name: esmart-ai-service
   - Root Directory: service
   - Environment: Python 3
   - Build Command: pip install -r requirements.txt
   - Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
4. Ở phần "Environment Variables", thêm OPENROUTER_API_KEY và các biến môi trường khác từ tệp `.env`
5. **Lưu ý**: FastAPI với Kandinsky yêu cầu tài nguyên cao, nên bạn cần chọn gói có RAM và CPU phù hợp

## 3. Thiết lập API Gateway (Tùy chọn)

Để định tuyến giữa các dịch vụ backend, bạn có thể triển khai API Gateway:

1. Trong Render.com, chọn "New" > "Web Service"
2. Liên kết với kho lưu trữ GitHub của bạn
3. Thiết lập:
   - Name: esmart-api-gateway
   - Root Directory: api-gateway
   - Environment: Node.js
   - Build Command: npm install
   - Start Command: npm start
4. Ở phần "Environment Variables", thêm:
   - EXPRESS_API_URL: URL của Node.js API (từ bước 2)
   - PYTHON_API_URL: URL của FastAPI service (từ bước 3)

## 4. Cập nhật Frontend để kết nối với Backend

Sau khi triển khai tất cả các dịch vụ backend, bạn cần cập nhật cấu hình frontend để trỏ đến các URL của backend:

1. Trong tệp nguồn của frontend, cập nhật các URL API để trỏ đến services trên Render
2. Triển khai lại frontend trên Netlify

## 5. Kiểm tra tích hợp

Sau khi triển khai xong tất cả các thành phần, kiểm tra:

1. Frontend có thể gọi đến Node.js API
2. Node.js API có thể gọi đến FastAPI service
3. Các tính năng như xử lý hình ảnh AI, postgresql và mongodb hoạt động đúng

## Giám sát và Xử lý sự cố

- Kiểm tra logs trên Render.com để xác định sự cố triển khai
- Kiểm tra các endpoint health check để đảm bảo các dịch vụ hoạt động
  - Node.js API: /api/health
  - FastAPI: /

## Kết luận

Nếu bạn cần hỗ trợ thêm, hãy tham khảo tài liệu của:

- [Netlify](https://docs.netlify.com)
- [Render](https://render.com/docs)
