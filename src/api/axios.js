import axios from "axios";
import { API_BASE_URL } from "./config";

// Tạo instance axios với URL gốc
const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // Tăng từ 30 lên 60 giây để xử lý các request lâu
  headers: {
    "Content-Type": "application/json",
  },
});

// Thêm interceptor để xử lý lỗi
axiosInstance.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Ghi log lỗi với nhiều thông tin hơn
    console.error("API error intercepted:", error);

    // Xử lý lỗi 500 từ server
    if (error.response && error.response.status === 500) {
      console.error("Server error details:", error.response.data);
      // Có thể thông báo cho người dùng hoặc thử lại request
    }

    // Xử lý lỗi timeout
    if (error.code === "ECONNABORTED") {
      console.error(
        "Request timeout. The server is taking too long to respond."
      );
    }

    // Xử lý lỗi network
    if (!error.response) {
      console.error("Network error. Please check your connection.");
    }

    return Promise.reject(error);
  }
);

export default axiosInstance;
