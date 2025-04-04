import axios from "axios";
import { API_BASE_URL } from "./config";

// Tạo instance axios với URL gốc
const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 giây
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
    console.error("API error intercepted:", error);
    return Promise.reject(error);
  }
);

export default axiosInstance;
