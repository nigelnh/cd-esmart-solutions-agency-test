<template>
  <div class="step-content">
    <h2>{{ $t("creatorAI.requirements.title") }}</h2>
    <div class="form-container">
      <div class="form-group">
        <label>{{ $t("creatorAI.requirements.projectTitle") }}</label>
        <input
          v-model="formData.title"
          type="text"
          placeholder="Enter project title"
        />
      </div>
      <div class="form-group">
        <label>{{ $t("creatorAI.requirements.projectDescription") }}</label>
        <textarea
          v-model="formData.description"
          placeholder="Enter project description"
          rows="3"
          style="resize: none"
        ></textarea>
      </div>
      <div class="form-group">
        <label>{{ $t("creatorAI.requirements.topic") }}</label>
        <input
          v-model="formData.topic"
          type="text"
          placeholder="Enter your topic"
        />
      </div>
      <div class="form-group">
        <label>{{ $t("creatorAI.requirements.keywords") }}</label>
        <input
          v-model="formData.keywords"
          type="text"
          :placeholder="$t('creatorAI.requirements.keywordsTip')"
        />
      </div>
      <div class="form-group">
        <label>{{ $t("creatorAI.requirements.tone") }}</label>
        <select v-model="formData.tone">
          <option value="professional">
            {{ $t("creatorAI.requirements.professional") }}
          </option>
          <option value="casual">
            {{ $t("creatorAI.requirements.casual") }}
          </option>
          <option value="friendly">
            {{ $t("creatorAI.requirements.friendly") }}
          </option>
          <option value="formal">
            {{ $t("creatorAI.requirements.formal") }}
          </option>
          <option value="persuasive">
            {{ $t("creatorAI.requirements.persuasive") }}
          </option>
          <option value="optimistic">
            {{ $t("creatorAI.requirements.optimistic") }}
          </option>
        </select>
      </div>
      <div class="form-group">
        <label>{{ $t("creatorAI.requirements.contentType") }}</label>
        <select v-model="formData.type">
          <option value="blog">{{ $t("creatorAI.requirements.blog") }}</option>
          <option value="social">
            {{ $t("creatorAI.requirements.socialMedia") }}
          </option>
          <option value="article">
            {{ $t("creatorAI.requirements.article") }}
          </option>
          <option value="product">
            {{ $t("creatorAI.requirements.productDescription") }}
          </option>
          <option value="email">
            {{ $t("creatorAI.requirements.emailNewsletter") }}
          </option>
        </select>
      </div>
    </div>
    <div class="button-group">
      <button class="secondary-button" @click="$emit('prev')">
        {{ $t("creatorAI.requirements.previous") }}
      </button>
      <button
        class="primary-button"
        @click="generateContent"
        :disabled="isLoading"
      >
        <span v-if="isLoading" class="loading-spinner"></span>
        {{
          isLoading
            ? $t("creatorAI.requirements.analyzing")
            : $t("creatorAI.requirements.next")
        }}
      </button>
    </div>
    <div v-if="isLoading" class="loading-message">
      <p>{{ $t("creatorAI.requirements.analyzing") }}</p>
    </div>
  </div>
</template>

<script>
import axios from "axios";

export default {
  name: "RequirementsStep",
  props: {
    initialData: {
      type: Object,
      required: true,
    },
  },
  data() {
    return {
      formData: {
        ...this.initialData,
        title: "",
        description: "",
      },
      isLoading: false,
      isGeneratingContent: false,
      isAnalyzingSEO: false,
      apiUrl: "http://localhost:3001/api/generate-content",
    };
  },
  computed: {
    apiEndpoint() {
      // Use environment variable if available, otherwise default to localhost
      return (
        process.env.VUE_APP_API_URL ||
        "https://esmart-api-lnni.onrender.com/api"
      );
    },
  },
  methods: {
    async generateContent() {
      try {
        // Validate input
        if (
          !this.formData.topic ||
          !this.formData.keywords ||
          !this.formData.title
        ) {
          alert(this.$t("creatorAI.requirements.requiredFields"));
          return;
        }

        this.isLoading = true;
        this.isGeneratingContent = true;

        // First create a project
        let projectId;
        let projectCreated = false;

        try {
          const projectResponse = await axios.post(
            `${this.apiEndpoint}/projects`,
            {
              title: this.formData.title,
              description:
                this.formData.description ||
                `Project about ${this.formData.topic}`,
              type: this.formData.type,
            }
          );

          if (projectResponse.data.success) {
            projectId = projectResponse.data.data.id;
            projectCreated = true;
            console.log("Created project with ID:", projectId);
          } else {
            throw new Error(
              "Failed to create project: " +
                (projectResponse.data.error || "Unknown error")
            );
          }
        } catch (projectError) {
          console.error("Error creating project:", projectError);

          // If we cannot create a project, use local content
          const placeholderContent = this.generatePlaceholderContent();

          const updatedData = {
            ...this.formData,
            content: placeholderContent,
          };

          this.$emit("update:data", updatedData);
          alert(
            "Could not create project: " +
              projectError.message +
              ". Using sample content."
          );
          this.$emit("next");
          this.isLoading = false;
          return;
        }

        // Now create content associated with this project
        if (projectCreated && projectId) {
          try {
            // Thêm logic thử lại và xử lý lỗi tốt hơn
            let retryCount = 0;
            const maxRetries = 2;
            let contentCreated = false;
            let contentError = null;

            while (retryCount <= maxRetries && !contentCreated) {
              try {
                console.log(
                  `Attempting to create content (attempt ${retryCount + 1}/${
                    maxRetries + 1
                  })...`
                );

                // Chuẩn hóa dữ liệu đầu vào để tránh lỗi từ server
                const sanitizedKeywords = this.sanitizeInput(
                  this.formData.keywords
                );
                const sanitizedTopic = this.sanitizeInput(this.formData.topic);

                console.log(
                  "Original keywords length:",
                  this.formData.keywords.length
                );
                console.log(
                  "Sanitized keywords length:",
                  sanitizedKeywords.length
                );

                const contentResponse = await axios.post(
                  `${this.apiEndpoint}/content/contents`,
                  {
                    projectId: projectId,
                    topic: sanitizedTopic,
                    keywords: sanitizedKeywords,
                    tone: this.formData.tone,
                    contentType: this.formData.type,
                  },
                  {
                    // Tăng timeout cho API call
                    timeout: 600000, // Tăng lên 10 phút (10 * 60 * 1000 = 600000ms)
                  }
                );

                if (contentResponse.data.success) {
                  contentCreated = true;

                  // Update the content data with the generated content
                  const updatedData = {
                    ...this.formData,
                    projectId: projectId,
                    contentId: contentResponse.data.data.id,
                    content:
                      contentResponse.data.data.generatedContent ||
                      "Không thể tạo nội dung",
                  };

                  // Emit update event with new data
                  this.$emit("update:data", updatedData);

                  // Go to next step
                  this.$emit("next");

                  // Hiển thị thông báo đang phân tích SEO
                  this.$emit("seo-analyzing", true);

                  break; // Thoát khỏi vòng lặp nếu thành công
                } else {
                  contentError = new Error(
                    contentResponse.data.error || "Lỗi không xác định từ server"
                  );
                  retryCount++;

                  if (retryCount <= maxRetries) {
                    console.log(
                      `Content creation failed, retrying (${retryCount}/${maxRetries})...`
                    );
                    await new Promise((resolve) => setTimeout(resolve, 2000)); // Đợi 2 giây trước khi thử lại
                  }
                }
              } catch (error) {
                contentError = error;
                console.error(
                  `Content creation attempt ${retryCount + 1} failed:`,
                  error
                );
                retryCount++;

                if (retryCount <= maxRetries) {
                  console.log(
                    `Retrying content creation (${retryCount}/${maxRetries})...`
                  );
                  await new Promise((resolve) => setTimeout(resolve, 2000)); // Đợi 2 giây trước khi thử lại
                }
              }
            }

            // Nếu sau tất cả các lần thử lại vẫn không tạo được nội dung
            if (!contentCreated) {
              // Kiểm tra nếu là lỗi timeout
              if (contentError && contentError.code === "ECONNABORTED") {
                const timeoutError = new Error(
                  "Server cần quá nhiều thời gian để tạo nội dung. Vui lòng thử lại với từ khóa ngắn hơn hoặc thử lại sau."
                );
                timeoutError.isTimeout = true;
                throw timeoutError;
              } else {
                throw (
                  contentError ||
                  new Error("Không thể tạo nội dung sau nhiều lần thử")
                );
              }
            }
          } catch (contentError) {
            console.error("Error generating content:", contentError);

            // If content generation failed but project was created, use placeholder
            const placeholderContent = this.generatePlaceholderContent();

            const updatedData = {
              ...this.formData,
              projectId: projectId,
              contentId: null,
              content: placeholderContent,
            };

            this.$emit("update:data", updatedData);

            // Hiển thị thông báo lỗi phù hợp
            let errorMessage = "Không thể tạo nội dung: ";
            if (contentError.isTimeout) {
              errorMessage += contentError.message;
            } else if (
              contentError.response &&
              contentError.response.status === 500
            ) {
              // Hiển thị chi tiết lỗi từ server nếu có
              const serverErrorDetails =
                contentError.response.data?.error ||
                contentError.response.data?.message ||
                "Lỗi máy chủ";
              console.error(
                "Server error details:",
                contentError.response.data
              );
              errorMessage += `Lỗi máy chủ (500): ${serverErrorDetails}. 
                              Vui lòng thử lại với từ khóa ngắn hơn hoặc báo cáo cho đội kỹ thuật.`;
            } else {
              errorMessage +=
                contentError.message + ". Đang sử dụng nội dung mẫu.";
            }

            alert(errorMessage);
            this.$emit("next");
          }
        }
      } catch (error) {
        console.error("Error in content generation process:", error);

        // If it's a network error, generate placeholder content for development/demos
        if (
          error.message.includes("Network Error") ||
          error.message.includes("Connection refused")
        ) {
          // Generate simple placeholder content for development
          const placeholderContent = this.generatePlaceholderContent();

          // Update the data with placeholder content
          const updatedData = {
            ...this.formData,
            content: placeholderContent,
          };

          // Emit update event with placeholder data
          this.$emit("update:data", updatedData);

          // Show a notification but still proceed
          alert("Không thể kết nối với server, đang sử dụng nội dung mẫu");

          // Go to next step
          this.$emit("next");
        } else {
          // For other errors, show the error message
          alert(
            "Không thể tạo nội dung: " + (error.message || "Lỗi không xác định")
          );
        }
      } finally {
        this.isLoading = false;
      }
    },

    generatePlaceholderContent() {
      const { topic, keywords, type } = this.formData;

      if (type === "blog") {
        return `# ${topic}: A Comprehensive Guide

## Introduction
Welcome to our comprehensive guide on ${topic}. In this article, we'll explore everything you need to know about this fascinating subject, with special attention to ${keywords}.

## Understanding ${topic}
${topic} has become increasingly important in today's world. When we consider ${keywords}, the significance becomes even more apparent. Experts in the field suggest that a thorough understanding of these concepts can lead to better outcomes.

## Key Benefits
- Improved understanding of ${keywords}
- Enhanced performance in related areas
- Greater appreciation for the complexities of ${topic}

## Practical Applications
The practical applications of ${topic} are numerous. Many professionals use these concepts daily, especially when dealing with ${keywords}.

## Conclusion
As we've explored throughout this article, ${topic} represents a significant area of interest, particularly when considered alongside ${keywords}.`;
      } else if (type === "social") {
        return `📢 Excited to share my thoughts on ${topic}!

Did you know that ${keywords} plays a crucial role in understanding this concept? I've been researching this topic extensively, and the results are fascinating.

What are your experiences with ${topic}? Comment below! 👇

#${topic.replace(/\s+/g, "")} #${keywords
          .split(",")[0]
          .trim()
          .replace(/\s+/g, "")} #ProfessionalInsights`;
      } else {
        return `# ${topic}: Unlock Your Potential

Are you looking to master ${topic}? Our comprehensive resources focus on ${keywords} to give you the edge you need.

## Why Choose Us?

- Expert guidance on ${topic}
- Specialized focus on ${keywords}
- Proven results for our clients

Don't wait to enhance your understanding of ${topic}. Start your journey today!

[Get Started] [Learn More] [Contact Us]`;
      }
    },

    sanitizeInput(input) {
      if (!input) return "";

      // Giới hạn độ dài keywords tối đa là 200 ký tự để tránh quá tải server
      let sanitized = input;
      if (sanitized.length > 200) {
        // Cắt tại dấu phẩy gần nhất trước 200 ký tự
        const commaIndex = sanitized.lastIndexOf(",", 200);
        if (commaIndex > 0) {
          sanitized = sanitized.substring(0, commaIndex);
        } else {
          sanitized = sanitized.substring(0, 200);
        }
      }

      // Loại bỏ các ký tự đặc biệt có thể gây lỗi
      sanitized = sanitized.replace(
        /[^\p{L}\p{N}\p{P}\p{Z}\p{Cf}\p{Cc}\p{M}]/gu,
        ""
      );

      return sanitized;
    },
  },
};
</script>

<style scoped>
.step-content {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

h2 {
  color: #1c1c4c;
  margin-bottom: 2rem;
  text-align: center;
}

.form-container {
  display: grid;
  gap: 1.5rem;
  max-width: 600px;
  margin: 0 auto;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.form-group label {
  font-weight: 500;
  color: #1c1c4c;
}

.form-group input,
.form-group select {
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 1rem;
}

.form-group textarea {
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 1rem;
  font-family: inherit;
  resize: vertical;
}

.button-group {
  display: flex;
  gap: 1rem;
  justify-content: center;
  margin-top: 2rem;
}

.primary-button {
  background: #1c1c4c;
  color: white;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: 6px;
  cursor: pointer;
  font-size: 1rem;
  transition: background-color 0.3s ease;
  position: relative;
}

.primary-button:hover:not(:disabled) {
  background: #2a2a6c;
}

.primary-button:disabled {
  background: #7a7a9c;
  cursor: not-allowed;
}

.secondary-button {
  background: white;
  color: #1c1c4c;
  border: 1px solid #1c1c4c;
  padding: 0.75rem 1.5rem;
  border-radius: 6px;
  cursor: pointer;
  font-size: 1rem;
  transition: all 0.3s ease;
}

.secondary-button:hover {
  background: #f8f9fa;
}

.loading-spinner {
  display: inline-block;
  width: 16px;
  height: 16px;
  border: 2px solid #ffffff;
  border-radius: 50%;
  border-top-color: transparent;
  animation: spin 1s linear infinite;
  margin-right: 8px;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.loading-message {
  background-color: #f0f5ff;
  border: 1px solid #d9e7ff;
  padding: 10px 15px;
  border-radius: 6px;
  margin-top: 15px;
  text-align: center;
  color: #1c1c4c;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% {
    background-color: #f0f5ff;
  }
  50% {
    background-color: #e5edff;
  }
  100% {
    background-color: #f0f5ff;
  }
}

@media (max-width: 768px) {
  .button-group {
    flex-direction: column;
  }

  .primary-button,
  .secondary-button {
    width: 100%;
  }
}
</style>
