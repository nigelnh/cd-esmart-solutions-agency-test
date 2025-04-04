module.exports = {
  routes: [
    {
      route: "/api",
      target: process.env.EXPRESS_API_URL || "http://localhost:3001",
    },
    {
      route: "/ai",
      target: process.env.PYTHON_API_URL || "http://localhost:8000",
    },
  ],
  options: {
    changeOrigin: true,
    pathRewrite: {
      "^/ai": "/",
    },
    logger: console,
  },
};
