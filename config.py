# Configuration file for Metacognitive Attention

# Flask Configuration
FLASK_ENV = "production"  # 'production' or 'development'
FLASK_DEBUG = False
FLASK_PORT = 5000
FLASK_HOST = "0.0.0.0"

# Model Configuration
MODEL_DEVICE = "auto"  # 'cuda', 'cpu', or 'auto' to detect
MODEL_PRECISION = "float32"  # 'float32' or 'float16' for GPU optimization
CACHE_MODELS = True

# Upload Configuration
MAX_FILE_SIZE = 16777216  # 16MB in bytes
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp"}

# Image Processing
MAX_IMAGE_WIDTH = 2048
MAX_IMAGE_HEIGHT = 2048
DEFAULT_PROCESS_SIZE = 256  # Size to resize for processing

# Saliency Model Configuration
SALIENCY_KERNEL_SIZE = 15
SALIENCY_NUM_SCALES = 4
SALIENCY_SIGMA_CENTER = 1.0
SALIENCY_SIGMA_SURROUND = 3.0

# CORS Configuration
CORS_ORIGINS = ["http://localhost:3000", "http://localhost:5000"]

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "app.log"

# Performance
ENABLE_CACHING = True
CACHE_MAX_SIZE = 100  # Max number of processed images to cache
PROCESSING_TIMEOUT = 30  # seconds

# API Rate Limiting (optional)
RATE_LIMIT_ENABLED = False
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_PERIOD = 3600  # 1 hour

# Security
SECURE_HEADERS = True
REQUIRE_HTTPS = False  # Set to True in production with HTTPS

# Frontend Configuration
FRONTEND_URL = "http://localhost:3000"
FRONTEND_BUILD_DIR = "frontend/build"
