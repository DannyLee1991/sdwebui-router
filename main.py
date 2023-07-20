import uvicorn
from app import app

if __name__ == "__main__":
    # 启动服务
    uvicorn.run(
        app="run:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )
