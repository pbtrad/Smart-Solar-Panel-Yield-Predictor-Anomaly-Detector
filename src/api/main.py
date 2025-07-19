from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
import structlog
from contextlib import asynccontextmanager

from .config import settings
from .middleware import add_security_middleware
from .endpoints import router
from .auth import create_access_token, authenticate_user
from .models import ErrorResponse

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Smart Solar API", version=settings.VERSION)
    yield
    logger.info("Shutting down Smart Solar API")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Smart Solar API for solar power forecasting and anomaly detection",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.include_router(router, prefix=settings.API_V1_STR)

app = add_security_middleware(app)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(
        "Unhandled exception",
        error=str(exc),
        url=str(request.url),
        method=request.method,
    )

    if settings.LOG_LEVEL == "DEBUG":
        detail = str(exc)
    else:
        detail = "Internal server error"

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(error="Internal server error", detail=detail).dict(),
    )


@app.get("/")
async def root():
    return {"message": "Smart Solar API", "version": settings.VERSION, "docs": "/docs"}


@app.post("/token")
async def login_for_access_token(username: str, password: str):
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user["username"]})

    logger.info("User login successful", username=username)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"username": user["username"], "role": user["role"]},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
