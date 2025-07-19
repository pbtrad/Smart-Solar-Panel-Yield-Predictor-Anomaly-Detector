from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import time
import structlog
from typing import Dict
from .config import settings

logger = structlog.get_logger()

# Rate limiting storage (use Redis in production)
rate_limit_storage: Dict[str, list] = {}


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host

        if not self._check_rate_limit(client_ip):
            return Response(
                content="Rate limit exceeded", status_code=429, media_type="text/plain"
            )

        return await call_next(request)

    def _check_rate_limit(self, client_ip: str) -> bool:
        current_time = time.time()
        minute_ago = current_time - 60

        if client_ip in rate_limit_storage:
            rate_limit_storage[client_ip] = [
                timestamp
                for timestamp in rate_limit_storage[client_ip]
                if timestamp > minute_ago
            ]
        else:
            rate_limit_storage[client_ip] = []

        if len(rate_limit_storage[client_ip]) >= settings.RATE_LIMIT_PER_MINUTE:
            return False

        rate_limit_storage[client_ip].append(current_time)
        return True


class SecurityHeadersMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response


class LoggingMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        logger.info(
            "API request",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host,
            user_agent=request.headers.get("user-agent", ""),
        )

        response = await call_next(request)

        duration = time.time() - start_time
        logger.info(
            "API response",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            duration=duration,
            client_ip=request.client.host,
        )

        return response


def add_security_middleware(app):

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["*"]  # Configure properly in production
    )

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(LoggingMiddleware)

    return app
