"""JWT auth configuration for the Litestar API."""

from __future__ import annotations

from typing import cast

from litestar.connection import ASGIConnection
from litestar.security.jwt import JWTAuth
from litestar.security.jwt.token import Token

from app.config import app_settings


AUTH_EXCLUDED_PATHS = [
    r"^/api/docs(?:/.*)?$",
    r"^/schema(?:/.*)?$",
    r"^/metrics(?:/.*)?$",
    r"^/frontend(?:/.*)?$",
]


async def retrieve_jwt_subject(
    token: Token,
    _connection: ASGIConnection,
) -> str | None:
    """Return the token subject as the authenticated principal."""

    return token.sub


def build_jwt_auth() -> JWTAuth[str, Token]:
    """Build the shared JWT auth config used by the API application."""

    return cast(
        JWTAuth[str, Token],
        JWTAuth(
            token_secret=app_settings.jwt_secret,
            retrieve_user_handler=retrieve_jwt_subject,
            default_token_expiration=app_settings.jwt_token_ttl,
            exclude=AUTH_EXCLUDED_PATHS,
            openapi_security_scheme_name="BearerAuth",
            description="JWT Bearer authentication for the Klin API.",
        ),
    )
