"""JWT auth endpoints for the API."""

from __future__ import annotations

import secrets
from collections.abc import Sequence

from litestar import Controller, MediaType, post
from litestar.exceptions import HTTPException
from litestar.status_codes import HTTP_201_CREATED, HTTP_401_UNAUTHORIZED

from app.application.dto import JWTLoginDto, JWTTokenDto
from app.config import app_settings
from app.presentation.litestar.auth import build_jwt_auth


class AuthController(Controller):
    """Endpoints used to issue API JWT tokens."""

    path = "/auth"
    tags: Sequence[str] | None = ["Auth"]

    @post(
        "/token",
        status_code=HTTP_201_CREATED,
        media_type=MediaType.JSON,
        opt={"exclude_from_auth": True},
    )
    async def issue_token(self, data: JWTLoginDto) -> JWTTokenDto:
        """Issue a signed Bearer token for API access."""

        if not secrets.compare_digest(data.secret, app_settings.jwt_secret):
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            )

        jwt_auth = build_jwt_auth()
        return JWTTokenDto(
            access_token=jwt_auth.create_token(identifier="klin-api"),
            token_type="bearer",
            expires_in=int(app_settings.jwt_token_ttl.total_seconds()),
        )
