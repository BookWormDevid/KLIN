# pylint: disable=too-few-public-methods
"""DTO contracts used by the API auth endpoints."""

from __future__ import annotations

import msgspec


class JWTLoginDto(msgspec.Struct, frozen=True):
    """Input DTO for obtaining an API JWT token."""

    secret: str


class JWTTokenDto(msgspec.Struct, frozen=True):
    """Output DTO containing a signed JWT access token."""

    access_token: str
    token_type: str
    expires_in: int
