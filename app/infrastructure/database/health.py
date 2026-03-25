"""
Database health helpers shared by worker startup hooks.
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


async def ping_database(engine: AsyncEngine) -> None:
    """Open a connection and execute a trivial query to verify connectivity."""
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))
