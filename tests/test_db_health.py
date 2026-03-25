from unittest.mock import AsyncMock, MagicMock

import pytest

from app.infrastructure.database.health import ping_database


@pytest.mark.anyio
async def test_ping_database_executes_select_1() -> None:
    connection = AsyncMock()
    connect_context = AsyncMock()
    connect_context.__aenter__.return_value = connection
    connect_context.__aexit__.return_value = None

    engine = MagicMock()
    engine.connect.return_value = connect_context

    await ping_database(engine)

    connection.execute.assert_awaited_once()
    statement = connection.execute.await_args.args[0]
    assert str(statement) == "SELECT 1"
