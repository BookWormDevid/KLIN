"""empty message

Revision ID: ab238fe8af7d
Revises: 9363382a1e84, e1883ee841c6
Create Date: 2026-03-26 21:35:34.248255

"""

from collections.abc import Sequence


# revision identifiers, used by Alembic.
revision: str = "ab238fe8af7d"
down_revision: str | Sequence[str] | None = ("9363382a1e84", "e1883ee841c6")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
