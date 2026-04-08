"""empty message

Revision ID: c836025fc887
Revises: 8aee103168a8, 9e0aaedc9dc1
Create Date: 2026-03-26 21:35:34.248255

"""

from collections.abc import Sequence


# revision identifiers, used by Alembic.
revision: str = "c836025fc887"
down_revision: str | Sequence[str] | None = ("8aee103168a8", "9e0aaedc9dc1")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:

    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
