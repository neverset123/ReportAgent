"""Initial migration

Revision ID: 3a2de3439fea
Revises: 0724815ba38d
Create Date: 2024-12-24 13:50:21.762181

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3a2de3439fea'
down_revision: Union[str, None] = '0724815ba38d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('articles', schema=None) as batch_op:
        batch_op.add_column(sa.Column('rag', sa.Boolean(), nullable=True))

    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('articles', schema=None) as batch_op:
        batch_op.drop_column('rag')

    # ### end Alembic commands ###
