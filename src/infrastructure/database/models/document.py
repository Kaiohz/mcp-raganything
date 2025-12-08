from sqlalchemy import Column, Integer, String, BigInteger
from infrastructure.database.models.base import Base


class DocumentModel(Base):
    """
    SQLAlchemy model for the documents table.
    Stores metadata about indexed documents.
    """
    
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    file_path = Column(String, unique=True, nullable=False, index=True)
    filename = Column(String, nullable=False)
    content_hash = Column(String, nullable=True)
    indexed_at = Column(BigInteger, nullable=True)
    status = Column(String, default="pending", nullable=False)
