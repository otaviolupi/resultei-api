from sqlalchemy import Column, String, Float, DateTime, Text
from sqlalchemy.sql import func
from app.db.database import Base


class SimulationJob(Base):
    __tablename__ = "simulation_jobs"

    id = Column(String, primary_key=True)
    procedure = Column(String, nullable=False)
    intensity = Column(Float, nullable=False)
    mode = Column(String, nullable=False)
    status = Column(String, default="completed")
    original_url = Column(Text)
    result_url = Column(Text)
    comparison_url = Column(Text)
    processing_time_ms = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())