from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    input_data = Column(JSON)
    predicted_price = Column(Float)
    model_version = Column(String)
    confidence_interval = Column(JSON)  # Stocké comme [lower, upper]

class Feedback(Base):
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    actual_price = Column(Float)
    feedback_type = Column(String)
    comments = Column(String)

# Création de la base de données
engine = create_engine("sqlite:///./real_estate.db")
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)