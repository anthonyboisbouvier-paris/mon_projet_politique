from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Video(Base):
    __tablename__ = 'videos'
    id = Column(Integer, primary_key=True)
    url = Column(String, unique=True, nullable=False)
    politician_id = Column(Integer, ForeignKey('politicians.id'))
    title = Column(String)
    description = Column(Text)
    published_at = Column(DateTime)

class Transcription(Base):
    __tablename__ = 'transcriptions'
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=False)
    texte = Column(Text)
    mots_par_minute = Column(Float)