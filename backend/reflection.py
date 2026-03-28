from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from dotenv import load_dotenv
import uuid
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sessions.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
Base = declarative_base()


class ReflectionLog(Base):
    __tablename__ = "reflections"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True, nullable=False)
    session_id = Column(String, nullable=True)
    student_message = Column(Text, nullable=False)
    agent_response = Column(Text, nullable=False)
    reflection = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def save_reflection(
    user_id: str,
    session_id: str,
    message: str,
    response: str,
    reflection: str
) -> None:
    """Persist a reflection log entry to the database."""
    db = SessionLocal()
    try:
        log = ReflectionLog(
            id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            student_message=message,
            agent_response=response,
            reflection=reflection,
        )
        db.add(log)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[Reflection] Error saving reflection: {e}")
    finally:
        db.close()


def get_reflections(user_id: str, limit: int = 10) -> list:
    """Fetch recent reflections to display in the UI."""
    db = SessionLocal()
    try:
        logs = (
            db.query(ReflectionLog)
            .filter(ReflectionLog.user_id == user_id)
            .order_by(ReflectionLog.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": log.id,
                "reflection": log.reflection,
                "timestamp": log.timestamp.isoformat(),
                "message_preview": (log.student_message[:80] + "...")
                if len(log.student_message) > 80
                else log.student_message,
                "response_preview": (log.agent_response[:120] + "...")
                if len(log.agent_response) > 120
                else log.agent_response,
            }
            for log in logs
        ]
    except Exception as e:
        print(f"[Reflection] Error fetching reflections: {e}")
        return []
    finally:
        db.close()
