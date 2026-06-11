from database import engine
from models_db import Base
import models_db  # important: ensures all models are registered

def create_tables():
    print("Creating tables in Verdixa AI database...")
    
    Base.metadata.create_all(bind=engine)
    
    print("✅ Tables created successfully!")

if __name__ == "__main__":
    create_tables()