import asyncpg
import asyncio

DB_URL = "postgresql://postgres:AshlinJoelSizzin@localhost:5432/face_detection_db"

async def init_db():
    conn = await asyncpg.connect(DB_URL)
    
    # Create extension for vector operations
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        print("Vector extension created or already exists")
    except Exception as e:
        print(f"Warning: Could not create vector extension: {e}")
        print("Make sure pgvector is installed in your PostgreSQL instance")
    
    # Create tables
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    print("Users table created or already exists")
    
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS user_face_profiles (
            profile_id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(user_id),
            image_path TEXT NOT NULL,
            embeddings VECTOR[],
            augment_count INTEGER,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    print("User face profiles table created or already exists")
    
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS bad_actors (
            actor_id SERIAL PRIMARY KEY,
            ip_address INET NOT NULL,
            upload_count INTEGER DEFAULT 0,
            last_seen TIMESTAMP DEFAULT NOW(),
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    print("Bad actors table created or already exists")
    
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            upload_id SERIAL PRIMARY KEY,
            actor_id INTEGER REFERENCES bad_actors(actor_id),
            file_path TEXT NOT NULL,
            upload_time TIMESTAMP DEFAULT NOW()
        )
    """)
    print("Uploads table created or already exists")
    
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            detection_id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(user_id),
            actor_id INTEGER REFERENCES bad_actors(actor_id),
            upload_id INTEGER REFERENCES uploads(upload_id),
            similarity FLOAT,
            detected_at TIMESTAMP DEFAULT NOW(),
            notified BOOLEAN DEFAULT FALSE
        )
    """)
    print("Detections table created or already exists")
    
    await conn.close()
    print("Database initialized successfully!")

if __name__ == "__main__":
    asyncio.run(init_db())