import asyncpg
import asyncio

DB_URL = "postgresql://postgres:AshlinJoelSizzin@localhost:5432/face_detection_db"

async def check_tables():
    conn = await asyncpg.connect(DB_URL)
    
    # Check if tables exist by querying their schemas
    tables = ['users', 'user_face_profiles', 'bad_actors', 'uploads', 'detections']
    
    for table in tables:
        try:
            result = await conn.fetch(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}' ORDER BY ordinal_position")
            print(f"\nTable: {table}")
            for row in result:
                print(f"  {row['column_name']}: {row['data_type']}")
        except Exception as e:
            print(f"Error checking table {table}: {e}")
    
    await conn.close()

if __name__ == "__main__":
    asyncio.run(check_tables())