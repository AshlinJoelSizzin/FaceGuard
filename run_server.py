import hypercorn
from face_monitor import app

if __name__ == "__main__":
    hypercorn.run(app, host="0.0.0.0", port=8000)