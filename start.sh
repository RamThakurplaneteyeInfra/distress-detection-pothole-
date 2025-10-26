#!/bin/bash
# Start Flask + SocketIO server using Gunicorn + Eventlet
source .venv/bin/activate
<<<<<<< HEAD
gunicorn -k eventlet -w 1 --bind 0.0.0.0:$PORT app:app
=======
gunicorn -k eventlet -w 1 --timeout 300 --bind 0.0.0.0:${PORT:-5000} "app:create_app()"
>>>>>>> dcb6b30d9a6b43fc416fed284ed7bf3d7b7b2c40
