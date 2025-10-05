#!/bin/bash
# Start Flask + SocketIO server using Gunicorn + Eventlet
source .venv/bin/activate
gunicorn -k eventlet -w 1 --bind 0.0.0.0:$PORT app:app
