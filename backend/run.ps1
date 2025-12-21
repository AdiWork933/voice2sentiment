# Allow script execution for this session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# Activate virtual environment
. .venv\Scripts\Activate.ps1

# Run FastAPI server
# uvicorn main_api:app --host 0.0.0.0 --port $PORT
# Example command to run your app
uvicorn main_api:app --reload --port 8000
