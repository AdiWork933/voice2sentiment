# # Allow script execution for this session
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# # Activate virtual environment
# . .venv\Scripts\Activate.ps1

# # Run FastAPI server
# # uvicorn main_api:app --host 0.0.0.0 --port $PORT
# # Example command to run your app
# uvicorn main_api:app --reload --port 8000


# Allow script execution for this session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# Activate virtual environment
. .venv\Scripts\Activate.ps1

# -----------------------------------------------------------
# NEW STEP: Open separate terminal for UI
# -----------------------------------------------------------
# This opens a new window, goes to your folder, activates venv, and runs app.py
Start-Process powershell -ArgumentList "-NoExit", "-Command & { cd '$PWD'; . .venv\Scripts\Activate.ps1; cd frontend; python app.py }"

# -----------------------------------------------------------
# Run FastAPI server (stays in this window)
# -----------------------------------------------------------
uvicorn backend.main_api:app --reload --port 8000