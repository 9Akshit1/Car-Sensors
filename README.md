## Installation & Setup

**Requirement:** Python 3.12.x (Verified on 3.12.4)  
*Note: Python 3.13 is currently not recommended due to MediaPipe compatibility issues.*

```powershell
# 1. Navigate to the project folder
cd piPupil

# 2. Create a virtual environment forcing Python 3.12
py -3.12 -m venv venv

# 3. Activate the environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# 4. Update pip and install verified dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Project
Ensure your USB camera is plugged in before starting the script.

```powershell
# 1. Navigate to the project folder
cd piPupil

# 2. Run the file
python classify_live_transformationM.py
```

### Configuration
You can modify the following variables inside `classify_live_transformationM.py` to suit your setup:

* **`CAMERA_ID`**: The script defaults to `1`. Change this to `0` or `2` if your webcam doesn't open immediately.
* **`IS_DISPLAY`**: Set to `False` for headless operation. This is highly recommended for performance when running on the **Raspberry Pi 5**.
* **`MODEL_FILES`**: Paths to the `.pkl` models used for classification (e.g., `Models/Cubic_SVM.pkl`).