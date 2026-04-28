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

# 2. Run the calibration file (only once needed to set the first reference and the first biased calibration). If the camera position changes, then make sure to recalibrate the bias
python calibrate.py

# 3. Run the classification file (it uses the stored files in calibration created by calibrate.py)
python classify.py
```