# EMG/IMU Signal Analysis Configuration

# Serial Communication
SERIAL_PORT = 'COM5'      # Serial port for device connection
BAUDRATE = 921600         # Baud rate for serial communication

# Data Buffer Settings
BUFFER_SIZE = 1000        # EMG buffer size (samples)
IMU_BUFFER_SIZE = 500     # IMU buffer size (samples)

# Data Processing
EMG_SAMPLE_RATE = 250     # EMG sampling rate (Hz) - fixed by hardware
IMU_SAMPLE_RATE = 100     # IMU sampling rate (Hz) - estimated dynamically

# Data Saving
SAVE_DATA = True          # Enable/disable data saving
DATA_DIR = "data"         # Directory for saved data files
MAX_FILE_SIZE = 100 * 1024 * 1024  # Maximum file size (100MB)

# Visualization
PLOT_UPDATE_INTERVAL = 50  # Plot update interval (ms)
EMG_YLIM = (-2000, 2000)   # EMG plot Y-axis limits (μV)
IMU_YLIM = (-5, 5)         # IMU plot Y-axis limits

# Logging
LOG_LEVEL = "INFO"        # Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Feature Extraction
FEATURE_WINDOW_SIZE = 50  # Window size for feature calculation (samples)
FEATURE_UPDATE_INTERVAL = 5  # Feature update interval (seconds)