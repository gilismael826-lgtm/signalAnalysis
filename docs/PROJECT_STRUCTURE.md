# EMG/IMU Signal Analysis Project Structure

```
signalAnalysis/
│
├── src/                          # Source code directory
│   ├── system_control.py          # System control center (GUI)
│   ├── requirements.txt            # Python dependencies
│   └── __init__.py                 # Package initialization file
│
├── config/                       # Configuration files
│   └── config.py                   # Main configuration parameters
│
├── data/                         # Data storage directory
│   ├── emg_*.csv                  # EMG data files (CSV format)
│   └── imu_*.csv                  # IMU data files (CSV format)
│
├── output/                       # Output files directory
│   ├── plots/                      # Generated plots and figures
│   └── reports/                    # Analysis reports
│
├── docs/                         # Documentation
│   ├── README.md                   # Main documentation
│   ├── 快速开始指南.md              # Quick start guide
│   ├── PROJECT_STRUCTURE.md         # Project structure documentation
│   └── 项目整理完成.md              # Project completion notes
│
├── tests/                        # Test files
│   └── test_receiver.py            # Unit tests
│
├── logs/                         # Log files directory
│   └── system_control.log         # System control center logs
│
├── main.py                       # Main entry point script
│
└── 唯理-EMG腕带-纯硬件版说明书.pdf     # Hardware manual
```

## Directory Usage

### `src/` - Source Code
Contains the main system control center with GUI interface for data acquisition and analysis.

### `config/` - Configuration
Centralized configuration files for easy parameter management.

### `data/` - Data Storage
Stores collected EMG/IMU data files in CSV format.

### `output/` - Results Output
Generated plots, analysis results, and reports are saved here.

### `docs/` - Documentation
Project documentation, user guides, and technical specifications.

### `tests/` - Testing
Unit tests and integration tests for system.

### `logs/` - Log Files
Runtime logs and system operation records.

## Usage Examples

### Start System Control Center
```bash
# Run system control center (default)
python main.py

# Run system control center (explicit)
python main.py --control

# List available serial ports
python main.py --list-ports
```

### System Control Center Features
- **Data Acquisition**: Serial port selection, start/stop control, data saving
- **Real-time Visualization**: EMG waveforms and IMU bar charts
- **Data Analysis**: CSV file analysis and statistics
- **System Tools**: Documentation viewer and system settings
- **Multi-threaded**: Separate threads for data reception and GUI updates

## File Organization Benefits

1. **Clean Structure**: Easy to navigate and understand project organization
2. **Separation of Concerns**: Code, data, and results are clearly separated
3. **Scalability**: Easy to add new features and modules
4. **Maintainability**: Configuration and code are properly organized
5. **User Friendly**: Single GUI interface for all system functions

## Key Features

### System Control Center
- **GUI Interface**: Professional tkinter-based interface
- **Real-time Plotting**: 8 EMG channels + 6 IMU channels
- **Data Management**: CSV format with automatic saving
- **Analysis Tools**: Built-in data analysis and statistics
- **System Configuration**: Easy parameter adjustment
- **Documentation Integration**: Direct access to system documentation

### Data Processing
- **Multi-threaded**: Efficient data reception and processing
- **Real-time**: Live data visualization and monitoring
- **Error Handling**: Comprehensive exception handling and logging
- **Format Support**: Standard CSV format for easy analysis