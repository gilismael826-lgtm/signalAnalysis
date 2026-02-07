#!/usr/bin/env python3
"""
EMG/IMU Signal Analysis System - Main Entry Point

This script provides a command-line interface to run the system control center.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    parser = argparse.ArgumentParser(
        description='EMG/IMU Signal Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run system control center
  python main.py
  
  # Run system control center (explicit)
  python main.py --control
  
  # List available serial ports
  python main.py --list-ports
        """
    )
    
    parser.add_argument('--control', action='store_true',
                       help='Run system control center (default)')
    parser.add_argument('--list-ports', action='store_true',
                       help='List available serial ports')
    
    args = parser.parse_args()
    
    # List serial ports if requested
    if args.list_ports:
        list_serial_ports()
        return
    
    # Run system control center
    run_control_center()

def list_serial_ports():
    """List available serial ports"""
    try:
        import serial.tools.list_ports
        
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("No serial ports found.")
            return
            
        print("Available serial ports:")
        for port in ports:
            print(f"  {port.device}: {port.description}")
            
    except ImportError:
        print("pyserial not installed. Please install it first:")
        print("pip install pyserial")

def run_control_center():
    """Run system control center"""
    try:
        print("Running system control center...")
        print("Note: System control center will open in a separate GUI window")
        
        # System control center needs to be run as a separate process
        # because it uses tkinter which requires its own main loop
        import subprocess
        import sys
        
        # Run the system control center script directly
        # Use Popen instead of run to avoid blocking
        process = subprocess.Popen([sys.executable, "src/system_control.py"])
        print(f"System control center started with PID: {process.pid}")
        print("Close the GUI window to exit the program")
        return process
            
    except FileNotFoundError as e:
        print(f"Error: system_control.py not found: {e}")
        print("Make sure that file exists in src/ directory")
    except Exception as e:
        print(f"Error running system control center: {e}")

if __name__ == '__main__':
    main()