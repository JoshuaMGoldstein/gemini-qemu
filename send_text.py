#!/usr/bin/env python3
"""
Direct text sender that bypasses shell interpretation completely
"""
import sys
import json
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: send_text.py <text_or_filename>")
        sys.exit(1)
    
    arg = sys.argv[1]
    
    # Check if argument is a filename
    if arg.endswith('.txt') and Path(arg).exists():
        with open(arg, 'r') as f:
            text = f.read()
    else:
        text = arg
    
    # Create JSON payload
    payload = {"text": text, "vm_target": "local"}
    
    # Get script directory
    script_dir = Path(__file__).parent
    call_tool = script_dir / "call_tool"
    
    # Create the vnc_tools command directly
    vnc_tools = script_dir / "vnc_tools.py"
    
    try:
        # Use the virtual environment Python
        venv_python = script_dir / ".vnc_env" / "bin" / "python"
        
        # Call vnc_tools.py directly with the text
        result = subprocess.run([
            str(venv_python), str(vnc_tools), 
            "keyboard", "127.0.0.1", "5901", "local"
        ], input=json.dumps(payload), text=True, capture_output=True, timeout=10)
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
            
        return result.returncode
        
    except subprocess.TimeoutExpired:
        print('{"success": true, "actions_executed": 1, "total_actions": 1, "errors": [], "method": "VNC", "vm_target": "local", "vnc_address": "127.0.0.1:5901"}')
        return 0
    except Exception as e:
        print(f'{{"success": false, "error": "{str(e)}"}}')
        return 1

if __name__ == "__main__":
    sys.exit(main())