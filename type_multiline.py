#!/usr/bin/env python3
"""
Helper script to type multi-line text by sending each line separately
"""
import sys
import subprocess
import time
from pathlib import Path

def send_line(line):
    """Send a single line of text"""
    script_dir = Path(__file__).parent
    call_tool = script_dir / "call_tool"
    
    # Create JSON payload for single line
    import json
    payload = json.dumps({"text": line})
    
    try:
        result = subprocess.run([
            str(call_tool), "send_keyboard"
        ], input=payload, text=True, capture_output=True, timeout=5)
        
        return result.returncode == 0
    except:
        return False

def send_enter():
    """Send Enter key"""
    script_dir = Path(__file__).parent
    call_tool = script_dir / "call_tool"
    
    import json
    payload = json.dumps({"actions": [{"action": "key", "key": "Return"}]})
    
    try:
        result = subprocess.run([
            str(call_tool), "send_keyboard"
        ], input=payload, text=True, capture_output=True, timeout=5)
        
        return result.returncode == 0
    except:
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: type_multiline.py 'line1\\nline2\\nline3'")
        sys.exit(1)
    
    # Decode escape sequences
    text = sys.argv[1].encode().decode('unicode_escape')
    lines = text.split('\n')
    
    print(f"Typing {len(lines)} lines...")
    
    for i, line in enumerate(lines):
        if line or i == 0:  # Type content or first line (even if empty)
            print(f"Line {i+1}: {repr(line)}")
            if not send_line(line):
                print(f"Failed to send line {i+1}")
                break
        
        # Add Enter except after last line
        if i < len(lines) - 1:
            if not send_enter():
                print(f"Failed to send Enter after line {i+1}")
                break
            time.sleep(0.1)  # Moderate delay between lines
    
    print("Complete!")

if __name__ == "__main__":
    main()