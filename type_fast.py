#!/usr/bin/env python3
"""
Fast multi-line typing by sending all actions in one command
"""
import sys
import subprocess
import json
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: type_fast.py 'line1\\nline2\\nline3'")
        sys.exit(1)
    
    # Decode escape sequences
    text = sys.argv[1].encode().decode('unicode_escape')
    lines = text.split('\n')
    
    # Build actions array
    actions = []
    
    for i, line in enumerate(lines):
        if line:  # Add text action for non-empty lines
            actions.append({"action": "type", "text": line})
        
        # Add Enter action except after last line
        if i < len(lines) - 1:
            actions.append({"action": "key", "key": "Return"})
    
    # Send all actions in one command
    script_dir = Path(__file__).parent
    call_tool = script_dir / "call_tool"
    
    payload = json.dumps({"actions": actions})
    
    try:
        result = subprocess.run([
            str(call_tool), "send_keyboard"
        ], input=payload, text=True, capture_output=True, timeout=10)
        
        if result.returncode == 0:
            print("Success!")
        else:
            print(f"Failed: {result.stderr}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()