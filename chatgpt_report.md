# Shell Interpretation Issue - Need Advice

## Problem Summary
We have a VNC-based UI automation system that can successfully type text into applications, but we're encountering a persistent issue where the `#` character gets converted to `3` during shell processing, preventing us from typing Python comments and other content with hash symbols.

## System Architecture
- **Shell script**: `call_tool` - receives JSON input and routes to Python
- **Python script**: `vnc_tools.py` - handles VNC keyboard input via vncdotool
- **Target**: QEMU VM running Puppy Linux with Geany IDE open
- **Goal**: Type complex Python programs including comments (# character)

## What We've Tried

### 1. Basic JSON Approach (FAILED)
```bash
echo '{"text": "# Python comment"}' | ./call_tool send_keyboard
```
**Result**: Only "3 Python comment" appears - `#` becomes `3`

### 2. Temporary File Approach (FAILED)
Modified call_tool to write JSON to temp file:
```bash
TEMP_FILE=$(mktemp)
echo "$ARGS" > "$TEMP_FILE"
```
**Result**: Still converts `#` to `3` during echo

### 3. Direct stdin with cat (FAILED)
```bash
cat > "$TEMP_FILE"  # Read directly from stdin
```
**Result**: Didn't work at all, no text typed

### 4. Base64 Encoding (FAILED)
```bash
ENCODED_ARGS=$(echo "$ARGS" | base64 -w 0)
```
**Result**: Still converts `#` to `3` during initial echo

### 5. Direct Python Script Bypass (FAILED)
Created `send_text.py` that calls vnc_tools.py directly:
```bash
python3 send_text.py "# comment"
```
**Result**: Shell still interprets `#` as comment start before Python gets it

### 6. File-Based Input (FAILED)
Created text file with Python program, read from file:
```bash
python3 send_text.py test_program.txt
```
**Result**: Even when reading from file, `#` becomes `3`

## Current Code Structure

### call_tool (bash script)
```bash
"send_keyboard")
    # Use base64 encoding to bypass all shell interpretation
    ENCODED_ARGS=$(echo "$ARGS" | base64 -w 0)
    # ... rest of processing
```

### vnc_tools.py (Python)
```python
elif command == "keyboard":
    # Handle base64 encoded input if provided as argument
    if len(sys.argv) > 5:
        import base64
        encoded_data = sys.argv[5]
        decoded_json = base64.b64decode(encoded_data).decode('utf-8')
        data = json.loads(decoded_json)
    # ... rest of processing
```

## Key Observations
1. **Shell interpretation happens VERY early** - even in variable assignments
2. **Multiple layers** of shell processing are occurring
3. **The `#` → `3` conversion** suggests shell comment processing or history expansion
4. **File-based approaches still fail** - indicates the issue might be in how bash handles the initial command line
5. **Base64 encoding fails** because the `#` gets converted before encoding

## Questions for ChatGPT
1. **Why does `#` become `3`?** Is this bash history expansion, comment processing, or something else?
2. **How can we completely bypass shell interpretation** for arbitrary text content?
3. **Should we use a different IPC mechanism?** Named pipes, sockets, etc.?
4. **Is there a way to escape or quote** that would prevent this conversion?
5. **Could we modify the approach entirely?** Maybe use a different method to send keyboard input?

## System Details
- **OS**: Linux (Arch)
- **Shell**: bash
- **Python**: 3.13
- **vncdotool**: Working correctly for actual VNC communication
- **Target VM**: QEMU with VNC on port 5901

## Success Cases
- Simple text without `#` works perfectly
- Key combinations (Ctrl+A) work correctly
- Mouse clicks work flawlessly
- VNC communication is solid

The core VNC automation works great - we just need to solve this shell interpretation issue to type complex code with comments.