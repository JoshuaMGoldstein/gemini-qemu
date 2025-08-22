# VNC Text Input Problem - Need Solution

## Core Issue
We have a VNC-based UI automation system that can successfully click coordinates and handle simple text, but we CANNOT reliably type complex text with quotes, especially code like `print("Hello world!")`. 

## What Works
- ✅ Simple text: `"hello"` types perfectly
- ✅ Text without quotes: `print(hello)` works fine  
- ✅ Mouse clicks work flawlessly
- ✅ Key combinations work (Ctrl+S, Alt+F, etc.)
- ✅ VNC connection is solid

## What Fails
- ❌ Text with quotes: `print("Hello world!")` only types `print(`
- ❌ Any complex strings with nested quotes
- ❌ Code with string literals

## System Architecture
```
Gemini → call_tool (bash) → vnc_tools.py (Python) → vncdotool → VNC → VM
```

## Approaches We've Tried

### 1. JSON Parsing (FAILED)
**Problem**: Bash interprets escape sequences before JSON parsing
```bash
echo '{"text": "print(\"Hello!\")"}' | ./call_tool send_keyboard
```
**Result**: JSON parsing errors on `\!` and doubled backslashes

### 2. Base64 Encoding (FAILED) 
**Problem**: Bash still interprets content before base64 encoding
```bash
ENCODED_ARGS=$(echo "$ARGS" | base64 -w 0)  # Bash processes $ARGS first
```
**Result**: Same escape sequence issues

### 3. Temporary Files (FAILED)
**Problem**: Even writing to files, bash processes the content
```bash
printf '%s' "$ARGS" > "$TEMP_FILE"  # Still bash interpretation
```
**Result**: Escape sequences get mangled

### 4. String Parsing Instead of JSON (FAILED)
**Problem**: Complex manual parsing of quotes and escapes
```python
# Extract text between "text": "..." manually
```
**Result**: `ord() expected a character, but string of length 10 found` errors

## Current Error Pattern
When we send: `print("Hello world!")`
We get errors like:
- `ord() expected a character, but string of length 10 found`
- Only `print(` gets typed
- Success reported but partial execution

## Key Insights
1. **Bash is the enemy** - Any bash processing corrupts the input
2. **JSON parsing is fragile** - Escape sequences break it every time
3. **The VNC layer works** - Simple text types perfectly
4. **The issue is data transport** - Not the actual typing mechanism

## Questions for ChatGPT
1. **How can we completely bypass bash processing** for arbitrary text input?
2. **Is there a way to pass raw binary data** through the tool chain?
3. **Should we use a different IPC mechanism** (named pipes, sockets, etc.)?
4. **Can we modify the tool interface** to avoid this processing entirely?
5. **What's the most robust way** to handle arbitrary text with any characters?

## System Requirements
- Must work with existing Gemini tool call interface
- Must handle ANY text input (code, quotes, special chars, etc.)
- Must be reliable and not fragile to edge cases
- Cannot change the basic VNC communication (that works perfectly)

## Current Tool Definition
Gemini calls: `send_keyboard` with JSON like `{"text": "print(\"hello\")"}`
The tool interface cannot be completely changed, but we can modify how it processes the input.

**We need a bulletproof way to get arbitrary text from Gemini's JSON to the VNC typing function without ANY bash interpretation corrupting it.**