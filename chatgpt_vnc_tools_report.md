# VNC Tools File - Need ChatGPT Code Review

## Problem Summary
We implemented ChatGPT's bulletproof approach for typing complex text with quotes through VNC, but it's partially working and has interference issues. Simple text types perfectly, but complex text with quotes fails with `ord() expected a character, but string of length 9 found`.

## What We Implemented (Following ChatGPT's Advice)

### 1. Bypass Bash Processing ✅
- Modified `call_tool` to pipe JSON directly to Python: `echo "$ARGS" | python vnc_tools.py keyboard --stdin-json`
- No more bash variable processing or JSON re-parsing in shell

### 2. String-Based JSON Parsing ✅  
- Implemented custom JSON text extraction using string parsing (no JSON library)
- Properly handles escape sequences: `\\"` → `"`, `\\!` → `!` (from bash history expansion)
- Successfully extracts correct text: `'print("Hello world!")'`

### 3. Keysym-Based Typing ✅ (Partially)
- Implemented `SYMBOL_KEYSYM` mapping: `'"': 'quotedbl'`, `'!': 'exclam'`, etc.
- Uses `client.keyPress(keysym)` for punctuation (no Shift chords needed)  
- Uses `client.type(ch)` for letters/digits

## Current Status

### ✅ **What Works:**
- Simple text: `{"text": "hello"}` → types "hello" perfectly
- JSON extraction: Correctly extracts `'print("Hello world!")'` from bash-mangled input
- Pipeline: stdin → string parsing → text extraction all working

### ❌ **What Fails:**
- Complex text: `{"text": "print(\"Hello world!\")"}` → only types "print" then fails
- Error: `ord() expected a character, but string of length 9 found`
- The "length 9" suggests "Hello wor" - indicating old code is processing multi-char strings

## Code Structure Issues

The `vnc_tools.py` file has **multiple keyboard processing paths** that may be interfering:

1. **New `--stdin-json` path** (our ChatGPT implementation)
2. **Old `--stdin-text` path** 
3. **Old `keyboard_file` path**
4. **Legacy `keyboard` path** with base64/shift-chord logic

**Suspected Issue:** The old shift-chord processing code is still being triggered somehow and trying to process multi-character strings instead of letting our new keysym approach handle individual characters.

## Specific Questions for ChatGPT

1. **Where is the ord() error coming from?** The error suggests old code is trying to process "Hello world" as a single string instead of character-by-character.

2. **Are multiple code paths interfering?** We have 4 different keyboard processing methods - is the wrong one being called?

3. **Is the keysym typing function actually being reached?** We added debug output but don't see it, suggesting the error happens before our new typing function.

4. **Should we remove the old keyboard processing entirely?** Clean up the legacy code that's causing conflicts.

## What We Need

**Clean, working implementation** of ChatGPT's approach:
- Stdin JSON → string parsing → keysym typing
- Remove/disable interfering legacy code paths  
- Ensure complex text like `print("Hello world!")` types completely

## Current Working Directory
The file is at `/home/jacob/partition/qemu-tools/vnc_tools.py`

## Test Cases
- ✅ Works: `echo '{"text": "hello"}' | ./call_tool send_keyboard`
- ❌ Fails: `echo '{"text": "print(\"Hello world!\")"}' | ./call_tool send_keyboard`
- Should type: `print("Hello world!")` but only types `print` then errors

**The core ChatGPT approach is sound - we just need to debug why the old code is interfering and ensure only the new keysym path executes.**