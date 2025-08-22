#!/usr/bin/env python3
"""
VNC tools for Gemini CLI using vncdotool
Handles screenshot capture, mouse clicks, and keyboard input via VNC protocol
"""

import sys
import json
import time
import tempfile
import subprocess
from typing import List, Dict, Any
from pathlib import Path

# Try to import vncdotool, provide clear error if not available
try:
    from vncdotool import api
    from PIL import Image
except ImportError as e:
    print(json.dumps({
        "success": False, 
        "error": f"Missing required library: {e}. Install with: pip install vncdotool pillow"
    }))
    sys.exit(1)

# Import the existing OpenRouter implementation
try:
    from openrouter import call_chat_vision
except ImportError:
    def call_chat_vision(**kwargs):
        return {"error": "OpenRouter module not available"}

def _compress_png_to_jpeg(image_bytes: bytes, quality: int = 85) -> bytes:
    """Compress PNG to JPEG to reduce payload size, exactly like the original code"""
    from PIL import Image
    import io
    
    # Load PNG from bytes
    png_img = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if needed (JPEG doesn't support transparency)
    if png_img.mode in ('RGBA', 'LA'):
        # Create white background for transparency
        background = Image.new('RGB', png_img.size, (255, 255, 255))
        if png_img.mode == 'RGBA':
            background.paste(png_img, mask=png_img.split()[-1])  # Use alpha channel as mask
        else:
            background.paste(png_img)
        png_img = background
    elif png_img.mode != 'RGB':
        png_img = png_img.convert('RGB')
    
    # Save as JPEG with compression
    jpeg_buffer = io.BytesIO()
    png_img.save(jpeg_buffer, format='JPEG', quality=quality, optimize=True)
    return jpeg_buffer.getvalue()

def _call_openrouter_vision(image_bytes: bytes) -> str:
    """Call OpenRouter vision API using two-pass TSV system"""
    try:
        # Get image dimensions  
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes))
        full_width, full_height = img.size
        print(f"🔍 Vision model receives: {full_width}x{full_height} image", file=sys.stderr)
        
        # Pass A: Get windows
        pass_a_result = _pass_a_windows_tsv(image_bytes, full_width, full_height)
        
        if not pass_a_result['success']:
            return f"Pass A failed: {pass_a_result.get('error', 'Unknown error')}"
        
        # Parse windows and taskbars using TSV format
        windows, taskbars = _parse_windows_tsv(pass_a_result['content'])
        if not windows and not taskbars:
            return "No windows or taskbars detected in Pass A"
        
        print(f"🔍 Found {len(windows)} windows and {len(taskbars)} taskbars", file=sys.stderr)
        
        # Now use the existing Pass B analysis for each window and taskbar
        results = []
        
        # Process windows
        for window in windows:
            # Use exact title bar dimensions from Pass A model
            # Title bar coordinates: ty1 to ty2 (e.g., 0 to 23)
            yT = window['ty1']  # Use exact title bar top
            yB = window['ty2']  # Use exact title bar bottom
            
            # Use full title bar width for complete context
            xL = window['x1']  # Left edge of window
            xR = window['x2']  # Right edge of window
            
            print(f"🔍 Window {window['id']}: bounds=({window['x1']},{window['y1']},{window['x2']},{window['y2']}) titlebar=({window['ty1']},{window['ty2']}) side={window['side']}", file=sys.stderr)
            print(f"🔍 ROI calculated: ({xL},{yT},{xR},{yB}) size={xR-xL}x{yB-yT}", file=sys.stderr)
            
            # Pass B: Analyze chrome in ROI
            pass_b_result = _pass_b_chrome_roi(image_bytes, xL, yT, xR, yB, window['ty1'], window['ty2'])
            
            if pass_b_result['success']:
                print(f"🔍 Pass B response: {pass_b_result['content'][:200]}", file=sys.stderr)
                
                # Try normalized parsing first
                roi_width = xR - xL
                roi_height = yB - yT
                size_ok, buttons = parse_normalized_tsv(pass_b_result['content'], roi_width, roi_height)
                
                if not size_ok:
                    print(f"🔍 SIZE gate failed, falling back to legacy parsing", file=sys.stderr)
                    buttons = _parse_chrome_tsv(pass_b_result['content'])
                else:
                    print(f"🔍 SIZE gate passed, using normalized coordinates", file=sys.stderr)
                
                # Convert LOCAL ROI coordinates to GLOBAL coordinates
                for role in buttons:
                    for button in buttons[role]:
                        x, y, w, h = button['vbbox']
                        # Add ROI offset to get global coordinates
                        print(f"🔍 Converting local to global: ({x},{y}) + ({xL},{yT}) -> ({x+xL},{y+yT})", file=sys.stderr)
                        button['vbbox'] = (x + xL, y + yT, w, h)
                        if button['ibbox']:
                            ix, iy, iw, ih = button['ibbox']
                            button['ibbox'] = (ix + xL, iy + yT, iw, ih)
                
                print(f"🔍 Buttons with global coords: {buttons}", file=sys.stderr)
                
                # Generate click coordinates for chrome buttons
                for role, role_name in [('C', 'close_button'), ('M', 'maximize_button'), ('N', 'minimize_button')]:
                    if buttons[role]:
                        click_coords = _click_chrome(buttons[role])
                        if click_coords:
                            print(f"🔍 {role_name}: click at {click_coords}", file=sys.stderr)
                            results.append(f"{role_name}:button - click {click_coords} - window {window['id']}")
        
        # Process taskbars - just add raw vision output without complex parsing
        for taskbar in taskbars:
            print(f"🔍 Taskbar {taskbar['id']}: bounds=({taskbar['x1']},{taskbar['y1']},{taskbar['x2']},{taskbar['y2']})", file=sys.stderr)
            
            # Analyze taskbar ROI (full width, but crop height if needed)
            xL = taskbar['x1']
            xR = taskbar['x2'] 
            yT = taskbar['y1']
            yB = taskbar['y2']
            
            print(f"🔍 Taskbar ROI calculated: ({xL},{yT},{xR},{yB}) size={xR-xL}x{yB-yT}", file=sys.stderr)
            
            # Pass B: Analyze taskbar icons in ROI
            pass_b_result = _pass_b_taskbar_roi(image_bytes, xL, yT, xR, yB)
            
            if pass_b_result['success']:
                print(f"🔍 Taskbar Pass B response: {pass_b_result['content'][:200]}", file=sys.stderr)
                icons = _parse_taskbar_tsv(pass_b_result['content'])
                
                # Convert LOCAL ROI coordinates to GLOBAL coordinates
                for icon in icons:
                    x, y, w, h = icon['bbox']
                    print(f"🔍 Converting taskbar local to global: ({x},{y}) + ({xL},{yT}) -> ({x+xL},{y+yT})", file=sys.stderr)
                    global_x = x + xL
                    global_y = y + yT
                    
                    # Calculate click point
                    click_x = global_x + w // 2
                    click_y = global_y + h // 2
                    
                    print(f"🔍 {icon['role']}:{icon['name']} - click at ({click_x}, {click_y})", file=sys.stderr)
                    results.append(f"{icon['role']}:{icon['name']} - click ({click_x}, {click_y}) - taskbar {taskbar['id']}")
        
        return '\n'.join(results) if results else "No buttons or icons detected"
        
    except Exception as e:
        # Fallback to simple analysis if analysis fails
        return _call_simple_vision_analysis(image_bytes)


def _format_vision_output_with_centers(raw_content: str) -> str:
    """Process raw vision output and add pre-calculated center coordinates"""
    formatted_lines = []
    lines = raw_content.strip().split('\n')
    
    current_element = {}
    
    for line in lines:
        line = line.strip()
        
        # Parse the structured format the model actually returns
        if line.startswith('- TYPE:'):
            # Start of new element
            if current_element:
                # Process previous element
                formatted_line = _format_element(current_element)
                if formatted_line:
                    formatted_lines.append(formatted_line)
            current_element = {'type': line.replace('- TYPE:', '').strip()}
        elif line.startswith('- TEXT:'):
            current_element['text'] = line.replace('- TEXT:', '').strip()
        elif line.startswith('- X,Y:'):
            coords = line.replace('- X,Y:', '').strip()
            if ',' in coords:
                try:
                    x, y = map(int, coords.split(','))
                    current_element['x'] = x
                    current_element['y'] = y
                except ValueError:
                    pass
        elif line.startswith('- W,H:'):
            size = line.replace('- W,H:', '').strip()
            if ',' in size:
                try:
                    w, h = map(int, size.split(','))
                    current_element['w'] = w
                    current_element['h'] = h
                except ValueError:
                    pass
        elif line and not line.startswith('#') and not line.startswith('*'):
            # Keep other lines as-is (headers, etc.)
            formatted_lines.append(line)
    
    # Process the last element
    if current_element:
        formatted_line = _format_element(current_element)
        if formatted_line:
            formatted_lines.append(formatted_line)
    
    return '\n'.join(formatted_lines)

def _format_element(element: dict) -> str:
    """Format a single element with optimized click coordinates"""
    if 'type' in element and 'x' in element and 'y' in element and 'w' in element and 'h' in element:
        x, y, w, h = element['x'], element['y'], element['w'], element['h']
        text = element.get('text', '')
        type_name = element['type']
        
        # Calculate click point using improved algorithm
        click_x, click_y = _calculate_click_point(x, y, w, h, type_name)
        
        # Format: TYPE:TEXT - click (click_x, click_y) - box (x,y,w,h)
        return f"{type_name}:{text} - click ({click_x}, {click_y}) - box ({x},{y},{w},{h})"
    return None

def _calculate_click_point(x: int, y: int, w: int, h: int, element_type: str) -> tuple:
    """Calculate optimal click point for UI elements using ChatGPT's improved algorithm"""
    W, H = 800, 600  # Screen dimensions
    
    # Inclusive center (no +0.5 bias on even sizes)
    cx = x + (w - 1) // 2
    cy = y + (h - 1) // 2
    
    # For title-bar controls, apply left-lean to avoid right padding/border
    is_title_control = element_type in {'close_button', 'maximize_button', 'minimize_button'}
    is_top_right_small = (x > W * 0.65 and y < H * 0.20 and w <= 24 and h <= 24)
    
    if is_title_control or is_top_right_small:
        # 2-3 px left bias avoids right border, stay within bounds
        cx = max(x + 2, cx - 3)
        cy = max(y + 2, min(y + h - 3, cy))
    
    return cx, cy

def _create_debug_image_with_crosshairs(image_path: str, click_points: list) -> str:
    """Create debug image with crosshairs at click points"""
    from PIL import Image, ImageDraw
    import tempfile
    
    # Open the screenshot
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Draw crosshairs for each click point
    for i, (x, y, element_type, text) in enumerate(click_points):
        # Use different colors for different element types
        color = 'red' if 'close' in element_type else 'blue' if 'button' in element_type else 'green'
        
        # Draw crosshair (+ shape)
        size = 5
        draw.line([(x-size, y), (x+size, y)], fill=color, width=2)  # Horizontal line
        draw.line([(x, y-size), (x, y+size)], fill=color, width=2)  # Vertical line
        
        # Draw small circle at center
        draw.ellipse([(x-1, y-1), (x+1, y+1)], fill=color)
    
    # Save debug image
    debug_path = tempfile.mktemp(suffix='_debug.png')
    img.save(debug_path)
    return debug_path


def _run_optimized_two_pass_analysis(image_bytes: bytes) -> dict:
    """Run optimized two-pass analysis for 15-25 second performance target"""
    import sys
    import io
    import math
    import random
    import concurrent.futures
    import threading
    from pathlib import Path
    from PIL import Image
    
    try:
        # Add the bin directory to Python path for imports
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        
        from openrouter import call_chat_vision
        
        # Get image dimensions
        img = Image.open(io.BytesIO(image_bytes))
        full_width, full_height = img.size
        
        # PASS A: Full resolution layout sweep
        import time
        start_time = time.time()
        
        pass_a_start = time.time()
        pass_a_result = _pass_a_windows_tsv(image_bytes, full_width, full_height)
        pass_a_end = time.time()
        
        print(f"⏱️ Pass A (Gemini 2.0 Flash) took: {pass_a_end - pass_a_start:.1f} seconds", file=sys.stderr)
        
        if not pass_a_result['success']:
            return {"success": False, "error": f"Pass A failed: {pass_a_result.get('error', 'Unknown')}"}
        
        # Parse Pass A results (simple format)
        try:
            content = pass_a_result['content']
            containers = []
            direct_clickables = []  # For window controls like close/minimize/maximize
            lines = content.strip().split('\n')
            
            print(f"🔍 Pass A content: {repr(content[:200])}", file=sys.stderr)
            print(f"🔍 Lines found: {len(lines)}", file=sys.stderr)
            
            for i, line in enumerate(lines[:30]):  # Max 30 elements
                line = line.strip()
                
                # Clean up the line: remove numbers, asterisks, and extra formatting
                line = line.split('. ', 1)[-1]  # Remove "1. " prefix
                line = line.replace('**', '')   # Remove markdown bold
                line = line.strip()
                
                if ':' in line and '@' in line:
                    try:
                        # Handle format: type:text@x,y,w,h
                        type_text, coords = line.split('@')
                        type_part, text_part = type_text.split(':', 1)
                        coords_clean = coords.replace('**', '').strip()
                        coord_parts = coords_clean.split(',')
                        
                        if len(coord_parts) == 4:
                            # Bounding box format: X,Y,W,H
                            x, y, w, h = map(int, coord_parts)
                            # Use improved click point calculation
                            center_x, center_y = _calculate_click_point(x, y, w, h, type_part.strip())
                            bounds = [x, y, x + w, y + h]
                        else:
                            # Legacy format: X,Y (center point)
                            x, y = map(int, coord_parts)
                            center_x = x
                            center_y = y
                            bounds = [x-50, y-25, x+50, y+25]
                        
                        element = {
                            'id': f'e{i+1}',
                            'type': type_part.strip(),
                            'text': text_part.strip(),
                            'center': [center_x, center_y],
                            'bounds': bounds
                        }
                        
                        # Window controls are direct clickables, others are containers for further analysis
                        if type_part.strip() in ['close_button', 'minimize_button', 'maximize_button']:
                            direct_clickables.append(element)
                            print(f"✅ Direct clickable: {type_part}:{text_part}@{x},{y}", file=sys.stderr)
                        else:
                            containers.append(element)
                            print(f"✅ Container: {type_part}:{text_part}@{x},{y}", file=sys.stderr)
                    except Exception as parse_err:
                        print(f"❌ Parse error for line '{line}': {parse_err}", file=sys.stderr)
                        continue
                else:
                    if line and not line.startswith('🔍'):
                        print(f"⚠️ Skipped line: '{line}' (no : or @)", file=sys.stderr)
                        
        except Exception as e:
            return {"success": False, "error": f"Pass A parse failed: {str(e)}"}
        
        # No scaling needed - already using full resolution
        
        # PASS B: Parallel per-container detail with hard cropping
        valid_containers = []
        for container in containers:
            container_bounds = container.get('bounds', [0, 0, full_width, full_height])
            if len(container_bounds) == 4:
                x1, y1, x2, y2 = container_bounds
                area = (x2 - x1) * (y2 - y1)
                if area >= 1000:  # Skip tiny containers
                    valid_containers.append(container)
        
        pass_b_results = []
        
        # Parallel processing with maximum workers for speed
        max_workers = max(1, min(6, len(valid_containers)))  # Ensure at least 1 worker
        
        def analyze_container_wrapper(container):
            container_id = container.get('id', 'unknown')
            container_bounds = container.get('bounds', [0, 0, full_width, full_height])
            
            return _pass_b_container_detail_cropped(
                image_bytes, container_id, container_bounds, 
                full_width, full_height, 0  # No delay for maximum speed
            )
        
        pass_b_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit ALL container analyses at once
            futures = [executor.submit(analyze_container_wrapper, container) for container in valid_containers]
            
            # Wait for ALL to complete using wait() instead of as_completed()
            done_futures, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
        pass_b_end = time.time()
        
        print(f"⏱️ Pass B (Gemini 2.0 Flash {len(valid_containers)} containers) took: {pass_b_end - pass_b_start:.1f} seconds", file=sys.stderr)
        
        # Process all results together
        for future in done_futures:
            try:
                detail_result = future.result()
                if detail_result['success']:
                    # Parse simple format for Pass B
                    try:
                        content = detail_result['content']
                        elements = []
                        lines = content.strip().split('\n')
                        
                        for j, line in enumerate(lines[:15]):  # Max 15 per container
                            if ':' in line and '@' in line:
                                try:
                                    type_text, coords = line.split('@')
                                    type_part, text_part = type_text.split(':', 1)
                                    x, y = map(int, coords.split(','))
                                    
                                    elements.append({
                                        'id': f'b{j+1}',
                                        'type': type_part.strip(),
                                        'text': text_part.strip(),
                                        'center': [x, y],
                                        'bounds': [x-25, y-15, x+25, y+15]
                                    })
                                except:
                                    continue
                        
                        if elements:
                            pass_b_results.append(elements)
                    except:
                        continue
            except Exception as e:
                print(f"Container analysis failed: {e}", file=sys.stderr)
        
        # Merge all results (no scaling needed - using full resolution)
        all_elements = _merge_results_optimized(containers, pass_b_results, 1.0, 1.0)
        
        return {
            "success": True,
            "screen_width": full_width,
            "screen_height": full_height,
            "raw_vision_output": pass_a_result['content'],
            "approach": "full_raw_output"
        }
        
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


def _downscale_image(image_bytes: bytes, target_width: int = 960, target_height: int = 600) -> tuple:
    """Downscale image for Pass A with scaling factors for coordinate mapping"""
    import io
    from PIL import Image
    
    try:
        img = Image.open(io.BytesIO(image_bytes))
        original_width, original_height = img.size
        
        scale_x = original_width / target_width
        scale_y = original_height / target_height
        
        downscaled_img = img.resize((target_width, target_height), Image.LANCZOS)
        
        output = io.BytesIO()
        downscaled_img.save(output, format='PNG')
        downscaled_bytes = output.getvalue()
        
        return downscaled_bytes, scale_x, scale_y
    except Exception as e:
        return image_bytes, 1.0, 1.0


def _crop_container_image(image_bytes: bytes, container_bounds: list, padding: int = 20) -> bytes:
    """Hard crop image to container bounds + padding for Pass B"""
    import io
    from PIL import Image
    
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        x1, y1, x2, y2 = container_bounds
        
        img_width, img_height = img.size
        crop_x1 = max(0, x1 - padding)
        crop_y1 = max(0, y1 - padding)
        crop_x2 = min(img_width, x2 + padding)
        crop_y2 = min(img_height, y2 + padding)
        
        cropped_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        
        output = io.BytesIO()
        cropped_img.save(output, format='PNG')
        cropped_bytes = output.getvalue()
        
        return cropped_bytes
    except Exception as e:
        return image_bytes


def _pass_a_windows_tsv(image_bytes: bytes, width: int, height: int) -> dict:
    """Pass A: Windows only TSV format - lightweight window detection"""
    user_prompt = f"""You are analyzing an EXACT {width}x{height} screenshot. Do not rescale.
Return WINDOWS and TASKBAR in TSV lines (no prose).

FORMAT (TSV):
WIN    id    x1 y1 x2 y2    ty1 ty2    side(R|L)    conf
TASKBAR    id    x1 y1 x2 y2    conf

RULES
- WIN: ty1..ty2 = title-bar vertical band inside the window.
- WIN: side = where close/min/max live: R or L.
- TASKBAR: bottom panel with icons/buttons.
- Max 6 lines total. Integers only. No extra text."""

    try:
        response = call_chat_vision(
            model="qwen/qwen2.5-vl-32b-instruct",
            system_prompt="",
            user_text=user_prompt,
            image_bytes=image_bytes,
            temperature=0.0,
            timeout=60,
            extra={"max_tokens": 1000, "top_p": 1.0}
        )
        
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            return {"success": True, "content": content}
        else:
            return {"success": False, "error": "No content in response"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


def _pass_b_chrome_roi(image_bytes: bytes, xL: int, yT: int, xR: int, yB: int, ty1: int, ty2: int) -> dict:
    """Pass B: Chrome buttons ROI analysis with top-k candidates"""
    
    # Crop to ROI
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(image_bytes))
    roi_img = img.crop((xL, yT, xR, yB))
    
    # Convert back to bytes
    roi_buffer = io.BytesIO()
    roi_img.save(roi_buffer, format='JPEG', quality=100)
    roi_bytes = roi_buffer.getvalue()
    
    # DEBUG: Save ROI image
    debug_roi_path = f"/home/jacob/partition/qemu-tools/roi_debug_{xL}_{yT}_{xR}_{yB}.jpg"
    with open(debug_roi_path, 'wb') as f:
        f.write(roi_bytes)
    print(f"🔍 Saved ROI to: {debug_roi_path} (size: {roi_img.size})", file=sys.stderr)
    
    roi_width = xR - xL
    roi_height = yB - yT
    user_prompt = f"""You see ONE image (title-bar ROI). Do not mention any sizes.
Return ONLY TSV lines. No prose.

SIZE  {roi_width}  {roi_height}
BTN   role(C|M|N)  k  px py pw ph  conf
# px,py,pw,ph are normalized floats in [0,1]. k=1..2. Max 6 BTN lines.
# If unsure, lower confidence. Do not output values outside [0,1].
# Any line not starting with SIZE or BTN will be ignored."""
    
    print(f"🔍 DEBUG PROMPT TO VISION MODEL: {user_prompt}", file=sys.stderr)

    try:
        response = call_chat_vision(
            model="qwen/qwen2.5-vl-32b-instruct",
            system_prompt="",
            user_text=user_prompt,
            image_bytes=roi_bytes,
            temperature=0.0,
            timeout=60,
            extra={"max_tokens": 1000, "top_p": 1.0}
        )
        
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            return {"success": True, "content": content}
        else:
            return {"success": False, "error": "No content in response"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


def _pass_b_taskbar_roi(image_bytes: bytes, xL: int, yT: int, xR: int, yB: int) -> dict:
    """Pass B: Taskbar icons ROI analysis"""
    
    # Crop to taskbar ROI
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(image_bytes))
    roi_img = img.crop((xL, yT, xR, yB))
    
    # Convert back to bytes
    roi_buffer = io.BytesIO()
    roi_img.save(roi_buffer, format='JPEG', quality=100)
    roi_bytes = roi_buffer.getvalue()
    
    # DEBUG: Save ROI image
    debug_roi_path = f"/home/jacob/partition/qemu-tools/taskbar_roi_debug_{xL}_{yT}_{xR}_{yB}.jpg"
    with open(debug_roi_path, 'wb') as f:
        f.write(roi_bytes)
    print(f"🔍 Saved taskbar ROI to: {debug_roi_path} (size: {roi_img.size})", file=sys.stderr)
    
    roi_width = xR - xL
    roi_height = yB - yT
    user_prompt = f"""Taskbar {roi_width}×{roi_height}. Return TSV only:

ICON role name x y w h conf

Detect all clickable icons/buttons
x,y LOCAL (0,0 to {roi_width-1},{roi_height-1})
Max 10 lines"""

    try:
        response = call_chat_vision(
            model="qwen/qwen2.5-vl-32b-instruct",
            system_prompt="",
            user_text=user_prompt,
            image_bytes=roi_bytes,
            temperature=0.0,
            timeout=60,
            extra={"max_tokens": 2000, "top_p": 1.0}
        )
        
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            return {"success": True, "content": content}
        else:
            return {"success": False, "error": "No content in response"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


def _parse_windows_tsv(content: str) -> tuple:
    """Parse Pass A TSV output into window and taskbar data"""
    windows = []
    taskbars = []
    
    for line in content.strip().split('\n'):
        line = line.strip()
        if line.startswith('WIN'):
            parts = line.split()
            if len(parts) >= 9:
                try:
                    window = {
                        'type': 'window',
                        'id': parts[1],
                        'x1': int(parts[2]), 'y1': int(parts[3]),
                        'x2': int(parts[4]), 'y2': int(parts[5]),
                        'ty1': int(parts[6]), 'ty2': int(parts[7]),
                        'side': parts[8],
                        'conf': float(parts[9]) if len(parts) > 9 else 0.9
                    }
                    windows.append(window)
                except (ValueError, IndexError):
                    continue
        elif line.startswith('TASKBAR'):
            parts = line.split()
            if len(parts) >= 6:
                try:
                    taskbar = {
                        'type': 'taskbar',
                        'id': parts[1],
                        'x1': int(parts[2]), 'y1': int(parts[3]),
                        'x2': int(parts[4]), 'y2': int(parts[5]),
                        'conf': float(parts[6]) if len(parts) > 6 else 0.9
                    }
                    taskbars.append(taskbar)
                except (ValueError, IndexError):
                    continue
    
    return windows, taskbars


def norm_to_px(px, py, pw, ph, W, H):
    """Convert normalized coordinates to pixels with bounds clamping"""
    x = int(round(px * (W-1))) + 1
    y = int(round(py * (H-1))) + 4
    w = max(1, int(round(pw * W)))
    h = max(1, int(round(ph * H)))
    # Clamp to ROI bounds
    x = max(0, min(W-1, x))
    y = max(0, min(H-1, y))
    if x + w > W: w = W - x
    if y + h > H: h = H - y
    return x, y, w, h

def parse_normalized_tsv(content: str, W: int, H: int) -> tuple:
    """Parse normalized TSV with SIZE gate validation"""
    lines = content.strip().split('\n')
    size_ok = False
    buttons = {'C': [], 'M': [], 'N': []}
    
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
            
        if parts[0] == "SIZE":
            if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                size_ok = (int(parts[1]) == W and int(parts[2]) == H)
                print(f"🔍 SIZE gate: expected {W}x{H}, got {parts[1]}x{parts[2]}, valid: {size_ok}", file=sys.stderr)
            continue
            
        if parts[0] == "BTN" and len(parts) >= 8:
            try:
                role = parts[1]
                k = int(parts[2])
                px, py, pw, ph = map(float, parts[3:7])
                conf = float(parts[7]) if len(parts) > 7 else 0.9
                
                # Validate normalized coordinates are in [0,1]
                if not (0.0 <= px <= 1.0 and 0.0 <= py <= 1.0 and 0.0 <= pw <= 1.0 and 0.0 <= ph <= 1.0):
                    print(f"🔍 Invalid normalized coords for {role}: ({px},{py},{pw},{ph})", file=sys.stderr)
                    continue
                    
                # Convert to pixels with clamping
                x, y, w, h = norm_to_px(px, py, pw, ph, W, H)
                
                if role in buttons:
                    button = {
                        'role': role,
                        'k': k,
                        'vbbox': (x, y, w, h),
                        'ibbox': (x, y, w, h),
                        'conf': conf
                    }
                    buttons[role].append(button)
                    print(f"🔍 Parsed {role} from normalized: ({px:.3f},{py:.3f},{pw:.3f},{ph:.3f}) -> pixels ({x},{y},{w},{h})", file=sys.stderr)
                    
            except (ValueError, IndexError) as e:
                print(f"🔍 Parse error on BTN line '{line}': {e}", file=sys.stderr)
                continue
                
    return size_ok, buttons

def _parse_chrome_tsv(content: str) -> dict:
    """Parse Pass B output - legacy fallback, prefer normalized parsing"""
    buttons = {'C': [], 'M': [], 'N': []}
    
    # Remove markdown code blocks and prose
    content = content.replace('```tsv', '').replace('```', '').strip()
    
    # Try parsing markdown format first
    import re
    
    # Look for markdown button sections
    button_sections = re.findall(r'\*\*([^*]+)\s*Button.*?Role.*?`([CMN])`.*?x.*?([0-9]+).*?y.*?([0-9]+).*?w.*?([0-9]+).*?h.*?([0-9]+)', content, re.DOTALL | re.IGNORECASE)
    
    for section in button_sections:
        try:
            button_type, role, x, y, w, h = section
            button = {
                'role': role,
                'k': 1,
                'vbbox': (int(x), int(y), int(w), int(h)),
                'ibbox': None,  # Not using ibbox for now
                'conf': 1.0
            }
            buttons[role].append(button)
            print(f"🔍 Parsed {role} from markdown: local=({x},{y}) size=({w},{h})", file=sys.stderr)
        except (ValueError, IndexError) as e:
            print(f"🔍 Markdown parse error: {e}", file=sys.stderr)
            continue
    
    # If markdown parsing found buttons, return those
    if any(buttons[role] for role in buttons):
        return buttons
    
    # Fall back to TSV parsing if no markdown buttons found
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('BTN role') or line.startswith('Here'):
            continue
            
        # Try parsing both BTN format and simple TSV format
        parts = line.split('\t') if '\t' in line else line.split()
        
        if line.startswith('BTN') and len(parts) >= 11:
            # Original BTN format
            try:
                button = {
                    'role': parts[1],
                    'k': int(parts[2]),
                    'vbbox': (int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])),
                    'ibbox': (int(parts[7]), int(parts[8]), int(parts[9]), int(parts[10])) if parts[7] != '0' else None,
                    'conf': float(parts[11]) if len(parts) > 11 else 0.9
                }
                if button['role'] in buttons:
                    buttons[button['role']].append(button)
            except (ValueError, IndexError):
                continue
        elif len(parts) >= 5:
            # Simple TSV format: role x y w h [ix iy iw ih conf]
            try:
                role = parts[0]
                # Map text roles to single letters
                if role in ['close', 'C']:
                    role = 'C'
                elif role in ['min', 'minimize', 'N']:
                    role = 'N'  
                elif role in ['max', 'maximize', 'M']:
                    role = 'M'
                    
                if role in ['C', 'M', 'N']:
                    button = {
                        'role': role,
                        'k': 1,
                        'vbbox': (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])),
                        'ibbox': (int(parts[5]), int(parts[6]), int(parts[7]), int(parts[8])) if len(parts) > 8 and parts[5] != '0' else None,
                        'conf': float(parts[9]) if len(parts) > 9 else 0.9
                    }
                    buttons[role].append(button)
                    print(f"🔍 Parsed {role}: local=({parts[1]},{parts[2]}) size=({parts[3]},{parts[4]})", file=sys.stderr)
            except (ValueError, IndexError) as e:
                print(f"🔍 Parse error on line '{line}': {e}", file=sys.stderr)
                continue
                
    return buttons


def _parse_taskbar_tsv(content: str) -> list:
    """Parse taskbar TSV output into icon data"""
    icons = []
    
    # Remove markdown code blocks and prose
    content = content.replace('```', '').strip()
    
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('ICON role') or line.startswith('Here'):
            continue
            
        parts = line.split('\t') if '\t' in line else line.split()
        
        # Skip header lines and empty lines
        if line.startswith('ICON role') or line.startswith('```') or not line.strip():
            continue
            
        # Try to parse lines with clickable elements
        try:
            # Handle various formats by looking for numeric coordinates
            coords_found = False
            for i in range(len(parts)-4):
                try:
                    # Look for a sequence of 4 integers (x, y, w, h)
                    x = int(parts[i])
                    y = int(parts[i+1]) 
                    w = int(parts[i+2])
                    h = int(parts[i+3])
                    
                    # Found coordinates, determine role and name
                    if i >= 2:
                        role = parts[i-2] if parts[i-2] != parts[i-1] else parts[i-1]
                        name = parts[i-1]
                    elif i >= 1:
                        role = parts[i-1]
                        name = parts[i-1]
                    else:
                        role = 'button'
                        name = 'button'
                    
                    icon = {
                        'role': role,
                        'name': name,
                        'bbox': (x, y, w, h),
                        'conf': float(parts[i+4]) if i+4 < len(parts) and parts[i+4].replace('.','').isdigit() else 0.9
                    }
                    icons.append(icon)
                    print(f"🔍 Parsed taskbar {role}: local=({x},{y}) size=({w},{h})", file=sys.stderr)
                    coords_found = True
                    break
                except (ValueError, IndexError):
                    continue
            
            if not coords_found:
                print(f"🔍 Could not parse taskbar line: '{line}'", file=sys.stderr)
                
        except Exception as e:
            print(f"🔍 Taskbar parse error on line '{line}': {e}", file=sys.stderr)
            continue
                
    return icons


def _pick_box(vbbox, ibbox):
    """Use clickable bbox if exists, else visual bbox"""
    return ibbox if ibbox else vbbox


def _click_point_from_box(x, y, w, h):
    """Calculate click point from bounding box"""
    # True center
    cx = x + w // 2
    cy = y + h // 2
    # Keep away from borders
    cy = max(y + 2, min(y + h - 3, cy))
    return cx, cy


def _click_chrome(buttons_for_role):
    """Get click coordinates for chrome button role"""
    if not buttons_for_role:
        return None
    
    # Choose top candidate (k=1) by confidence
    best = max(buttons_for_role, key=lambda b: b['conf'])
    
    # Use visual box for coordinate calculation
    x, y, w, h = best['vbbox']
    cx, cy = _click_point_from_box(x, y, w, h)
    
    # Return coordinates without adjustment - global coordinates are already correct
    return cx, cy


def _calculate_adaptive_detail_cap(container_bounds: list, base_cap: int = 80) -> int:
    """Calculate adaptive detail cap based on container area"""
    x1, y1, x2, y2 = container_bounds
    area = (x2 - x1) * (y2 - y1)
    
    if area < 10000:      # Very small containers
        return min(base_cap, 20)
    elif area < 50000:    # Small containers
        return min(base_cap, 40)
    elif area < 100000:   # Medium containers
        return min(base_cap, 60)
    else:                 # Large containers
        return base_cap


def _pass_b_container_detail_cropped(image_bytes: bytes, container_id: str, container_bounds: list,
                                   width: int, height: int, delay: float = 0) -> dict:
    """Pass B: Per-container detail analysis with hard cropped image"""
    import time
    import random
    
    if delay > 0:
        jitter = random.uniform(0.8, 1.2) * delay
        time.sleep(jitter)
    
    x1, y1, x2, y2 = container_bounds
    detail_cap = _calculate_adaptive_detail_cap(container_bounds)
    
    cropped_bytes = _crop_container_image(image_bytes, container_bounds, padding=20)
    
    user_prompt = f"""List clickable items in this cropped image section. Use original desktop coordinates (not crop coords).

Format: TYPE:TEXT@X,Y

PRIORITY: Window title bar controls (close_button:X, minimize_button:-, maximize_button:□)
Also find: buttons, links, icons, inputs

Max {min(detail_cap, 15)} items."""

    try:
        response = call_chat_vision(
            model="qwen/qwen2.5-vl-32b-instruct",
            system_prompt="",
            user_text=user_prompt,
            image_bytes=cropped_bytes,
            temperature=0.1,
            timeout=60,
            extra={"max_tokens": 8192}
        )
        
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            return {"success": True, "content": content, "container_id": container_id}
        else:
            return {"success": False, "error": "No content in response", "container_id": container_id}
            
    except Exception as e:
        return {"success": False, "error": str(e), "container_id": container_id}


def _parse_json_response(content: str) -> dict:
    """Parse JSON response handling code fences"""
    try:
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        parsed = json.loads(content)
        
        if "elements" in parsed:
            return {"success": True, "elements": parsed["elements"]}
        else:
            return {"success": False, "error": "No elements array in JSON"}
            
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON parse error: {e}", "raw_content": content[:500]}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _scale_container_bounds(container_bounds: list, scale_x: float, scale_y: float) -> list:
    """Scale container bounds from downscaled to full resolution"""
    x1, y1, x2, y2 = container_bounds
    return [
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y)
    ]


def _merge_results_optimized(pass_a_elements: list, pass_b_results: list, scale_x: float, scale_y: float) -> list:
    """Merge Pass A and Pass B results with global ID renumbering and coordinate scaling"""
    all_elements = []
    id_counter = 1
    
    # Add Pass A containers with new IDs and scaled coordinates
    id_mapping = {}  # old_id -> new_id
    
    for element in pass_a_elements:
        old_id = element.get('id', f'container_{id_counter}')
        new_id = f'e{id_counter}'
        id_mapping[old_id] = new_id
        
        # Scale coordinates back to full resolution (already done above but ensure consistency)
        if 'bounds' in element:
            bounds = element['bounds']
            if len(bounds) == 4 and all(isinstance(b, (int, float)) for b in bounds):
                element['bounds'] = [
                    int(bounds[0] * scale_x) if bounds[0] < 1000 else bounds[0],  # Only scale if looks downscaled
                    int(bounds[1] * scale_y) if bounds[1] < 1000 else bounds[1],
                    int(bounds[2] * scale_x) if bounds[2] < 1000 else bounds[2],
                    int(bounds[3] * scale_y) if bounds[3] < 1000 else bounds[3]
                ]
        
        if 'center' in element:
            center = element['center']
            if len(center) == 2 and all(isinstance(c, (int, float)) for c in center):
                element['center'] = [
                    int(center[0] * scale_x) if center[0] < 1000 else center[0],
                    int(center[1] * scale_y) if center[1] < 1000 else center[1]
                ]
        
        element['id'] = new_id
        all_elements.append(element)
        id_counter += 1
    
    # Add Pass B elements with updated parent references (no scaling needed - already full res)
    for pass_b_result in pass_b_results:
        for element in pass_b_result:
            element['id'] = f'e{id_counter}'
            
            # Update parent reference
            old_parent = element.get('parent')
            if old_parent and old_parent in id_mapping:
                element['parent'] = id_mapping[old_parent]
            
            all_elements.append(element)
            id_counter += 1
    
    return all_elements


def _convert_two_pass_results_to_gemini_format(analysis_result: dict) -> str:
    """Convert optimized two-pass analysis results to Gemini-friendly format"""
    if not analysis_result.get("success", False):
        return f"Two-pass analysis failed: {analysis_result.get('error', 'Unknown error')}"
    
    elements = analysis_result.get("elements", [])
    containers_found = analysis_result.get("containers_found", 0)
    valid_containers = analysis_result.get("valid_containers", 0)
    
    # Create simple, clean descriptions for Gemini
    clickable_elements = []
    all_elements_info = []
    screen_description = "Desktop with GUI elements and controls"
    
    for element in elements:
        try:
            # Extract element information from JSON structure
            elem_id = element.get('id', 'unknown')
            elem_type = element.get('type', 'unknown')
            elem_text = element.get('text', '')
            center = element.get('center', [0, 0])
            bounds = element.get('bounds', [0, 0, 0, 0])
            confidence = element.get('confidence', 0.0)
            
            # Track all elements for debugging
            display_text = elem_text if elem_text else elem_type
            all_elements_info.append(f"{display_text} ({elem_type}, conf:{confidence:.2f})")
            
            # Determine if element is clickable based on type
            clickable_types = {
                'button', 'checkbox', 'radio', 'toggle', 'input', 'dropdown', 
                'list_item', 'link', 'icon', 'image', 'scrollbar', 'progress',
                'chip', 'badge', 'window', 'dialog', 'menu', 'toolbar'
            }
            
            if elem_type in clickable_types and len(center) >= 2:
                # Create a simple description with click coordinates
                x, y = int(center[0]), int(center[1])
                description = f"{display_text} (click {x},{y})"
                clickable_elements.append(description)
                        
        except Exception as e:
            all_elements_info.append(f"Parse error: {element} - {e}")
            continue
    
    # Get screen dimensions
    screen_size = "1024x768"  # Default
    if analysis_result.get("screen_width") and analysis_result.get("screen_height"):
        screen_size = f"{analysis_result['screen_width']}x{analysis_result['screen_height']}"
    
    # Create final description
    result = f"{screen_size}|{screen_description}\n\n"
    result += f"TWO-PASS ANALYSIS: Found {len(elements)} total elements\n"
    result += f"Containers: {containers_found} found, {valid_containers} analyzed\n"
    result += f"Approach: {analysis_result.get('approach', 'optimized_two_pass')}\n\n"
    
    result += f"ALL ELEMENTS ({len(all_elements_info)}):\n"
    for i, element_info in enumerate(all_elements_info[:15], 1):  # Show first 15
        result += f"  {i}. {element_info}\n"
    if len(all_elements_info) > 15:
        result += f"  ... and {len(all_elements_info) - 15} more elements\n"
    result += "\n"
    
    if clickable_elements:
        result += "CLICKABLE ELEMENTS:\n"
        for i, element in enumerate(clickable_elements[:25], 1):  # Up to 25 clickable elements
            result += f"{i}. {element}\n"
    else:
        result += "No clickable elements detected.\n"
    
    return result.strip()


def _call_simple_vision_analysis(image_bytes: bytes) -> str:
    """Fallback to simple vision analysis if adaptive partitioning fails"""
    try:
        # Compress PNG to JPEG like the original code
        jpeg_bytes = _compress_png_to_jpeg(image_bytes, quality=85)
        
        # Simplified prompt focused on clickable elements
        system_prompt = "You are a UI analysis expert. Focus on finding clickable elements and provide clear click targets."
        
        user_text = """Analyze this screenshot and identify clickable elements.

FORMAT:
First line: WIDTHxHEIGHT|Brief screen description
Then list: CLICKABLE ELEMENTS:
1. Element description (click x,y)
2. Another element (click x,y)

Focus ONLY on:
- Buttons that can be clicked
- Icons that can be clicked  
- Menu items
- File/folder icons
- Window controls

Provide the exact pixel coordinates where Gemini should click. Be precise.
Return ONLY the format above, no other text."""
        
        response = call_chat_vision(
            model="qwen/qwen2.5-vl-32b-instruct",
            system_prompt=system_prompt,
            user_text=user_text,
            image_bytes=jpeg_bytes,  # Use compressed JPEG like original
            temperature=0.1,
            timeout=30,  # Match original (int, not float)
            extra={
                "max_tokens": 16384,  # High limit like original
            }
        )
        
        # Extract content from response
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        if content:
            return content
        else:
            return f"OpenRouter response empty. Full response: {response}"
        
    except Exception as e:
        import traceback
        return f"OpenRouter vision analysis failed: {str(e)}\nTraceback: {traceback.format_exc()}"


def _analyze_windows_and_taskbars(image_bytes: bytes, width: int, height: int) -> str:
    """Analyze windows and taskbars with proper ROI processing for both"""
    try:
        # Pass A: Detect windows and taskbars  
        pass_a_result = _pass_a_windows_tsv(image_bytes, width, height)
        if not pass_a_result['success']:
            return f"Window/taskbar analysis failed: {pass_a_result.get('error', 'Unknown error')}"
            
        # Parse windows and taskbars using TSV format
        windows, taskbars = _parse_windows_tsv(pass_a_result['content'])
        
        print(f"🔍 Found {len(windows)} windows and {len(taskbars)} taskbars", file=sys.stderr)
        
        results = []
        
        # Process each window with ROI analysis
        for window in windows:
            try:
                # Use existing window ROI analysis
                ty1, ty2 = window['ty1'], window['ty2']
                xL, yT = window['x1'], ty1  
                xR, yB = window['x2'], ty2
                
                print(f"🔍 Window {window['id']} ROI: ({xL},{yT},{xR},{yB}) size={xR-xL}x{yB-yT}", file=sys.stderr)
                
                # Pass B: Analyze window chrome buttons in ROI
                pass_b_result = _pass_b_chrome_roi(image_bytes, xL, yT, xR, yB, ty1, ty2)
                
                if pass_b_result['success']:
                    print(f"🔍 Window Pass B response: {pass_b_result['content'][:200]}", file=sys.stderr)
                    
                    # Try normalized parsing first
                    roi_width = xR - xL
                    roi_height = yB - yT  
                    size_ok, buttons_dict = parse_normalized_tsv(pass_b_result['content'], roi_width, roi_height)
                    
                    if not size_ok:
                        print(f"🔍 SIZE gate failed, falling back to legacy parsing", file=sys.stderr)
                        buttons_dict = _parse_chrome_tsv(pass_b_result['content'])
                    
                    # Convert button dictionary to coordinates
                    for role, button_list in buttons_dict.items():
                        for button in button_list:
                            if 'vbbox' in button:
                                x, y, w, h = button['vbbox']
                                global_x = x + xL
                                global_y = y + yT
                                
                                # Calculate click point
                                click_x, click_y = _click_point_from_box(global_x, global_y, w, h)
                                
                                role_name = {'C': 'close', 'M': 'minimize', 'N': 'maximize'}[role]
                                print(f"🔍 {role_name} button - click at ({click_x}, {click_y})", file=sys.stderr)
                                results.append(f"{role_name} button (click {click_x}, {click_y}) - window {window['id']}")
                        
            except Exception as e:
                print(f"❌ Window {window['id']} analysis failed: {e}", file=sys.stderr)
                continue
        
        # Process each taskbar with ROI analysis  
        for taskbar in taskbars:
            try:
                xL = taskbar['x1']
                xR = taskbar['x2'] 
                yT = taskbar['y1']
                yB = taskbar['y2']
                
                print(f"🔍 Taskbar ROI calculated: ({xL},{yT},{xR},{yB}) size={xR-xL}x{yB-yT}", file=sys.stderr)
                
                # Pass B: Analyze taskbar icons in ROI
                pass_b_result = _pass_b_taskbar_roi(image_bytes, xL, yT, xR, yB)
                
                if pass_b_result['success']:
                    print(f"🔍 Taskbar Pass B response: {pass_b_result['content'][:200]}", file=sys.stderr)
                    icons = _parse_taskbar_tsv(pass_b_result['content'])
                    
                    # Convert LOCAL ROI coordinates to GLOBAL coordinates
                    for icon in icons:
                        x, y, w, h = icon['bbox']
                        print(f"🔍 Converting taskbar local to global: ({x},{y}) + ({xL},{yT}) -> ({x+xL},{y+yT})", file=sys.stderr)
                        global_x = x + xL
                        global_y = y + yT
                        
                        # Calculate click point
                        click_x = global_x + w // 2
                        click_y = global_y + h // 2
                        
                        print(f"🔍 {icon['role']}:{icon['name']} - click at ({click_x}, {click_y})", file=sys.stderr)
                        results.append(f"{icon['role']}:{icon['name']} - click ({click_x}, {click_y}) - taskbar {taskbar['id']}")
                        
            except Exception as e:
                print(f"❌ Taskbar {taskbar['id']} analysis failed: {e}", file=sys.stderr)
                continue
        
        # Format results for user  
        if results:
            return f"{width}x{height}|Desktop with clickable elements\n\nCLICKABLE ELEMENTS:\n" + "\n".join(f"{i+1}. {result}" for i, result in enumerate(results))
        else:
            return f"{width}x{height}|Desktop detected but no clickable buttons or icons found"
            
    except Exception as e:
        return f"Analysis error: {str(e)}"


def analyze_screenshot(host: str, port: int, vm_target: str = "local") -> Dict[str, Any]:
    """Capture and analyze a QMP screenshot using adaptive partitioning"""
    try:
        # Get QMP connection info from config
        script_dir = Path(__file__).parent
        config_file = script_dir / "vm_config.json"
        
        qmp_host = None
        qmp_port = None
        
        try:
            with open(config_file) as f:
                config = json.load(f)
                vm_info = config['vm_targets'].get(vm_target, config['vm_targets']['local'])
                if 'qmp_port' in vm_info:
                    qmp_host = host
                    qmp_port = vm_info['qmp_port']
                else:
                    raise Exception("No QMP port configured")
        except Exception as e:
            raise Exception(f"Failed to get QMP config: {e}")
        
        # Take screenshot using QMP like the working test script
        from qmp_client import QMPClient
        
        with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as tmp:
            screenshot_path = tmp.name
        
        try:
            client = QMPClient(qmp_host, qmp_port)
            client.connect()
            success = client.screendump(screenshot_path)
            client.disconnect()
            
            if not success:
                raise Exception("QMP screendump failed")
        except Exception as e:
            raise Exception(f"QMP screenshot failed: {e}")
        
        # Convert to JPEG for vision analysis like the working test script
        def compress_image_to_jpeg(image_path: str, quality: int = 100) -> bytes:
            """Compress image to JPEG for vision analysis"""
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[-1])
                else:
                    background.paste(img)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as JPEG
            import io
            jpeg_buffer = io.BytesIO()
            img.save(jpeg_buffer, format='JPEG', quality=quality, optimize=True)
            return jpeg_buffer.getvalue()
        
        # Convert PPM to JPEG like the working test script
        image_bytes = compress_image_to_jpeg(screenshot_path)
        
        # DEBUG: Save the exact image we send to vision model
        debug_vision_path = "/home/jacob/partition/qemu-tools/vision_model_input.jpg"
        with open(debug_vision_path, 'wb') as f:
            f.write(image_bytes)
        print(f"🔍 Saved vision model input to: {debug_vision_path}", file=sys.stderr)
        
        # Get image dimensions
        img = Image.open(screenshot_path)
        width, height = img.size
        print(f"🔍 Original PPM screenshot: {width}x{height}", file=sys.stderr)
        
        # Analyze using window/taskbar TSV system
        vision_description = _analyze_windows_and_taskbars(image_bytes, width, height)
        
        return {
            "success": True,
            "screen_info": {
                "width": width,
                "height": height,
                "description": "QMP Desktop Screenshot"
            },
            "vision_analysis": vision_description,
            "vm_target": vm_target,
            "method": "QMP"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        # Clean up temp file
        try:
            Path(screenshot_path).unlink(missing_ok=True)
        except:
            pass


def send_mouse_clicks(host: str, port: int, clicks: List[Dict], vm_target: str = "local") -> Dict[str, Any]:
    """Send mouse click events via QMP or VNC"""
    try:
        # Get QMP connection info
        qmp_host = None
        qmp_port = None
        
        script_dir = Path(__file__).parent
        config_file = script_dir / "vm_config.json"
        
        try:
            with open(config_file) as f:
                config = json.load(f)
                vm_info = config['vm_targets'].get(vm_target, config['vm_targets']['local'])
                if 'qmp_port' in vm_info:
                    qmp_host = host
                    qmp_port = vm_info['qmp_port']
        except Exception:
            pass
        
        success_count = 0
        errors = []
        
        # Force VNC for testing
        if False and qmp_host and qmp_port:
            try:
                from qmp_client import QMPClient
                client = QMPClient(qmp_host, qmp_port)
                client.connect()
                
                for click in clicks:
                    try:
                        x = click['x']
                        y = click['y']
                        button = click.get('button', 1)
                        action = click.get('action', 'click')
                        
                        if action == 'click':
                            client.send_mouse_event(x, y, button, True)   # Press
                            time.sleep(0.05)
                            client.send_mouse_event(x, y, button, False)  # Release
                        elif action == 'double_click':
                            for _ in range(2):
                                client.send_mouse_event(x, y, button, True)
                                time.sleep(0.05)
                                client.send_mouse_event(x, y, button, False)
                                time.sleep(0.1)
                        
                        success_count += 1
                    except Exception as e:
                        errors.append(f"QMP click at ({x},{y}): {str(e)}")
                
                client.disconnect()
                
                return {
                    "success": True,
                    "clicks_executed": success_count,
                    "total_clicks": len(clicks),
                    "errors": errors,
                    "method": "QMP"
                }
            except Exception:
                pass  # Fall back to VNC
        
        # Fallback to vncdotool
        display = int(port) - 5900
        vnc_address = f"{host}:{display}"
        
        client = api.connect(vnc_address, timeout=5)
        
        success_count = 0
        errors = []
        
        try:
            for click in clicks:
                try:
                    x = click['x']
                    y = click['y']
                    # Convert button name to number for VNC
                    button_raw = click.get('button', 1)
                    if button_raw == 'left':
                        button = 1
                    elif button_raw == 'middle':
                        button = 2
                    elif button_raw == 'right':
                        button = 3
                    elif button_raw == 'move':
                        button = None  # No button for move
                    else:
                        button = int(button_raw) if button_raw != 'move' else 1
                    
                    action = click.get('action', 'click')
                    
                    # Move to position
                    client.mouseMove(x, y)
                    time.sleep(0.05)
                    
                    if action == 'click':
                        # Single click
                        client.mousePress(button)
                        time.sleep(0.05)
                        client.mouseUp(button)
                    elif action == 'double_click':
                        # Double click
                        client.mousePress(button)
                        time.sleep(0.05)
                        client.mouseUp(button)
                        time.sleep(0.1)
                        client.mousePress(button)
                        time.sleep(0.05)
                        client.mouseUp(button)
                        
                    success_count += 1
                    
                except Exception as e:
                    errors.append(f"Click at ({x},{y}): {str(e)}")
        finally:
            try:
                client.disconnect()
            except:
                pass
            # Force exit to prevent vncdotool hanging
            import os
            os._exit(0)
        
        return {
            "success": True,
            "clicks_executed": success_count,
            "total_clicks": len(clicks),
            "errors": errors,
            "method": "VNC"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def _type_simple_text(client, text, shift_mappings):
    """Type simple text without newlines using proper shift chords"""
    def press_shifted(client, base_key):
        """Send explicit Shift chord for symbols"""
        client.keyDown('shift')
        time.sleep(0.02)
        client.keyPress(base_key)
        time.sleep(0.02)
        client.keyUp('shift')
        time.sleep(0.01)
    
    for char in text:
        if char == '\t':
            client.keyPress('Tab')
        elif char in shift_mappings:
            # Use explicit shift chord for symbols
            base_key = shift_mappings[char]
            press_shifted(client, base_key)
        else:
            # Regular character - letters, digits, unshifted symbols
            client.keyPress(char)
        time.sleep(0.01)

def send_keyboard_input(host: str, port: int, actions: List[Dict], vm_target: str = "local") -> Dict[str, Any]:
    """Send keyboard input via QMP or VNC"""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Keyboard operation timed out")
    
    # Set 15 second timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(15)
    
    try:
        # Get QMP connection info
        qmp_host = None
        qmp_port = None
        
        script_dir = Path(__file__).parent
        config_file = script_dir / "vm_config.json"
        
        try:
            with open(config_file) as f:
                config = json.load(f)
                vm_info = config['vm_targets'].get(vm_target, config['vm_targets']['local'])
                if 'qmp_port' in vm_info:
                    qmp_host = host
                    qmp_port = vm_info['qmp_port']
        except Exception:
            pass
        
        success_count = 0
        errors = []
        
        # Force VNC for testing
        if False and qmp_host and qmp_port:
            try:
                from qmp_client import QMPClient
                client = QMPClient(qmp_host, qmp_port)
                client.connect()
                
                for action in actions:
                    try:
                        action_type = action['action']
                        
                        if action_type == 'type':
                            text = action.get('text', '')
                            # Type each character via QMP
                            for char in text:
                                # Convert char to QMP key code (simplified mapping)
                                key_code = char.lower() if char.isalpha() else char
                                client.send_key_event(key_code, True)   # Press
                                time.sleep(0.02)
                                client.send_key_event(key_code, False)  # Release
                                time.sleep(0.02)
                                
                        elif action_type == 'key':
                            key = action.get('key', '')
                            
                            # Handle key combinations for QMP
                            if '+' in key:
                                keys = key.split('+')
                                # Press all modifier keys
                                for k in keys[:-1]:
                                    client.send_key_event(k.lower(), True)
                                    time.sleep(0.02)
                                # Press main key
                                client.send_key_event(keys[-1].lower(), True)
                                time.sleep(0.05)
                                client.send_key_event(keys[-1].lower(), False)
                                # Release modifier keys
                                for k in reversed(keys[:-1]):
                                    client.send_key_event(k.lower(), False)
                                    time.sleep(0.02)
                            else:
                                # Single key
                                client.send_key_event(key.lower(), True)
                                time.sleep(0.05)
                                client.send_key_event(key.lower(), False)
                                
                        success_count += 1
                        time.sleep(0.05)
                        
                    except Exception as e:
                        errors.append(f"QMP action {action}: {str(e)}")
                
                client.disconnect()
                
                return {
                    "success": True,
                    "actions_executed": success_count,
                    "total_actions": len(actions),
                    "errors": errors,
                    "method": "QMP"
                }
            except Exception:
                pass  # Fall back to VNC
        
        # Fallback to vncdotool
        display = int(port) - 5900
        vnc_address = f"{host}:{display}"
        
        client = api.connect(vnc_address, timeout=3)
        
        success_count = 0
        errors = []
        
        for action in actions:
            try:
                action_type = action['action']
                
                if action_type == 'type':
                    text = action.get('text', '')
                    print(f"HOST GOT: {repr(text)}", file=sys.stderr)
                    
                    def press_shifted(client, base_char):
                        """Send explicit Shift chord for symbols using base character"""
                        client.keyDown('shift')
                        time.sleep(0.02)
                        client.keyPress(base_char)
                        time.sleep(0.02)
                        client.keyUp('shift')
                        time.sleep(0.01)
                    
                    # US keyboard layout shift mappings - map to base character
                    SHIFT_MAPPINGS = {
                        '#': '3',
                        '!': '1',
                        '@': '2',
                        '$': '4',
                        '%': '5',
                        '^': '6',
                        '&': '7',
                        '*': '8',
                        '(': '9',
                        ')': '0',
                        '_': '-',
                        '+': '=',
                        '{': '[',
                        '}': ']',
                        '|': '\\',
                        ':': ';',
                        '"': "'",
                        '<': ',',
                        '>': '.',
                        '?': '/',
                        '~': '`',
                    }
                    
                    # Type text using explicit shift chords - no keysyms
                    for ch in text:
                        if ch == '\n':
                            client.keyPress('enter')
                        elif ch == '\t':
                            client.keyPress('Tab')
                        elif ch in SHIFT_MAPPINGS:
                            # Use explicit shift chord for symbols
                            base_key = SHIFT_MAPPINGS[ch]
                            press_shifted(client, base_key)
                        else:
                            # Regular character - letters, digits, unshifted symbols
                            client.keyPress(ch)
                        time.sleep(0.01)
                    
                elif action_type == 'key':
                    key = action.get('key', '')
                    
                    # Handle key combinations for vncdotool
                    if '+' in key:
                        # For vncdotool, use the exact format like "ctrl-a"
                        key_combo = key.replace('+', '-')
                        client.keyPress(key_combo)
                    else:
                        # Single key - handle common key name variations
                        if key.lower() == 'return':
                            client.keyPress('enter')
                        elif key.lower() == 'backspace':
                            client.keyPress('bsp')
                        elif key.lower() == 'delete':
                            client.keyPress('del')
                        else:
                            client.keyPress(key)
                        
                success_count += 1
                time.sleep(0.05)  # Small delay between actions
                
            except Exception as e:
                errors.append(f"VNC action {action}: {str(e)}")
                
        try:
            client.disconnect()
        except:
            pass  # Ignore disconnect errors
        
        return {
            "success": True,
            "actions_executed": success_count,
            "total_actions": len(actions),
            "errors": errors,
            "method": "VNC"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        signal.alarm(0)  # Clear the alarm


def main():
    """Main entry point for the VNC tools"""
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "Invalid arguments"}))
        sys.exit(1)
        
    command = sys.argv[1]
    
    # For stdin keyboard variants, we only need command and flag  
    if len(sys.argv) >= 3 and command == "keyboard" and (sys.argv[2] == "--stdin-text" or sys.argv[2] == "--stdin-json"):
        pass  # Handle in the stdin sections
    # For keyboard_file, we only need command and filename
    elif command == "keyboard_file":
        pass  # Handle in the keyboard_file section
    else:
        # For other commands, extract host/port/target
        if len(sys.argv) < 5:
            print(json.dumps({"success": False, "error": "Invalid arguments"}))
            sys.exit(1)
        host = sys.argv[2]
        port = int(sys.argv[3])
        vm_target = sys.argv[4]
    
    if command == "screenshot":
        result = analyze_screenshot(host, port, vm_target)
        result["vm_target"] = vm_target
        result["vnc_address"] = f"{host}:{port}"
        print(json.dumps(result))
        sys.stdout.flush()
        sys.exit(0)
        
    elif command == "mouse":
        # Read clicks from stdin
        try:
            data = json.loads(sys.stdin.read())
            clicks = data.get('clicks', [])
        except:
            clicks = []
            
        result = send_mouse_clicks(host, port, clicks, vm_target)
        result["vm_target"] = vm_target
        result["vnc_address"] = f"{host}:{port}"
        print(json.dumps(result))
        sys.stdout.flush()
        sys.exit(0)
        
    elif len(sys.argv) >= 3 and command == "keyboard" and sys.argv[2] == "--stdin-json":
        # Read raw JSON from stdin and extract text using string parsing
        try:
            raw_input = sys.stdin.read().strip()
            print(f"DEBUG: Raw stdin input: {repr(raw_input)}", file=sys.stderr)
            
            # Parse input for either text or actions using string parsing
            actions = []
            
            if '"text":' in raw_input:
                # Extract text field
                start = raw_input.find('"text":')
                if start != -1:
                    start += 7
                    while start < len(raw_input) and raw_input[start].isspace():
                        start += 1
                    if start < len(raw_input) and raw_input[start] == '"':
                        start += 1
                        end = start
                        while end < len(raw_input):
                            if raw_input[end] == '"':
                                if end > start and raw_input[end-1] == '\\':
                                    backslash_count = 0
                                    check = end - 1
                                    while check >= start and raw_input[check] == '\\':
                                        backslash_count += 1
                                        check -= 1
                                    if backslash_count % 2 == 1:
                                        end += 1
                                        continue
                                break
                            end += 1
                        
                        if end < len(raw_input):
                            text = raw_input[start:end]
                            text = text.replace('\\n', '\n')
                            text = text.replace('\\t', '\t')
                            text = text.replace('\\"', '"')
                            text = text.replace('\\!', '!')
                            text = text.replace('\\\\', '\\')
                            print(f"DEBUG: Extracted text: {repr(text)}", file=sys.stderr)
                            actions = [{'action': 'type', 'text': text}]
            
            elif '"actions":' in raw_input:
                # For actions, just parse using json module since it's more complex
                try:
                    data = json.loads(raw_input)
                    actions = data.get('actions', [])
                    print(f"DEBUG: Extracted actions: {actions}", file=sys.stderr)
                except Exception as e:
                    print(f'{{"success": false, "error": "Failed to parse actions: {str(e)}"}}')
                    sys.exit(1)
            
            if not actions:
                print('{"success": false, "error": "No text or actions found"}')
                sys.exit(1)
            host = '127.0.0.1'
            port = 5901
            vm_target = 'local'
            
            result = send_keyboard_input(host, port, actions, vm_target)
            result["vm_target"] = vm_target
            result["vnc_address"] = f"{host}:{port}"
            print(json.dumps(result))
            sys.stdout.flush()
            sys.exit(0)
            
        except Exception as e:
            print(f'{{"success": false, "error": "{str(e)}"}}')
            sys.exit(1)
    
    # --stdin-text handler removed to eliminate interference
    
    elif command == "keyboard_file":
        # Read JSON from file - no bash interpretation at all
        if len(sys.argv) < 3:
            print('{"success": false, "error": "No file provided"}')
            sys.exit(1)
        
        json_file = sys.argv[2]
        try:
            with open(json_file, 'r') as f:
                raw_input = f.read().strip()
                
            # Extract text using string parsing instead of JSON
            # Look for "text": "..." pattern
            if '"text":' in raw_input:
                # Find the start of the text value
                start_marker = '"text":'
                start_pos = raw_input.find(start_marker) + len(start_marker)
                
                # Skip whitespace and opening quote
                while start_pos < len(raw_input) and raw_input[start_pos] in ' \t':
                    start_pos += 1
                
                if start_pos < len(raw_input) and raw_input[start_pos] == '"':
                    start_pos += 1  # Skip opening quote
                    
                    # Find the matching closing quote (handle escaped quotes)
                    text_content = ""
                    pos = start_pos
                    while pos < len(raw_input):
                        char = raw_input[pos]
                        if char == '"':
                            # Check if it's escaped
                            if pos > 0 and raw_input[pos-1] == '\\':
                                # Count preceding backslashes
                                backslash_count = 0
                                check_pos = pos - 1
                                while check_pos >= 0 and raw_input[check_pos] == '\\':
                                    backslash_count += 1
                                    check_pos -= 1
                                
                                # If odd number of backslashes, the quote is escaped
                                if backslash_count % 2 == 1:
                                    text_content += char
                                else:
                                    # Even number means the quote is not escaped
                                    break
                            else:
                                # Unescaped quote - end of string
                                break
                        else:
                            text_content += char
                        pos += 1
                    
                    # Process escape sequences in the extracted text
                    text_content = text_content.replace('\\"', '"')
                    text_content = text_content.replace('\\\\', '\\')
                    text_content = text_content.replace('\\n', '\n')
                    text_content = text_content.replace('\\t', '\t')
                    
                    actions = [{'action': 'type', 'text': text_content}]
                else:
                    actions = []
            else:
                actions = []
            
            # Get connection details
            vm_target = 'local'  # Default since we're not parsing full JSON
            host = '127.0.0.1'
            port = 5901
            
            result = send_keyboard_input(host, port, actions, vm_target)
            result["vm_target"] = vm_target
            result["vnc_address"] = f"{host}:{port}"
            print(json.dumps(result))
            sys.stdout.flush()
            sys.exit(0)
            
        except Exception as e:
            print(f'{{"success": false, "error": "{str(e)}"}}')
            sys.exit(1)
    
    elif command == "keyboard":
        # Handle base64 encoded input if provided as argument, otherwise read from stdin
        try:
            if len(sys.argv) > 5:  # base64 encoded argument provided
                import base64
                encoded_data = sys.argv[5]
                decoded_json = base64.b64decode(encoded_data).decode('utf-8')
                data = json.loads(decoded_json)
            else:
                # Fallback to reading from stdin
                data = json.loads(sys.stdin.read())
            
            actions = data.get('actions', [])
            print(f"DEBUG: Initial actions: {actions}", file=sys.stderr)
            print(f"DEBUG: Data keys: {list(data.keys())}", file=sys.stderr)
            
            # Support simple text field for convenience
            if not actions and 'text' in data:
                # Properly decode escape sequences like \n, \t
                text = data['text'].encode().decode('unicode_escape')
                print(f"DEBUG: Converting text field to action: {repr(text)}", file=sys.stderr)
                actions = [{'action': 'type', 'text': text}]
            
            # Fix malformed actions (handle mixed formats and truncated actions)
            fixed_actions = []
            for action in actions:
                if isinstance(action, dict):
                    # Handle missing or truncated action type
                    action_type = action.get('action', '')
                    if not action_type and 'text' in action:
                        action_type = 'type'  # Default to type if text is present
                    elif not action_type and 'key' in action:
                        action_type = 'key'   # Default to key if key is present
                    elif action_type.startswith('typ'):
                        action_type = 'type'  # Fix truncated 'type'
                    elif action_type.startswith('key'):
                        action_type = 'key'   # Fix truncated 'key'
                    
                    # Create properly formatted action
                    fixed_action = {'action': action_type}
                    if 'text' in action:
                        fixed_action['text'] = action['text']
                    if 'key' in action:
                        fixed_action['key'] = action['key']
                    
                    fixed_actions.append(fixed_action)
            
            actions = fixed_actions
        except Exception as e:
            actions = []
            
        result = send_keyboard_input(host, port, actions, vm_target)
        result["vm_target"] = vm_target
        result["vnc_address"] = f"{host}:{port}"
        print(json.dumps(result))
        sys.stdout.flush()
        sys.exit(0)
        
    else:
        print(json.dumps({"success": False, "error": f"Unknown command: {command}"}))
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()