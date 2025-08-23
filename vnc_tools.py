#!/usr/bin/env python3
"""
VNC tools for Gemini CLI using vncdotool
Handles screenshot capture, mouse clicks, and keyboard input via VNC protocol
"""

import os
# Suppress transformers warnings and auto-trust remote code before importing any ML libraries
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TRUST_REMOTE_CODE'] = 'true'

# Monkey patch transformers to always trust remote code
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')

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

def _call_ollama_vision(image_bytes: bytes, width: int, height: int) -> dict:
    """Call Ollama vision model (llava) for GUI element detection with normalized coordinates"""
    try:
        import requests
        import base64
        
        print(f"🔍 Attempting Ollama llava vision detection for {width}x{height} image...", file=sys.stderr)
        
        # Convert image to base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Prepare the prompt with normalized coordinates requirement
        prompt = f"""You see ONE desktop screenshot. Output TSV format ONLY:

SIZE  {width}  {height}
PT  close  0.975  0.025  0.9
PT  minimize  0.925  0.025  0.9
PT  maximize  0.950  0.025  0.9
PT  menu  0.050  0.950  0.8
PT  firefox  0.100  0.100  0.7

Rules:
- First line MUST be: SIZE  {width}  {height}
- Each PT line: PT [label] [x_norm] [y_norm] [confidence]
- x_norm, y_norm are floats in [0,1] (0=left/top, 1=right/bottom)
- Common labels: close, minimize, maximize, menu, start, firefox, chrome, folder, terminal, settings, trash
- Output up to 20 PT lines for clickable elements you see
- No other text allowed"""

        # Call Ollama API with strict settings
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llava',
                'prompt': prompt,
                'images': [image_b64],
                'stream': False,
                'temperature': 0.0,  # Zero temperature for consistency
                'num_predict': 300,  # Limit response length
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result.get('response', '')
            print(f"🔍 Ollama response length: {len(content)} chars", file=sys.stderr)
            # Clean up markdown code blocks if present
            if '```' in content:
                content = content.replace('```', '').strip()
            print(f"🔍 Ollama raw response preview: {repr(content[:300])}", file=sys.stderr)
            return {"success": True, "content": content, "width": width, "height": height}
        else:
            print(f"❌ Ollama API error: {response.status_code}", file=sys.stderr)
            return {"success": False, "error": f"API error: {response.status_code}"}
            
    except Exception as e:
        print(f"❌ Ollama error: {e}", file=sys.stderr)
        return {"success": False, "error": str(e)}

def _norm_to_px(px: float, py: float, W: int, H: int) -> tuple:
    """Convert normalized coordinates to pixels with bounds clamping"""
    x = int(round(px * (W - 1)))
    y = int(round(py * (H - 1)))
    x = max(0, min(W - 1, x))
    y = max(0, min(H - 1, y))
    return x, y

def _parse_ollama_result(result: dict) -> str:
    """Parse Ollama TSV output with SIZE gate validation and normalized coordinates"""
    try:
        content = result.get("content", "")
        width = result.get("width", 800)
        height = result.get("height", 600)
        
        lines = content.strip().split('\n')
        size_ok = False
        points = []
        
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
                
            # Check SIZE gate
            if parts[0] == "SIZE":
                if len(parts) >= 3:
                    try:
                        w, h = int(parts[1]), int(parts[2])
                        size_ok = (w == width and h == height)
                        print(f"🔍 SIZE gate: expected {width}x{height}, got {w}x{h}, valid: {size_ok}", file=sys.stderr)
                    except:
                        pass
                continue
                
            # Parse PT lines
            if parts[0] == "PT" and len(parts) >= 5:
                try:
                    label = parts[1]
                    px = float(parts[2])
                    py = float(parts[3])
                    conf = float(parts[4]) if len(parts) > 4 else 0.9
                    
                    # Validate normalized coordinates are in [0,1]
                    if not (0.0 <= px <= 1.0 and 0.0 <= py <= 1.0):
                        print(f"🔍 Dropping OOB point: {label} at ({px},{py})", file=sys.stderr)
                        continue
                        
                    # Convert to pixel coordinates
                    x, y = _norm_to_px(px, py, width, height)
                    points.append({
                        "label": label,
                        "x": x,
                        "y": y,
                        "conf": conf
                    })
                    print(f"🔍 Parsed {label}: normalized ({px:.3f},{py:.3f}) -> pixels ({x},{y})", file=sys.stderr)
                    
                except (ValueError, IndexError) as e:
                    print(f"🔍 Parse error on PT line '{line}': {e}", file=sys.stderr)
                    continue
        
        # If SIZE gate failed, retry once or fall back
        if not size_ok:
            print("⚠️ SIZE gate failed, Ollama may be confused about dimensions", file=sys.stderr)
            # Could retry here with same prompt, but for now just warn
        
        # Format results
        if points:
            elements = []
            for pt in points:
                # Map common labels to more descriptive types
                label_map = {
                    'close': 'close_button',
                    'minimize': 'minimize_button', 
                    'maximize': 'maximize_button',
                    'min': 'minimize_button',
                    'max': 'maximize_button'
                }
                label = label_map.get(pt['label'], pt['label'])
                elements.append(f"{label}:{pt['label']} - click ({pt['x']}, {pt['y']})")
            return "\n".join(elements)
        else:
            return "No valid GUI elements detected"
            
    except Exception as e:
        print(f"❌ Failed to parse Ollama result: {e}", file=sys.stderr)
        return "Failed to parse Ollama output"

def _remove_duplicate_elements(parsed_content_list):
    """Remove duplicate elements from overlapping quadrant regions"""
    import sys
    
    # Group elements by approximate location and content similarity
    POSITION_TOLERANCE = 0.05  # 5% screen tolerance for same position
    TEXT_SIMILARITY_THRESHOLD = 0.7  # 70% similarity for text content
    
    unique_elements = []
    
    for element in parsed_content_list:
        bbox = element.get('bbox', [])
        content = element.get('content', '')
        
        if len(bbox) < 4:
            unique_elements.append(element)
            continue
            
        # Check if this element is a duplicate of an existing one
        is_duplicate = False
        
        for existing in unique_elements:
            existing_bbox = existing.get('bbox', [])
            existing_content = existing.get('content', '')
            
            if len(existing_bbox) < 4:
                continue
                
            # Check position similarity
            pos_diff = abs(bbox[0] - existing_bbox[0]) + abs(bbox[1] - existing_bbox[1])
            if pos_diff < POSITION_TOLERANCE:
                # Check content similarity
                if content == existing_content:
                    # Exact match - definitely duplicate
                    is_duplicate = True
                    break
                elif content and existing_content:
                    # Check if one is substring of other (partial overlap case)
                    if content in existing_content or existing_content in content:
                        # Keep the longer one
                        if len(content) > len(existing_content):
                            unique_elements.remove(existing)
                            break
                        else:
                            is_duplicate = True
                            break
        
        if not is_duplicate:
            unique_elements.append(element)
    
    print(f"🔍 Deduplication: {len(parsed_content_list)} -> {len(unique_elements)} elements", file=sys.stderr)
    return unique_elements

def _reconstruct_cross_quadrant_text(parsed_content_list, width, height):
    """Reconstruct text elements that may span across quadrant boundaries"""
    import sys
    
    # Separate text and non-text elements
    text_elements = [e for e in parsed_content_list if e.get('type') == 'text']
    non_text_elements = [e for e in parsed_content_list if e.get('type') != 'text']
    
    # Group text elements by line (similar Y coordinates)
    LINE_TOLERANCE = 0.08  # 8% of screen height tolerance for same line - increased for better line detection
    text_lines = []
    
    for element in text_elements:
        bbox = element.get('bbox', [])
        if len(bbox) >= 4:
            y_center = (bbox[1] + bbox[3]) / 2
            
            # Find existing line this element belongs to
            found_line = None
            for line in text_lines:
                line_y = line['y_center']
                if abs(y_center - line_y) < LINE_TOLERANCE:
                    found_line = line
                    break
            
            if found_line:
                found_line['elements'].append(element)
                # Update line Y to average
                found_line['y_center'] = sum(e['bbox'][1] + e['bbox'][3] for e in found_line['elements']) / (2 * len(found_line['elements']))
            else:
                text_lines.append({
                    'y_center': y_center,
                    'elements': [element]
                })
    
    # Reconstruct each line by sorting elements left-to-right and merging adjacent text
    reconstructed_text = []
    
    for line in text_lines:
        # Sort elements by X coordinate
        line['elements'].sort(key=lambda e: e['bbox'][0] if len(e['bbox']) >= 4 else 0)
        
        # Group adjacent elements (within same quadrant boundary or very close)
        ADJACENCY_TOLERANCE = 0.15  # 15% of screen width - increased to capture longer gaps
        merged_elements = []
        current_group = []
        
        for element in line['elements']:
            if not current_group:
                current_group = [element]
            else:
                # Check if this element is adjacent to the last one in current group
                last_element = current_group[-1]
                last_x_end = last_element['bbox'][2] if len(last_element['bbox']) >= 4 else 0
                current_x_start = element['bbox'][0] if len(element['bbox']) >= 4 else 0
                
                gap = current_x_start - last_x_end
                
                if gap < ADJACENCY_TOLERANCE:
                    # Adjacent - add to current group
                    current_group.append(element)
                else:
                    # Not adjacent - finalize current group and start new one
                    if len(current_group) > 1:
                        merged_element = _merge_text_elements(current_group)
                        merged_elements.append(merged_element)
                    else:
                        merged_elements.extend(current_group)
                    current_group = [element]
        
        # Don't forget the last group
        if current_group:
            if len(current_group) > 1:
                merged_element = _merge_text_elements(current_group)
                merged_elements.append(merged_element)
            else:
                merged_elements.extend(current_group)
        
        reconstructed_text.extend(merged_elements)
    
    print(f"🔍 Text reconstruction: {len(text_elements)} -> {len(reconstructed_text)} elements", file=sys.stderr)
    
    # Return combined list
    return reconstructed_text + non_text_elements

def _merge_text_elements(elements):
    """Merge multiple text elements into a single element"""
    if len(elements) <= 1:
        return elements[0] if elements else None
    
    # Sort by X coordinate
    elements.sort(key=lambda e: e['bbox'][0] if len(e['bbox']) >= 4 else 0)
    
    # Combine content with spaces
    combined_content = ' '.join(e.get('content', '') for e in elements if e.get('content'))
    
    # Calculate combined bounding box
    min_x = min(e['bbox'][0] for e in elements if len(e['bbox']) >= 4)
    min_y = min(e['bbox'][1] for e in elements if len(e['bbox']) >= 4)
    max_x = max(e['bbox'][2] for e in elements if len(e['bbox']) >= 4)
    max_y = max(e['bbox'][3] for e in elements if len(e['bbox']) >= 4)
    
    # Create merged element based on first element
    merged_element = elements[0].copy()
    merged_element['content'] = combined_content
    merged_element['bbox'] = [min_x, min_y, max_x, max_y]
    merged_element['quadrant'] = f"merged({','.join(e.get('quadrant', '') for e in elements)})"
    
    return merged_element

def _detect_ui_patterns(image_bytes: bytes, width: int, height: int) -> dict:
    """Stage 1: Detect structural UI patterns (windows, toolbars, tab bars, etc.)"""
    from PIL import Image
    import io
    import numpy as np
    
    # Convert to PIL Image for analysis
    image = Image.open(io.BytesIO(image_bytes))
    img_array = np.array(image)
    
    patterns = {
        "windows": [],
        "tab_bars": [],
        "toolbars": [],
        "title_bars": []
    }
    
    h, w = img_array.shape[:2]
    
    # Improved window detection using edge detection and horizontal line analysis
    def detect_window_boundaries():
        """Detect actual window boundaries using edge detection"""
        windows = []
        
        # Convert to grayscale for edge detection
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
            
        # Find horizontal edges (window title bars typically have strong horizontal edges)
        horizontal_edges = []
        
        # Scan for significant horizontal changes in brightness
        for y in range(10, h - 50, 5):  # Skip very top and bottom, step by 5 for efficiency
            # Check if this row has title-bar-like characteristics
            row = gray[y, :]
            
            # Look for patterns typical of title bars:
            # 1. Relatively uniform color across most of the width
            # 2. Distinct brightness difference from row above/below
            # 3. Contains some variation (buttons, text) but not too much
            
            if y > 10 and y < h - 10:
                row_above = gray[y-5, :]
                row_below = gray[y+5, :]
                
                # Calculate difference with surrounding rows
                diff_above = np.mean(np.abs(row - row_above))
                diff_below = np.mean(np.abs(row - row_below))
                
                # Check if this looks like a title bar boundary
                if diff_above > 15 or diff_below > 15:  # Significant brightness change
                    # Check if the row itself has title-bar characteristics
                    row_std = np.std(row)
                    if 10 < row_std < 50:  # Some variation but not too noisy
                        # Check if there's a clear left/right pattern (window borders)
                        left_region = row[:50] if len(row) > 50 else row[:len(row)//3]
                        right_region = row[-50:] if len(row) > 50 else row[-len(row)//3:]
                        
                        # Look for window-like bounds
                        if np.std(left_region) > 5 and np.std(right_region) > 5:
                            horizontal_edges.append(y)
        
        # Group nearby horizontal edges into windows
        if not horizontal_edges:
            # Fallback: assume full screen window
            windows.append({
                "type": "main_window",
                "bbox": [0, 0, w, h],
                "title_region": [0, 0, w, 50]
            })
        else:
            # Group edges that are close together
            edge_groups = []
            current_group = [horizontal_edges[0]] if horizontal_edges else []
            
            for edge in horizontal_edges[1:]:
                if edge - current_group[-1] < 30:  # Within 30 pixels
                    current_group.append(edge)
                else:
                    if current_group:
                        edge_groups.append(current_group)
                    current_group = [edge]
            
            if current_group:
                edge_groups.append(current_group)
            
            # Create window regions from edge groups
            for i, group in enumerate(edge_groups):
                title_y = min(group)
                
                # Determine window bounds
                window_top = max(0, title_y - 5)
                
                # Find window bottom (next group or screen bottom)
                if i + 1 < len(edge_groups):
                    window_bottom = min(edge_groups[i + 1]) - 5
                else:
                    window_bottom = h - 40  # Leave space for taskbar
                
                # Ensure reasonable window size
                if window_bottom - window_top > 50:
                    windows.append({
                        "type": f"window_{i+1}",
                        "bbox": [0, window_top, w, window_bottom],
                        "title_region": [0, title_y, w, title_y + 20]  # 20px title bar
                    })
        
        return windows
    
    # Use improved window detection
    patterns["windows"] = detect_window_boundaries()
    
    # Detect title bars for each window
    for window in patterns["windows"]:
        title_region = window.get("title_region", [])
        if len(title_region) >= 4:
            tx1, ty1, tx2, ty2 = title_region
            patterns["title_bars"].append({
                "type": "title_bar",
                "bbox": title_region,
                "region": "title_area",
                "parent_window": window
            })
    
    # Detect tab areas within windows
    for window in patterns["windows"]:
        bbox = window.get("bbox", [])
        if len(bbox) >= 4:
            wx1, wy1, wx2, wy2 = bbox
            # Look for tabs just below title bar
            tab_start_y = wy1 + 25  # Below title
            tab_end_y = min(wy1 + 60, wy2)  # 35px tab area
            
            if tab_end_y > tab_start_y:
                tab_region = img_array[tab_start_y:tab_end_y, wx1:wx2]
                if np.std(tab_region) > 15:  # Visual complexity suggests tabs
                    patterns["tab_bars"].append({
                        "type": "tab_bar",
                        "bbox": [wx1, tab_start_y, wx2, tab_end_y],
                        "region": "tabs",
                        "parent_window": window
                    })
    
    print(f"🔍 Stage 1: Detected {len(patterns['windows'])} windows, {len(patterns['title_bars'])} title bars, {len(patterns['tab_bars'])} tab areas", file=sys.stderr)
    return patterns

def _build_window_hypotheses(elements: list, width: int, height: int) -> list:
    """Build window hypotheses from element clustering"""
    import numpy as np
    from sklearn.cluster import DBSCAN
    
    if len(elements) < 3:
        return []
    
    # Convert elements to points for clustering
    points = []
    element_map = {}
    
    for i, element in enumerate(elements):
        bbox = element.get('bbox', [])
        if len(bbox) >= 4:
            # Convert normalized coordinates to pixels
            x = int((bbox[0] + bbox[2]) / 2 * width)  # center x
            y = int((bbox[1] + bbox[3]) / 2 * height)  # center y
            points.append([x, y])
            element_map[i] = element
    
    if len(points) < 3:
        return []
    
    P = np.array(points, dtype=float)
    
    # Screen-normalized anisotropic distances (horizontal proximity counts more)
    X = np.stack([P[:, 0] / width / 0.7, P[:, 1] / height / 1.0], axis=1)
    
    # Cluster elements spatially
    clustering = DBSCAN(eps=0.08, min_samples=3).fit(X)  # Tuned for desktop windows
    labels = clustering.labels_
    
    windows = []
    for lbl in set(labels):
        if lbl == -1:  # Noise points
            continue
            
        # Get cluster indices
        idx = np.where(labels == lbl)[0]
        if len(idx) < 3:
            continue
            
        # Calculate bounding box for this cluster
        xs = P[idx, 0]
        ys = P[idx, 1]
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        
        # Expand with padding
        pad = 20
        win = {
            'id': f'win_{len(windows)}',
            'x1': max(0, int(x1 - pad)), 
            'y1': max(0, int(y1 - pad)),
            'x2': min(width - 1, int(x2 + pad)), 
            'y2': min(height - 1, int(y2 + pad)),
            'members': [element_map[i]['id'] if 'id' in element_map[i] else f"elem_{i}" for i in idx if i in element_map],
            'score': 0.0, 
            'title': None,
            'elements': [element_map[i] for i in idx if i in element_map]
        }
        
        # Score based on title anchor (single text line in top band)
        top_band = win['y1'] + 0.12 * (win['y2'] - win['y1'])
        titles = []
        for i in idx:
            if i in element_map:
                elem = element_map[i]
                content = elem.get('content', '').strip()
                elem_type = elem.get('type', '')
                bbox = elem.get('bbox', [])
                if (elem_type == 'text' and len(content) >= 3 and len(bbox) >= 4):
                    elem_y = int((bbox[1] + bbox[3]) / 2 * height)
                    if elem_y < top_band:
                        titles.append(content)
        
        if len(titles) == 1:
            win['title'] = titles[0]
            win['score'] += 1.0
        elif len(titles) > 1:
            # Pick the longest title as most likely window title
            win['title'] = max(titles, key=len)
            win['score'] += 0.8
        
        # Score based on close-like buttons in top-right
        right_band_x = win['x1'] + 0.84 * (win['x2'] - win['x1'])
        top_band_y = win['y1'] + 0.14 * (win['y2'] - win['y1'])
        
        closers = 0
        for i in idx:
            if i in element_map:
                elem = element_map[i]
                bbox = elem.get('bbox', [])
                content = elem.get('content', '').lower()
                elem_type = elem.get('type', '')
                
                if len(bbox) >= 4:
                    elem_x = int((bbox[0] + bbox[2]) / 2 * width)
                    elem_y = int((bbox[1] + bbox[3]) / 2 * height)
                    elem_w = int((bbox[2] - bbox[0]) * width)
                    elem_h = int((bbox[3] - bbox[1]) * height)
                    
                    is_close_like = (
                        'x' in content or 'close' in content or 'dismiss' in content or
                        (elem_w * elem_h <= 26 * 26 and elem_type in ('button', 'icon'))
                    )
                    
                    if is_close_like and elem_x >= right_band_x and elem_y <= top_band_y:
                        closers += 1
        
        if closers > 0:
            win['score'] += 0.6
        
        windows.append(win)
    
    print(f"🔍 Window Hypotheses: Generated {len(windows)} window candidates", file=sys.stderr)
    for w in windows:
        print(f"🔍   Window {w['id']}: title='{w['title']}', score={w['score']:.1f}, elements={len(w['elements'])}", file=sys.stderr)
    
    return windows

def _assign_elements_to_windows(elements: list, windows: list, width: int, height: int) -> list:
    """Assign every element to its best parent window"""
    
    def inside(elem_bbox, win):
        if len(elem_bbox) < 4:
            return False
        elem_x = int((elem_bbox[0] + elem_bbox[2]) / 2 * width)
        elem_y = int((elem_bbox[1] + elem_bbox[3]) / 2 * height)
        return (win['x1'] <= elem_x <= win['x2']) and (win['y1'] <= elem_y <= win['y2'])
    
    def outside_dist(elem_bbox, win):
        if len(elem_bbox) < 4:
            return 1000
        elem_x = int((elem_bbox[0] + elem_bbox[2]) / 2 * width)
        elem_y = int((elem_bbox[1] + elem_bbox[3]) / 2 * height)
        dx = max(win['x1'] - elem_x, 0, elem_x - win['x2'])
        dy = max(win['y1'] - elem_y, 0, elem_y - win['y2'])
        return (dx * dx + dy * dy) ** 0.5
    
    results = []
    for i, elem in enumerate(elements):
        bbox = elem.get('bbox', [])
        elem_type = elem.get('type', '')
        content = elem.get('content', '').strip()
        
        if not windows or len(bbox) < 4:
            # No windows or invalid element
            results.append({**elem, 'parent_window_id': None, 'parent_window_title': None})
            continue
        
        best = None
        best_cost = 1e9
        
        for win in windows:
            cost = 0.0
            
            # Strong reward for containment
            if inside(bbox, win):
                cost -= 2.0
            else:
                # Penalty for being outside
                cost += 0.02 * outside_dist(bbox, win)
            
            # Reward for text in title band
            title_band = win['y1'] + 0.15 * (win['y2'] - win['y1'])
            if elem_type == 'text' and len(content) >= 3:
                elem_y = int((bbox[1] + bbox[3]) / 2 * height)
                if elem_y <= title_band:
                    cost -= 0.5
            
            # Reward for close-like elements in top-right
            right_band_x = win['x1'] + 0.86 * (win['x2'] - win['x1'])
            top_band_y = win['y1'] + 0.12 * (win['y2'] - win['y1'])
            
            content_lower = content.lower()
            is_close_like = (
                'close' in content_lower or content_lower.strip() in ('x', '×') or
                'dismiss' in content_lower or
                (len(bbox) >= 4 and 
                 (bbox[2] - bbox[0]) * width * (bbox[3] - bbox[1]) * height <= 26 * 26 and 
                 elem_type in ('button', 'icon'))
            )
            
            if is_close_like:
                elem_x = int((bbox[0] + bbox[2]) / 2 * width)
                elem_y = int((bbox[1] + bbox[3]) / 2 * height)
                if elem_x >= right_band_x and elem_y <= top_band_y:
                    cost -= 0.5
            
            # Prefer higher-scoring windows
            cost -= 0.2 * win['score']
            
            # Tie-break by smaller area (more specific window)
            win_area = (win['x2'] - win['x1']) * (win['y2'] - win['y1'])
            cost += win_area * 0.000001  # Very small penalty for large windows
            
            if cost < best_cost:
                best_cost = cost
                best = win
        
        # Assign to best window
        parent_id = best['id'] if best else None
        parent_title = best['title'] if best else None
        
        results.append({
            **elem, 
            'parent_window_id': parent_id, 
            'parent_window_title': parent_title
        })
    
    # Count assignments
    assigned = sum(1 for r in results if r['parent_window_id'] is not None)
    print(f"🔍 Element Assignment: {assigned}/{len(elements)} elements assigned to windows", file=sys.stderr)
    
    return results

def _build_element_relationships(elements: list, ui_patterns: dict, width: int, height: int) -> dict:
    """Stage 3: Build spatial relationships between UI elements and windows"""
    
    # Step 1: Build window hypotheses from element clustering
    windows = _build_window_hypotheses(elements, width, height)
    
    if not windows:
        print("🔍 No window hypotheses generated, skipping relationship building", file=sys.stderr)
        return {}
    
    # Step 2: Assign every element to its best parent window
    elements_with_parents = _assign_elements_to_windows(elements, windows, width, height)
    
    # Step 3: Build semantic relationships
    relationships = {}
    
    for elem in elements_with_parents:
        if elem['parent_window_id'] is None:
            continue
            
        bbox = elem.get('bbox', [])
        content = elem.get('content', '').strip()
        elem_type = elem.get('type', '')
        parent_title = elem['parent_window_title'] or 'unknown_window'
        
        if len(bbox) < 4:
            continue
        
        # Calculate click coordinates
        click_x = int((bbox[0] + bbox[2]) / 2 * width)
        click_y = int((bbox[1] + bbox[3]) / 2 * height)
        
        # Determine semantic role
        content_lower = content.lower()
        is_close_like = (
            'close' in content_lower or content_lower.strip() in ('x', '×') or
            'dismiss' in content_lower or
            (elem_type in ('button', 'icon') and 
             (bbox[2] - bbox[0]) * width * (bbox[3] - bbox[1]) * height <= 26 * 26)
        )
        
        # Find parent window details
        parent_window = next((w for w in windows if w['id'] == elem['parent_window_id']), None)
        
        if parent_window and is_close_like:
            # Check if it's in the top-right area (close button position)
            right_band_x = parent_window['x1'] + 0.86 * (parent_window['x2'] - parent_window['x1'])
            top_band_y = parent_window['y1'] + 0.12 * (parent_window['y2'] - parent_window['y1'])
            
            if click_x >= right_band_x and click_y <= top_band_y:
                # This is likely a close button
                relationships[f"close_button_{len(relationships)}"] = {
                    "action": "close_window",
                    "target_name": parent_title,
                    "element": elem,
                    "click_point": [click_x, click_y],
                    "semantic_description": f"close_button_in_{parent_title.replace(' ', '_')}_window"
                }
        
        # Add general containment relationship
        role = "close_button" if is_close_like else elem_type
        window_name = parent_title.replace(' ', '_').lower()
        relationships[f"{role}_in_{window_name}_{len(relationships)}"] = {
            "action": f"{role}_action",
            "target_name": parent_title,
            "element": elem,
            "click_point": [click_x, click_y],
            "semantic_description": f"{role}_in_{window_name}_window:{content}"
        }
    
    print(f"🔍 Stage 3: Built {len(relationships)} element relationships", file=sys.stderr)
    return relationships

def _call_omniparser_replicate(image_bytes: bytes) -> dict:
    """Call OmniParser v2.0 via Replicate API with hierarchical detection"""
    try:
        import replicate
        import base64
        import io
        import json
        from PIL import Image
        from pathlib import Path
        
        print("🔍 Using Hierarchical Detection with Replicate API for OmniParser v2.0...", file=sys.stderr)
        
        # Load configuration
        script_dir = Path(__file__).parent
        config_path = script_dir / "vm_config.json"
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            replicate_config = config.get('replicate', {})
        except Exception as e:
            print(f"⚠️ Failed to load config, using defaults: {e}", file=sys.stderr)
            replicate_config = {
                "model": "microsoft/omniparser-v2:49cf3d41b8d3aca1360514e83be4c97131ce8f0d99abfc365526d8384caa88df",
                "api_key_env": "REPLICATE_API_TOKEN",
                "default_params": {"imgsz": 640, "box_threshold": 0.05, "iou_threshold": 0.1}
            }
        
        # Get API key from environment variable
        import os
        api_key_env = replicate_config.get('api_key_env', 'REPLICATE_API_TOKEN')
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            return {"success": False, "error": f"Replicate API key not found in environment variable '{api_key_env}'"}
        
        os.environ["REPLICATE_API_TOKEN"] = api_key
        
        # Process image for 5-call system: full image + 4 quadrants
        import tempfile
        import concurrent.futures
        import threading
        
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"🔍 Using 5-call system: full image + 4 quadrants for {width}x{height} image", file=sys.stderr)
        
        # Get model from config
        model = replicate_config.get('model', 'microsoft/omniparser-v2:49cf3d41b8d3aca1360514e83be4c97131ce8f0d99abfc365526d8384caa88df')
        
        # Prepare 5 images: full + 4 quadrants with overlap
        OVERLAP = 50  # pixels overlap to catch UI elements spanning boundaries
        images_to_process = [
            ("full_image", image, (0, 0)),
            ("top_left", image.crop((0, 0, width//2 + OVERLAP, height//2 + OVERLAP)), (0, 0)),
            ("top_right", image.crop((width//2 - OVERLAP, 0, width, height//2 + OVERLAP)), (width//2 - OVERLAP, 0)),
            ("bottom_left", image.crop((0, height//2 - OVERLAP, width//2 + OVERLAP, height)), (0, height//2 - OVERLAP)),
            ("bottom_right", image.crop((width//2 - OVERLAP, height//2 - OVERLAP, width, height)), (width//2 - OVERLAP, height//2 - OVERLAP))
        ]
        
        def call_replicate_api(name, img, offset):
            """Call Replicate API for a single image"""
            try:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    img.save(temp_file, format='PNG')
                    temp_path = temp_file.name
                
                try:
                    print(f"🔍 Starting {name} API call...", file=sys.stderr)
                    with open(temp_path, 'rb') as f:
                        result = replicate.run(
                            model,
                            input={
                                "image": f,
                                "imgsz": 640,
                                "box_threshold": 0.05,
                                "iou_threshold": 0.1
                            }
                        )
                    print(f"🔍 Completed {name} API call", file=sys.stderr)
                    return (name, result, offset, img.size)
                finally:
                    import os
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            except Exception as e:
                print(f"❌ Error in {name} API call: {e}", file=sys.stderr)
                return (name, None, offset, img.size)
        
        # Execute 5 API calls with staggered start to avoid rate limiting
        import time
        start_time = time.time()
        print("🔍 Starting 5 parallel Replicate API calls with 0.5s delays...", file=sys.stderr)
        
        # Submit all futures with small delays between submissions
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i, (name, img, offset) in enumerate(images_to_process):
                # Add 0.5 second delay between each submission
                if i > 0:
                    time.sleep(0.5)
                future = executor.submit(call_replicate_api, name, img, offset)
                futures.append((future, name))
                print(f"🔍 Submitted call {i+1}/5: {name}", file=sys.stderr)
            
            submit_time = time.time()
            print(f"🔍 All 5 API calls submitted with delays in {submit_time - start_time:.2f}s", file=sys.stderr)
            
            # Collect results as they complete
            api_results = {}
            for future, name in futures:
                name_result, result, offset, img_size = future.result()
                api_results[name_result] = (result, offset, img_size)
        
        total_api_time = time.time() - start_time
        print(f"🔍 All 5 API calls completed in {total_api_time:.2f}s", file=sys.stderr)
        
        # STAGE 1: Detect UI patterns from the full image
        print("🔍 Stage 1: Analyzing UI patterns...", file=sys.stderr)
        ui_patterns = _detect_ui_patterns(image_bytes, width, height)
        
        # Process results from all 5 API calls
        print("🔍 Processing results from 5 API calls...", file=sys.stderr)
        
        # Separate full image results from quadrant results
        full_image_result = api_results.get('full_image', (None, None, None))[0]
        
        # Parse full image for text (like the old system)
        full_text_elements = []
        if full_image_result and isinstance(full_image_result, dict):
            elements_str = full_image_result.get('elements', '')
            if elements_str:
                import ast
                lines = elements_str.strip().split('\n')
                for line in lines:
                    if line.strip() and ':' in line:
                        try:
                            parts = line.split(':', 1)
                            if len(parts) >= 2:
                                dict_str = parts[1].strip()
                                element_dict = ast.literal_eval(dict_str)
                                
                                # Only keep text elements from full image
                                if element_dict.get('type') == 'text':
                                    full_text_elements.append({
                                        'type': 'text',
                                        'content': element_dict.get('content', ''),
                                        'bbox': element_dict.get('bbox', [0, 0, 1, 1]),
                                        'interactivity': False,
                                        'source': 'full_image_text',
                                        'quadrant': 'full_image'
                                    })
                        except:
                            continue
        
        print(f"🔍 Full image: {len(full_text_elements)} text elements", file=sys.stderr)
        
        # Parse quadrant results for UI elements
        all_ui_elements = []
        for quad_name in ['top_left', 'top_right', 'bottom_left', 'bottom_right']:
            quad_data = api_results.get(quad_name)
            if not quad_data or not quad_data[0]:
                continue
                
            quad_result, (offset_x, offset_y), (quad_width, quad_height) = quad_data
            
            if isinstance(quad_result, dict):
                elements_str = quad_result.get('elements', '')
                if elements_str:
                    import ast
                    lines = elements_str.strip().split('\n')
                    for line in lines:
                        if line.strip() and ':' in line:
                            try:
                                parts = line.split(':', 1)
                                if len(parts) >= 2:
                                    dict_str = parts[1].strip()
                                    element_dict = ast.literal_eval(dict_str)
                                    
                                    # Only keep icon elements from quadrants
                                    if element_dict.get('type') == 'icon':
                                        # Convert quadrant coordinates to full image coordinates
                                        bbox = element_dict.get('bbox', [0, 0, 1, 1])
                                        x1, y1, x2, y2 = bbox
                                        
                                        # Convert from quadrant ratios to full image ratios
                                        full_x1 = (x1 * quad_width + offset_x) / width
                                        full_y1 = (y1 * quad_height + offset_y) / height
                                        full_x2 = (x2 * quad_width + offset_x) / width  
                                        full_y2 = (y2 * quad_height + offset_y) / height
                                        
                                        all_ui_elements.append({
                                            'type': 'icon',
                                            'content': element_dict.get('content', ''),
                                            'bbox': [full_x1, full_y1, full_x2, full_y2],
                                            'interactivity': True,
                                            'source': 'quadrant_ui',
                                            'quadrant': quad_name
                                        })
                            except:
                                continue
        
        print(f"🔍 Quadrants: {len(all_ui_elements)} UI elements", file=sys.stderr)
        
        # Combine results like the old dual approach
        parsed_content_list = full_text_elements + all_ui_elements
        
        print(f"🔍 Total: {len(parsed_content_list)} elements", file=sys.stderr)
        
        return {
            "success": True,
            "result": {
                "parsed_content_list": parsed_content_list,
                "coordinates": {},
                "labeled_image": full_image_result.get('img', '') if full_image_result else '',
                "width": width,
                "height": height
            }
        }
            
    except Exception as e:
        print(f"❌ Replicate API error: {e}", file=sys.stderr)
        return {"success": False, "error": str(e)}

def _call_omniparser_vision(image_bytes: bytes) -> dict:
    """Call OmniParser v2.0 for GUI element detection using local installation"""
    try:
        import tempfile
        import os
        import sys
        import io
        from pathlib import Path
        from PIL import Image
        import torch
        
        print("🔍 Attempting OmniParser v2.0 local detection...", file=sys.stderr)
        
        # Get script directory and OmniParser path
        script_dir = Path(__file__).parent
        omniparser_dir = script_dir / "OmniParser"
        weights_dir = omniparser_dir / "weights"
        
        if not omniparser_dir.exists():
            print(f"❌ OmniParser directory not found at {omniparser_dir}", file=sys.stderr)
            return {"success": False, "error": "OmniParser directory not found"}
        
        if not weights_dir.exists():
            print(f"❌ OmniParser weights not found at {weights_dir}", file=sys.stderr)
            return {"success": False, "error": "OmniParser weights not found"}
        
        # Add OmniParser to Python path
        sys.path.insert(0, str(omniparser_dir))
        
        
        try:
            # Suppress verbose model loading output
            import os
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["ULTRALYTICS_SILENT"] = "1"
            
            # Also suppress warnings and verbose output at Python level
            import warnings
            warnings.filterwarnings("ignore")
            
            # Suppress specific library verbose output
            import logging
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("torch").setLevel(logging.ERROR)
            logging.getLogger("ultralytics").setLevel(logging.ERROR)
            
            # Try to install flash_attn if missing (last resort)
            try:
                import flash_attn
            except ImportError:
                print("🔧 flash_attn not found, attempting emergency install...", file=sys.stderr)
                import subprocess
                try:
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", "flash-attn", 
                        "--no-build-isolation", "--quiet"
                    ], check=True, timeout=120)
                    print("🔧 Emergency flash_attn install succeeded", file=sys.stderr)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    print("🔧 Emergency flash_attn install failed, model may not work", file=sys.stderr)
            
            from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
            
            print("🔍 Loading OmniParser models...", file=sys.stderr)
            
            # Load YOLO model normally (no Florence-2 warnings)
            yolo_model = get_yolo_model(model_path=str(weights_dir / 'icon_detect' / 'model.pt'))
            
            # Temporarily suppress stdout/stderr only during Florence-2 model loading
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                # Redirect to null during Florence-2 loading
                sys.stdout = open(os.devnull, 'w')
                sys.stderr = open(os.devnull, 'w')
                
                # Load caption model (Florence-2) - this is where the warning appears
                caption_model_processor = get_caption_model_processor(
                    model_name="florence2", 
                    model_name_or_path=str(weights_dir / 'icon_caption_florence')
                )
                
            finally:
                # Always restore stdout/stderr
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            
            print("🔍 Models loaded, processing image...", file=sys.stderr)
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            width, height = image.size
            
            print("🔍 Using 2-pass hierarchical detection: windows → window buttons...", file=sys.stderr)
            
            # PASS 1: Detect windows and general elements from full image
            print("🔍 Pass 1: Detecting windows and general elements from full image...", file=sys.stderr)
            
            # Calculate overlay ratio and drawing config for full image
            box_overlay_ratio = width / 3200
            draw_bbox_config = {
                'text_scale': 0.8 * box_overlay_ratio,
                'text_thickness': max(int(2 * box_overlay_ratio), 1),
                'text_padding': max(int(3 * box_overlay_ratio), 2),
                'thickness': max(int(3 * box_overlay_ratio), 2)
            }
            
            # Run OCR on full image 
            (full_text, full_ocr_bbox), _ = check_ocr_box(
                image, 
                display_img=False, 
                output_bb_format='xyxy', 
                easyocr_args={'text_threshold': 0.3}, 
                use_paddleocr=False
            )
            
            # Get SOM labeled image and parsed content for full image (Pass 1)
            try:
                dino_labeled_img, label_coordinates, pass1_elements = get_som_labeled_img(
                    image, 
                    yolo_model, 
                    BOX_TRESHOLD=0.05, 
                    output_coord_in_ratio=True, 
                    ocr_bbox=full_ocr_bbox or [], 
                    draw_bbox_config=draw_bbox_config, 
                    caption_model_processor=caption_model_processor, 
                    ocr_text=full_text or [], 
                    use_local_semantics=True, 
                    iou_threshold=0.7, 
                    imgsz=640
                )
                
                if label_coordinates is None:
                    label_coordinates = {}
                if pass1_elements is None:
                    pass1_elements = []
                
            except Exception as e:
                print(f"❌ Pass 1 OmniParser error: {e}", file=sys.stderr)
                return {"success": False, "error": f"Pass 1 OmniParser error: {e}"}
            
            print(f"🔍 Pass 1 result: {len(pass1_elements)} elements detected", file=sys.stderr)
            
            # PASS 2: Detect window controls by cropping individual windows
            print("🔍 Pass 2: Detecting window controls from individual window crops...", file=sys.stderr)
            
            # Build window hypotheses from Pass 1 elements
            windows = _build_window_hypotheses(pass1_elements, width, height)
            print(f"🔍 Pass 2: Found {len(windows)} window candidates for detailed analysis", file=sys.stderr)
            
            # PASS 2: Process each window individually to detect buttons
            print("🔍 Pass 2: Processing individual windows to detect buttons and controls...", file=sys.stderr)
            
            all_window_elements = []
            all_coordinates = {}
            element_count = 0
            
            for i, window in enumerate(windows):
                win_x1, win_y1, win_x2, win_y2 = window['x1'], window['y1'], window['x2'], window['y2']
                win_width = win_x2 - win_x1
                win_height = win_y2 - win_y1
                
                # Skip very small windows (likely noise)
                if win_width < 50 or win_height < 50:
                    continue
                    
                print(f"🔍 Processing window {i+1}/{len(windows)}: {window['title']} at ({win_x1}, {win_y1}) - {win_width}x{win_height}", file=sys.stderr)
                
                # Crop the window from the full image
                window_image = image.crop((win_x1, win_y1, win_x2, win_y2))
                
                # Calculate drawing config for this window
                box_overlay_ratio = win_width / 3200
                draw_bbox_config = {
                    'text_scale': 0.8 * box_overlay_ratio,
                    'text_thickness': max(int(2 * box_overlay_ratio), 1),
                    'text_padding': max(int(3 * box_overlay_ratio), 1),
                    'thickness': max(int(3 * box_overlay_ratio), 1),
                }
                
                # Run OCR on the window crop
                try:
                    (window_text, window_ocr_bbox), _ = check_ocr_box(
                        window_image, 
                        display_img=False, 
                        output_bb_format='xyxy', 
                        easyocr_args={'text_threshold': 0.7},
                        use_paddleocr=False
                    )
                except Exception as e:
                    print(f"⚠️ OCR failed for window {i+1}: {e}", file=sys.stderr)
                    window_text = []
                    window_ocr_bbox = []
                
                # Run OmniParser on the cropped window
                try:
                    win_dino_labeled_img, win_label_coordinates, win_parsed_content_list = get_som_labeled_img(
                        window_image, 
                        yolo_model, 
                        BOX_TRESHOLD=0.03,  # Lower threshold for window buttons
                        output_coord_in_ratio=True, 
                        ocr_bbox=window_ocr_bbox or [], 
                        draw_bbox_config=draw_bbox_config, 
                        caption_model_processor=caption_model_processor, 
                        ocr_text=window_text or [], 
                        use_local_semantics=True, 
                        iou_threshold=0.5,  # Lower threshold to catch more buttons
                        scale_img=False, 
                        batch_size=128
                    )
                    
                    # Handle None returns
                    if win_label_coordinates is None:
                        win_label_coordinates = {}
                    if win_parsed_content_list is None:
                        win_parsed_content_list = []
                        
                except Exception as e:
                    print(f"❌ OmniParser error processing window {i+1}: {e}", file=sys.stderr)
                    win_label_coordinates = {}
                    win_parsed_content_list = []
                
                # Convert window-relative coordinates back to full image coordinates
                for element in win_parsed_content_list:
                    if 'bbox' in element:
                        bbox = element['bbox']
                        # Convert from window ratios to full image coordinates
                        global_x1 = win_x1 + (bbox[0] * win_width)
                        global_y1 = win_y1 + (bbox[1] * win_height)  
                        global_x2 = win_x1 + (bbox[2] * win_width)
                        global_y2 = win_y1 + (bbox[3] * win_height)
                        
                        # Convert to full image ratios
                        element['bbox'] = [
                            global_x1 / width, 
                            global_y1 / height,
                            global_x2 / width, 
                            global_y2 / height
                        ]
                        element['window_id'] = i
                        element['window_title'] = window['title']
                    
                    all_window_elements.append(element)
                
                # Convert window label coordinates to global coordinates
                for key, coords in win_label_coordinates.items():
                    global_key = f"win{i}_{key}"
                    global_x = win_x1 + (coords[0] * win_width)
                    global_y = win_y1 + (coords[1] * win_height)
                    global_coords = [
                        global_x / width,
                        global_y / height,
                        coords[2] * win_width / width,  # width ratio
                        coords[3] * win_height / height  # height ratio
                    ]
                    all_coordinates[global_key] = global_coords
                
                element_count += len(win_parsed_content_list)
                print(f"🔍 Window {i+1}: found {len(win_parsed_content_list)} elements", file=sys.stderr)
            
            print(f"🔍 Pass 2 complete: {element_count} total elements from {len(windows)} windows", file=sys.stderr)
            
            # Combine Pass 1 elements (windows, text) with Pass 2 elements (window buttons)
            print("🔍 Combining Pass 1 and Pass 2 elements...", file=sys.stderr)
            all_elements = pass1_elements + all_window_elements
            
            # STAGE 1: Detect UI patterns from the full image
            print("🔍 Stage 1: Analyzing UI patterns...", file=sys.stderr)
            ui_patterns = _detect_ui_patterns(image_bytes, width, height)
            
            # STAGE 3: Build element relationships for context-aware clicking
            print("🔍 Stage 3: Building element relationships...", file=sys.stderr)
            relationships = _build_element_relationships(all_elements, ui_patterns, width, height)
            
            # No need for complex text reconstruction since we got complete text from full image
            reconstructed_content = all_elements
            
            # Use the full image as the labeled image (we could composite quadrants later if needed)
            dino_labeled_img = ""  # Placeholder
            label_coordinates = all_coordinates
            parsed_content_list = reconstructed_content
            
            print(f"🔍 OmniParser detected {len(parsed_content_list)} elements", file=sys.stderr)
            print(f"🔍 RAW OMNIPARSER OUTPUT:", file=sys.stderr)
            print(f"🔍 dino_labeled_img type: {type(dino_labeled_img)}", file=sys.stderr)
            print(f"🔍 label_coordinates type: {type(label_coordinates)}, content: {label_coordinates}", file=sys.stderr)
            print(f"🔍 parsed_content_list type: {type(parsed_content_list)}, content: {parsed_content_list}", file=sys.stderr)
            
            # Check for any close-like elements in raw data
            close_candidates = []
            for i, elem in enumerate(parsed_content_list):
                content = str(elem.get('content', '')).lower()
                if any(keyword in content for keyword in ['close', 'x', '×', 'exit', 'min', 'max']):
                    close_candidates.append(f"Element {i}: {elem}")
                    print(f"🔍 CLOSE/MIN/MAX CANDIDATE {i}: {elem}", file=sys.stderr)
            
            if not close_candidates:
                print(f"🔍 NO WINDOW CONTROL CANDIDATES found in {len(parsed_content_list)} raw elements", file=sys.stderr)
                # Show first few elements to see what is being detected
                print(f"🔍 FIRST 5 RAW ELEMENTS:", file=sys.stderr)
                for i, elem in enumerate(parsed_content_list[:5]):
                    print(f"🔍   {i}: {elem}", file=sys.stderr)
            
            return {
                "success": True, 
                "result": {
                    "labeled_image": dino_labeled_img,
                    "coordinates": label_coordinates,
                    "parsed_content_list": parsed_content_list,
                    "ui_patterns": ui_patterns,
                    "relationships": relationships
                }
            }
                
        except ImportError as e:
            print(f"❌ Failed to import OmniParser modules: {e}", file=sys.stderr)
            return {"success": False, "error": f"Failed to import OmniParser modules: {e}"}
        finally:
            # Remove from Python path
            if str(omniparser_dir) in sys.path:
                sys.path.remove(str(omniparser_dir))
                
    except Exception as e:
        print(f"❌ OmniParser error: {e}", file=sys.stderr)
        return {"success": False, "error": str(e)}

def _parse_omniparser_result(result, width=800, height=600) -> dict:
    """Parse OmniParser output into hierarchical JSON structure grouped by windows"""
    try:
        # New format from our updated OmniParser function
        if isinstance(result, dict) and "parsed_content_list" in result:
            parsed_content_list = result["parsed_content_list"]
            coordinates = result.get("coordinates", [])
            relationships = result.get("relationships", {})
            ui_patterns = result.get("ui_patterns", {})
            
            print(f"🔍 Parsing {len(parsed_content_list)} OmniParser elements with {len(relationships)} relationships", file=sys.stderr)
            
            # Create hierarchical structure: desktop -> windows -> elements
            desktop_structure = {
                "desktop": {
                    "taskbar": [],
                    "background_elements": []
                },
                "windows": {},
                "global_controls": [],
                "summary": {
                    "total_elements": len(parsed_content_list),
                    "total_windows": 0
                }
            }
            
            # Process each element and assign to appropriate window or desktop area
            for i, element in enumerate(parsed_content_list):
                element_type = element.get('type', 'unknown')
                content = element.get('content', f'{element_type}_{i}')
                window_id = element.get('window_id')
                window_title = element.get('window_title', 'Unknown Window')
                
                # Get coordinates from element bbox if available
                bbox = element.get('bbox')
                if bbox and len(bbox) >= 4:
                    # bbox is in ratio format [x1, y1, x2, y2], convert to pixels
                    x1, y1, x2, y2 = bbox[:4]
                    center_x_ratio = (x1 + x2) / 2
                    center_y_ratio = (y1 + y2) / 2
                    
                    # Convert to pixel coordinates
                    center_x_px = int(center_x_ratio * width)
                    center_y_px = int(center_y_ratio * height)
                    
                    # Create element object
                    element_obj = {
                        "type": element_type,
                        "content": content,
                        "coordinates": {
                            "x": center_x_px,
                            "y": center_y_px,
                            "bbox": [int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)]
                        },
                        "is_interactive": element.get('interactivity', True),
                        "source": element.get('source', 'omniparser')
                    }
                    
                    print(f"🔍 Element {i}: {element_type}:{content} - click ({center_x_px}, {center_y_px}) (window: {window_title})", file=sys.stderr)
                    
                    # Classify and assign to appropriate container
                    if window_id is not None:
                        # Element belongs to a specific window
                        window_key = f"window_{window_id}"
                        if window_key not in desktop_structure["windows"]:
                            desktop_structure["windows"][window_key] = {
                                "title": window_title,
                                "window_id": window_id,
                                "controls": [],
                                "content": [],
                                "buttons": []
                            }
                        
                        # Categorize element within the window
                        if any(keyword in content.lower() for keyword in ['close', 'minimize', 'maximize', 'x', '×']):
                            desktop_structure["windows"][window_key]["controls"].append(element_obj)
                        elif element_type == 'icon' and any(keyword in content.lower() for keyword in ['button']):
                            desktop_structure["windows"][window_key]["buttons"].append(element_obj)
                        else:
                            desktop_structure["windows"][window_key]["content"].append(element_obj)
                            
                    else:
                        # Element is not assigned to a specific window
                        if any(keyword in content.lower() for keyword in ['menu', 'places', 'start', 'taskbar', 'browser', 'firefox']):
                            desktop_structure["desktop"]["taskbar"].append(element_obj)
                        elif any(keyword in content.lower() for keyword in ['close', 'minimize', 'maximize']):
                            desktop_structure["global_controls"].append(element_obj)
                        else:
                            desktop_structure["desktop"]["background_elements"].append(element_obj)
                else:
                    # No coordinates available
                    print(f"🔍 Element {i}: {element_type}:{content} - (no coordinates)", file=sys.stderr)
            
            # Update summary
            desktop_structure["summary"]["total_windows"] = len(desktop_structure["windows"])
            
            return desktop_structure
        
        # Fallback for old format
        elif isinstance(result, (tuple, list)) and len(result) >= 2:
            # Second element is usually the text descriptions
            if len(result) > 1 and result[1]:
                text_output = str(result[1])
                lines = text_output.strip().split('\n')
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse element descriptions
                    # Expected formats might be:
                    # "button: Click here @ (100, 200, 50, 30)"
                    # "icon: Settings <select>100, 200, 150, 230</select>"
                    
                    import re
                    
                    # Try to extract coordinates
                    coord_match = re.search(r'[\(<](\d+),?\s*(\d+),?\s*(\d+),?\s*(\d+)[>\)]', line)
                    if coord_match:
                        x, y, x2, y2 = map(int, coord_match.groups())
                        # Calculate center point
                        center_x = (x + x2) // 2 if x2 > x else x
                        center_y = (y + y2) // 2 if y2 > y else y
                        
                        # Extract type and description
                        desc_part = line[:coord_match.start()].strip()
                        if ':' in desc_part:
                            elem_type, desc = desc_part.split(':', 1)
                            elem_type = elem_type.strip()
                            desc = desc.strip()
                        else:
                            elem_type = "element"
                            desc = desc_part
                        
                        elements.append(f"{elem_type}:{desc} - click ({center_x}, {center_y})")
        
        if elements:
            return "\n".join(elements)
        else:
            return "No elements detected by OmniParser"
            
    except Exception as e:
        print(f"❌ Failed to parse OmniParser result: {e}", file=sys.stderr)
        return "Failed to parse OmniParser output"

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
    x = int(round(px * (W-1)))
    y = int(round(py * (H-1)))
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
                # Skip header lines like "BTN   role(C|M|N)  k  px py pw ph  conf"
                if role == "role(C|M|N)" or not role.isalpha():
                    continue
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
        
        # Try OmniParser for GUI detection (using Replicate API)
        # Try local OmniParser first, then Replicate as fallback
        omniparser_result = _call_omniparser_vision(image_bytes)
        
        if omniparser_result.get("success"):
            print("✅ Using local OmniParser v2.0 for GUI detection", file=sys.stderr)
            hierarchical_gui_data = _parse_omniparser_result(omniparser_result.get("result"), width, height)
        else:
            print("❌ Local OmniParser failed, trying Replicate API...", file=sys.stderr)
            omniparser_result = _call_omniparser_replicate(image_bytes)
            
            if omniparser_result.get("success"):
                print("✅ Using OmniParser v2.0 via Replicate API for GUI detection", file=sys.stderr)
                hierarchical_gui_data = _parse_omniparser_result(omniparser_result.get("result"), width, height)
            else:
                print("❌ Both local and Replicate OmniParser failed, falling back to OpenRouter TSV system", file=sys.stderr)
                # Fallback to window/taskbar TSV system
                fallback_description = _analyze_windows_and_taskbars(image_bytes, width, height)
                # Create simple hierarchical structure for fallback
                hierarchical_gui_data = {
                    "desktop": {"taskbar": [], "background_elements": []},
                    "windows": {},
                    "global_controls": [],
                    "summary": {"total_elements": 0, "total_windows": 0},
                    "fallback_description": fallback_description
                }
        
        return {
            "success": True,
            "screen_info": {
                "width": width,
                "height": height,
                "description": "QMP Desktop Screenshot"
            },
            "gui_hierarchy": hierarchical_gui_data,
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
            
            if '"actions":' in raw_input:
                # For actions, just parse using json module since it's more complex
                try:
                    data = json.loads(raw_input)
                    actions = data.get('actions', [])
                    print(f"DEBUG: Extracted actions: {actions}", file=sys.stderr)
                except Exception as e:
                    print(f'{{"success": false, "error": "Failed to parse actions: {str(e)}"}}')
                    sys.exit(1)
            
            elif '"text":' in raw_input:
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