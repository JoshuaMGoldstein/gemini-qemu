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
    """Call OpenRouter vision API with raw output"""
    try:
        # Get image dimensions  
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes))
        full_width, full_height = img.size
        
        # Just run Pass A and return raw output
        pass_a_result = _pass_a_layout_sweep_full_resolution(image_bytes, full_width, full_height)
        
        if pass_a_result['success']:
            return pass_a_result['content']
        else:
            return f"Vision analysis failed: {pass_a_result.get('error', 'Unknown error')}"
        
    except Exception as e:
        # Fallback to simple analysis if analysis fails
        return _call_simple_vision_analysis(image_bytes)


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
        pass_a_result = _pass_a_layout_sweep_full_resolution(image_bytes, full_width, full_height)
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
                        type_text, coords = line.split('@')
                        type_part, text_part = type_text.split(':', 1)
                        coords_clean = coords.replace('**', '').strip()  # Clean coords too
                        x, y = map(int, coords_clean.split(','))
                        
                        element = {
                            'id': f'e{i+1}',
                            'type': type_part.strip(),
                            'text': text_part.strip(),
                            'center': [x, y],
                            'bounds': [x-50, y-25, x+50, y+25]  # Estimate bounds
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


def _pass_a_layout_sweep_full_resolution(image_bytes: bytes, width: int, height: int) -> dict:
    """Pass A: Full resolution layout sweep - containers + major text only"""
    user_prompt = f"""Analyze this {width}×{height} desktop screenshot. List UI elements in simple format.

Format: TYPE:TEXT@X,Y

PRIORITY: Find window title bar controls (close X, minimize -, maximize buttons)
Also find: windows, dialogs, major buttons, important text

Examples:
window:LXTerminal@400,200
close_button:X@785,35
minimize_button:-@765,35
maximize_button:□@745,35
button:OK@500,300

Max 30 items. Focus on clickable elements and window controls."""

    try:
        response = call_chat_vision(
            model="qwen/qwen2.5-vl-32b-instruct",
            system_prompt="",
            user_text=user_prompt,
            image_bytes=image_bytes,
            temperature=0.1,
            timeout=60,
            extra={"max_tokens": 8192}
        )
        
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            return {"success": True, "content": content}
        else:
            return {"success": False, "error": "No content in response"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


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
    import json
    
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
        def compress_image_to_jpeg(image_path: str, quality: int = 85) -> bytes:
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
        
        # Get image dimensions
        img = Image.open(screenshot_path)
        width, height = img.size
        
        # Analyze using adaptive partitioning with JPEG bytes
        vision_description = _call_openrouter_vision(image_bytes)
        
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
                
        client.disconnect()
        
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


def send_keyboard_input(host: str, port: int, actions: List[Dict], vm_target: str = "local") -> Dict[str, Any]:
    """Send keyboard input via QMP or VNC"""
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
        
        client = api.connect(vnc_address, timeout=5)
        
        success_count = 0
        errors = []
        
        for action in actions:
            try:
                action_type = action['action']
                
                if action_type == 'type':
                    text = action.get('text', '')
                    # Type text using vncdotool
                    client.type(text)
                    
                elif action_type == 'key':
                    key = action.get('key', '')
                    
                    # Handle key combinations
                    if '+' in key:
                        # Split combination like "ctrl+c"
                        keys = key.split('+')
                        key_str = ' '.join(keys)
                        client.keyPress(key_str)
                    else:
                        # Single key
                        client.keyPress(key)
                        
                success_count += 1
                time.sleep(0.05)  # Small delay between actions
                
            except Exception as e:
                errors.append(f"VNC action {action}: {str(e)}")
                
        client.disconnect()
        
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


def main():
    """Main entry point for the VNC tools"""
    if len(sys.argv) < 5:
        print(json.dumps({"success": False, "error": "Invalid arguments"}))
        sys.exit(1)
        
    command = sys.argv[1]
    host = sys.argv[2]
    port = int(sys.argv[3])
    vm_target = sys.argv[4]
    
    if command == "screenshot":
        result = analyze_screenshot(host, port, vm_target)
        result["vm_target"] = vm_target
        result["vnc_address"] = f"{host}:{port}"
        print(json.dumps(result))
        
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
        
    elif command == "keyboard":
        # Read actions from stdin
        try:
            data = json.loads(sys.stdin.read())
            actions = data.get('actions', [])
        except:
            actions = []
            
        result = send_keyboard_input(host, port, actions, vm_target)
        result["vm_target"] = vm_target
        result["vnc_address"] = f"{host}:{port}"
        print(json.dumps(result))
        
    else:
        print(json.dumps({"success": False, "error": f"Unknown command: {command}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()