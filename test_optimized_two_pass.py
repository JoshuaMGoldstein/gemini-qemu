#!/usr/bin/env python3
"""
Optimized two-pass analysis with InternVL3-14B for 15-25 second performance target
- Pass A: Downscaled layout sweep (960×600, containers + major text, cap 60)
- Pass B: Parallel per-container detail (hard cropped, 3-5 concurrent, cap 80 each)
"""

import sys
import json
import time
import math
import random
from pathlib import Path
from PIL import Image
import tempfile
import concurrent.futures
import threading

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from qmp_client import QMPClient
from openrouter import call_chat_vision


def take_screenshot(output_path: str) -> bool:
    """Take a screenshot using QMP"""
    try:
        client = QMPClient('127.0.0.1', 4444)
        client.connect()
        success = client.screendump(output_path)
        client.disconnect()
        return success
    except Exception as e:
        print(f"Screenshot failed: {e}")
        return False


def convert_ppm_to_png(pmp_path: str, png_path: str) -> bool:
    """Convert PPM to PNG without quality loss"""
    try:
        img = Image.open(pmp_path)
        img.save(png_path, format='PNG')
        return True
    except Exception as e:
        print(f"PPM to PNG conversion failed: {e}")
        return False


def downscale_image(image_bytes: bytes, target_width: int = 960, target_height: int = 600) -> tuple[bytes, float, float]:
    """Downscale image for Pass A with scaling factors for coordinate mapping"""
    try:
        # Open image from bytes
        import io
        img = Image.open(io.BytesIO(image_bytes))
        original_width, original_height = img.size
        
        # Calculate scaling factors
        scale_x = original_width / target_width
        scale_y = original_height / target_height
        
        # Downscale image
        downscaled_img = img.resize((target_width, target_height), Image.LANCZOS)
        
        # Convert back to bytes
        output = io.BytesIO()
        downscaled_img.save(output, format='PNG')
        downscaled_bytes = output.getvalue()
        
        return downscaled_bytes, scale_x, scale_y
    except Exception as e:
        print(f"Image downscaling failed: {e}")
        return image_bytes, 1.0, 1.0


def crop_container_image(image_bytes: bytes, container_bounds: list, padding: int = 20) -> bytes:
    """Hard crop image to container bounds + padding for Pass B"""
    try:
        import io
        img = Image.open(io.BytesIO(image_bytes))
        
        x1, y1, x2, y2 = container_bounds
        
        # Add padding but stay within image bounds
        img_width, img_height = img.size
        crop_x1 = max(0, x1 - padding)
        crop_y1 = max(0, y1 - padding)
        crop_x2 = min(img_width, x2 + padding)
        crop_y2 = min(img_height, y2 + padding)
        
        # Crop the image
        cropped_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        
        # Convert back to bytes
        output = io.BytesIO()
        cropped_img.save(output, format='PNG')
        cropped_bytes = output.getvalue()
        
        return cropped_bytes
    except Exception as e:
        print(f"Image cropping failed: {e}")
        return image_bytes


def pass_a_layout_sweep_downscaled(image_bytes: bytes, width: int, height: int) -> dict:
    """
    Pass A: Downscaled layout sweep - containers + major text only
    """
    system_prompt = ""
    
    user_prompt = f"""You are analyzing a desktop screenshot at {width}×{height} pixels. Do not rescale. Origin is top-left (0,0).

Pass A — Layout sweep (containers + major text, cap 60)
Enumerate ONLY: ["window","dialog","menu","toolbar","sidebar","panel","tab","statusbar","text"].
Cap elements at 60. No children of controls.

Example (snippet, {width}x{height}):
{{
  "screen_size":[{width},{height}],
  "elements":[
    {{"id":"e1","type":"dialog","text":"","bounds":[420,220,860,520],"center":[640,370],
     "state":{{"checked":null,"disabled":false,"selected":true,"active":true}},"role":"dialog",
     "parent":null,"z_index":7,"confidence":0.98}},
    {{"id":"e2","type":"text","text":"Are you sure?","bounds":[460,260,820,292],"center":[640,276],
     "state":{{"checked":null,"disabled":false,"selected":false,"active":false}},"role":"staticText",
     "parent":"e1","z_index":8,"confidence":0.96}}
  ]
}}

Follow this structure exactly. Return only JSON. No prose. Reject controls like buttons/inputs at this stage. Focus on layout containers and major text blocks."""

    try:
        print(f"📡 Pass A: Downscaled layout sweep ({width}x{height})...")
        start_time = time.time()
        
        response = call_chat_vision(
            model="opengvlab/internvl3-14b",
            system_prompt=system_prompt,
            user_text=user_prompt,
            image_bytes=image_bytes,
            temperature=0.1,
            timeout=60,
            extra={
                "max_tokens": 8192
            }
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Pass A completed in {elapsed:.1f} seconds")
        
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            return {"success": True, "content": content, "time": elapsed}
        else:
            return {"success": False, "error": "No content in response"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


def calculate_adaptive_detail_cap(container_bounds: list, base_cap: int = 80) -> int:
    """Calculate adaptive detail cap based on container area"""
    x1, y1, x2, y2 = container_bounds
    area = (x2 - x1) * (y2 - y1)
    
    # Area thresholds and caps
    if area < 10000:      # Very small containers
        return min(base_cap, 20)
    elif area < 50000:    # Small containers
        return min(base_cap, 40)
    elif area < 100000:   # Medium containers
        return min(base_cap, 60)
    else:                 # Large containers
        return base_cap


def pass_b_container_detail_cropped(image_bytes: bytes, container_id: str, container_bounds: list, 
                                   width: int, height: int, delay: float = 0) -> dict:
    """
    Pass B: Per-container detail analysis with hard cropped image
    """
    # Add jittered delay for rate limiting
    if delay > 0:
        jitter = random.uniform(0.8, 1.2) * delay
        time.sleep(jitter)
    
    system_prompt = ""
    
    x1, y1, x2, y2 = container_bounds
    detail_cap = calculate_adaptive_detail_cap(container_bounds)
    
    # Crop image to container bounds
    cropped_bytes = crop_container_image(image_bytes, container_bounds, padding=20)
    
    user_prompt = f"""You are analyzing a cropped section of a desktop screenshot. The crop shows a container region that was originally at bounds [{x1},{y1},{x2},{y2}] in a {width}×{height} pixel desktop.

Pass B — Per-container detail
Only enumerate CHILDREN inside this cropped container region.
Allowed types: ["button","checkbox","radio","toggle","input","dropdown","list_item","link","icon","image","scrollbar","progress","chip","badge","pagination","breadcrumb","text"].
Hard cap: {detail_cap} elements.

Adjust coordinates to match the original full desktop coordinates (not crop coordinates).
Use "parent":"{container_id}" on all results.

Example structure:
{{
  "screen_size":[{width},{height}],
  "elements":[
    {{"id":"e3","type":"button","text":"OK","bounds":[700,468,804,504],"center":[752,486],
     "state":{{"checked":null,"disabled":false,"selected":true,"active":false}},"role":"button",
     "parent":"{container_id}","z_index":8,"confidence":0.96}}
  ]
}}

Follow this structure exactly. Return only JSON. No prose. Focus only on interactive controls and detailed elements within the container."""

    try:
        print(f"📡 Pass B: Analyzing cropped container {container_id} (cap: {detail_cap})...")
        start_time = time.time()
        
        response = call_chat_vision(
            model="opengvlab/internvl3-14b", 
            system_prompt=system_prompt,
            user_text=user_prompt,
            image_bytes=cropped_bytes,
            temperature=0.1,
            timeout=60,
            extra={
                "max_tokens": 8192
            }
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Pass B for {container_id} completed in {elapsed:.1f} seconds")
        
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            return {"success": True, "content": content, "time": elapsed, "container_id": container_id}
        else:
            return {"success": False, "error": "No content in response", "container_id": container_id}
            
    except Exception as e:
        return {"success": False, "error": str(e), "container_id": container_id}


def parse_json_response(content: str) -> dict:
    """Parse JSON response handling code fences"""
    try:
        # Strip code fences
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        # Parse JSON
        parsed = json.loads(content)
        
        if "elements" in parsed:
            return {"success": True, "elements": parsed["elements"]}
        else:
            return {"success": False, "error": "No elements array in JSON"}
            
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON parse error: {e}", "raw_content": content[:500]}
    except Exception as e:
        return {"success": False, "error": str(e)}


def scale_container_bounds(container_bounds: list, scale_x: float, scale_y: float) -> list:
    """Scale container bounds from downscaled to full resolution"""
    x1, y1, x2, y2 = container_bounds
    return [
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y)
    ]


def merge_results_optimized(pass_a_elements: list, pass_b_results: list, scale_x: float, scale_y: float) -> list:
    """Merge Pass A and Pass B results with global ID renumbering and coordinate scaling"""
    all_elements = []
    id_counter = 1
    
    # Add Pass A containers with new IDs and scaled coordinates
    id_mapping = {}  # old_id -> new_id
    
    for element in pass_a_elements:
        old_id = element.get('id', f'container_{id_counter}')
        new_id = f'e{id_counter}'
        id_mapping[old_id] = new_id
        
        # Scale coordinates back to full resolution
        if 'bounds' in element:
            bounds = element['bounds']
            if len(bounds) == 4:
                element['bounds'] = [
                    int(bounds[0] * scale_x),
                    int(bounds[1] * scale_y),
                    int(bounds[2] * scale_x),
                    int(bounds[3] * scale_y)
                ]
        
        if 'center' in element:
            center = element['center']
            if len(center) == 2:
                element['center'] = [
                    int(center[0] * scale_x),
                    int(center[1] * scale_y)
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


def run_optimized_two_pass_analysis():
    """Run the optimized two-pass analysis targeting 15-25 seconds"""
    print("⚡ Optimized Two-Pass Analysis with InternVL3-14B")
    print("🎯 Target: 15-25 seconds (vs 73.2s baseline)")
    print("=" * 55)
    
    overall_start = time.time()
    
    # Take screenshot
    print("📸 Taking screenshot...")
    with tempfile.NamedTemporaryFile(suffix='.ppm', delete=False) as tmp_ppm:
        ppm_path = tmp_ppm.name
    
    if not take_screenshot(ppm_path):
        print("❌ Failed to take screenshot")
        return
    
    # Convert to PNG
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_png:
        png_path = tmp_png.name
    
    if not convert_ppm_to_png(ppm_path, png_path):
        print("❌ Failed to convert PPM to PNG")
        return
    
    # Read full resolution image
    with open(png_path, 'rb') as f:
        full_image_bytes = f.read()
    
    img = Image.open(png_path)
    full_width, full_height = img.size
    print(f"🖼️  Full image: {full_width}x{full_height}, {len(full_image_bytes)} bytes")
    
    # PASS A: Downscaled layout sweep (960×600)
    print(f"\n--- PASS A: Downscaled Layout Sweep (960×600) ---")
    downscaled_bytes, scale_x, scale_y = downscale_image(full_image_bytes, 960, 600)
    print(f"🔽 Downscaled: scale_x={scale_x:.2f}, scale_y={scale_y:.2f}, {len(downscaled_bytes)} bytes")
    
    pass_a_result = pass_a_layout_sweep_downscaled(downscaled_bytes, 960, 600)
    
    if not pass_a_result['success']:
        print(f"❌ Pass A failed: {pass_a_result.get('error', 'Unknown error')}")
        return
    
    # Parse Pass A results
    parsed_a = parse_json_response(pass_a_result['content'])
    
    if not parsed_a['success']:
        print(f"❌ Pass A JSON parse failed: {parsed_a.get('error', 'Unknown error')}")
        print(f"📄 Raw response: {pass_a_result['content'][:500]}")
        return
    
    containers = parsed_a['elements']
    print(f"✅ Pass A found {len(containers)} containers in {pass_a_result['time']:.1f}s")
    
    # Scale container bounds back to full resolution
    for container in containers:
        if 'bounds' in container:
            container['bounds'] = scale_container_bounds(container['bounds'], scale_x, scale_y)
        if 'center' in container:
            center = container['center']
            container['center'] = [int(center[0] * scale_x), int(center[1] * scale_y)]
    
    # Show containers found
    for i, container in enumerate(containers[:5], 1):
        container_type = container.get('type', 'unknown')
        text = container.get('text', '')[:30] + ('...' if len(container.get('text', '')) > 30 else '')
        bounds = container.get('bounds', [0,0,0,0])
        confidence = container.get('confidence', 0.0)
        print(f"  {i}. {container_type}: '{text}' bounds:{bounds} conf:{confidence:.2f}")
    
    if len(containers) > 5:
        print(f"  ... and {len(containers) - 5} more containers")
    
    # PASS B: Parallel per-container detail with hard cropping
    print(f"\n--- PASS B: Parallel Container Detail (3-5 concurrent) ---")
    pass_b_results = []
    pass_b_times = []
    
    # Filter containers (skip tiny ones)
    valid_containers = []
    for container in containers:
        container_bounds = container.get('bounds', [0, 0, full_width, full_height])
        if len(container_bounds) == 4:
            x1, y1, x2, y2 = container_bounds
            area = (x2 - x1) * (y2 - y1)
            if area >= 1000:  # Skip tiny containers
                valid_containers.append(container)
    
    print(f"📦 Processing {len(valid_containers)} valid containers (skipped {len(containers) - len(valid_containers)} tiny)")
    
    # Parallel processing with 4 workers (middle of 3-5 range)
    max_workers = min(4, len(valid_containers))
    base_delay = 0.3  # 300ms base delay between starts
    
    def analyze_container_wrapper(args):
        container, delay_multiplier = args
        container_id = container.get('id', 'unknown')
        container_bounds = container.get('bounds', [0, 0, full_width, full_height])
        delay = base_delay * delay_multiplier
        
        return pass_b_container_detail_cropped(
            full_image_bytes, container_id, container_bounds, 
            full_width, full_height, delay
        )
    
    # Prepare arguments with staggered delays
    container_args = [(container, i * 0.7) for i, container in enumerate(valid_containers)]
    
    parallel_start = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"🔄 Starting {max_workers} concurrent workers...")
        
        # Submit all tasks
        futures = [executor.submit(analyze_container_wrapper, args) for args in container_args]
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            detail_result = future.result()
            
            if detail_result['success']:
                parsed_b = parse_json_response(detail_result['content'])
                
                if parsed_b['success']:
                    elements = parsed_b['elements']
                    container_id = detail_result.get('container_id', 'unknown')
                    print(f"✅ Container {container_id}: {len(elements)} elements ({detail_result['time']:.1f}s)")
                    pass_b_results.append(elements)
                    pass_b_times.append(detail_result['time'])
                else:
                    container_id = detail_result.get('container_id', 'unknown')
                    print(f"❌ Container {container_id}: JSON parse failed")
            else:
                container_id = detail_result.get('container_id', 'unknown')
                print(f"❌ Container {container_id}: Analysis failed - {detail_result.get('error', 'unknown')}")
    
    parallel_time = time.time() - parallel_start
    
    # Merge all results
    print(f"\n--- MERGING RESULTS ---")
    all_elements = merge_results_optimized(containers, pass_b_results, scale_x, scale_y)
    
    # Final timing
    total_time = time.time() - overall_start
    pass_b_total_time = sum(pass_b_times) if pass_b_times else 0
    
    print(f"\n📋 PERFORMANCE RESULTS")
    print("=" * 40)
    print(f"Pass A time: {pass_a_result['time']:.1f}s (downscaled 960×600)")
    print(f"Pass B time: {parallel_time:.1f}s (parallel, {max_workers} workers)")
    print(f"Pass B sum: {pass_b_total_time:.1f}s (individual call times)")
    print(f"Total time: {total_time:.1f}s 🎯 (target: 15-25s)")
    print(f"Speedup: {73.2/total_time:.1f}x vs baseline (73.2s)")
    print()
    print(f"Containers found: {len(containers)}")
    print(f"Valid containers processed: {len(valid_containers)}")
    print(f"Total elements: {len(all_elements)}")
    print(f"Avg elements/container: {len(all_elements)//len(containers) if containers else 0}")
    
    # Performance assessment
    if total_time <= 25:
        if total_time <= 15:
            print("🏆 EXCELLENT: Under 15 seconds!")
        else:
            print("✅ SUCCESS: Within 15-25 second target!")
    else:
        print(f"⚠️  NEEDS WORK: {total_time - 25:.1f}s over target")
    
    # Save results
    results = {
        "approach": "optimized_two_pass_analysis",
        "model": "opengvlab/internvl3-14b",
        "optimizations": {
            "pass_a_downscaled": "960x600",
            "pass_b_cropped": True,
            "parallel_workers": max_workers,
            "adaptive_caps": True,
            "rate_limiting": "300ms base + jitter"
        },
        "performance": {
            "pass_a_time": pass_a_result['time'],
            "pass_b_parallel_time": parallel_time,
            "pass_b_sum_time": pass_b_total_time,
            "total_time": total_time,
            "target_achieved": total_time <= 25,
            "speedup_vs_baseline": 73.2 / total_time
        },
        "image_size": {"width": full_width, "height": full_height},
        "scaling_factors": {"scale_x": scale_x, "scale_y": scale_y},
        "containers_found": len(containers),
        "valid_containers": len(valid_containers),
        "total_elements": len(all_elements),
        "elements": all_elements
    }
    
    results_path = "/tmp/optimized_two_pass_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")
    
    # Cleanup
    try:
        Path(ppm_path).unlink()
        Path(png_path).unlink()
    except:
        pass


if __name__ == "__main__":
    import os
    os.environ.pop('OPENROUTER_MODEL', None)
    
    run_optimized_two_pass_analysis()