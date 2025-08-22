#!/usr/bin/env python3
"""
Two-pass analysis with InternVL3-14B for maximum accuracy
Pass A: Layout sweep (containers + major text, cap 60)
Pass B: Per-container detail (controls inside containers, cap 80 each)
"""

import sys
import json
import time
from pathlib import Path
from PIL import Image
import tempfile

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


def pass_a_layout_sweep(image_bytes: bytes, width: int, height: int) -> dict:
    """
    Pass A: Layout sweep - containers + major text only
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
        print(f"📡 Pass A: Layout sweep...")
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


def pass_b_container_detail(image_bytes: bytes, container_id: str, container_bounds: list, width: int, height: int) -> dict:
    """
    Pass B: Per-container detail analysis
    """
    system_prompt = ""
    
    x1, y1, x2, y2 = container_bounds
    
    user_prompt = f"""You are analyzing a desktop screenshot at {width}×{height} pixels. Do not rescale. Origin is top-left (0,0).

Pass B — Per-container detail
Only enumerate CHILDREN inside parent="{container_id}" with bounds=[{x1},{y1},{x2},{y2}].
Allowed types: ["button","checkbox","radio","toggle","input","dropdown","list_item","link","icon","image","scrollbar","progress","chip","badge","pagination","breadcrumb","text"].
Do not include the parent container itself. Hard cap per container: 80 elements.

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

Follow this structure exactly. Return only JSON. No prose. Focus only on interactive controls and detailed elements within the specified container bounds."""

    try:
        print(f"📡 Pass B: Analyzing container {container_id} at {container_bounds}...")
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


def merge_results(pass_a_elements: list, pass_b_results: list) -> list:
    """Merge Pass A and Pass B results with global ID renumbering"""
    all_elements = []
    id_counter = 1
    
    # Add Pass A containers with new IDs
    id_mapping = {}  # old_id -> new_id
    
    for element in pass_a_elements:
        old_id = element.get('id', f'container_{id_counter}')
        new_id = f'e{id_counter}'
        id_mapping[old_id] = new_id
        
        element['id'] = new_id
        all_elements.append(element)
        id_counter += 1
    
    # Add Pass B elements with updated parent references
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


def run_two_pass_analysis():
    """Run the complete two-pass analysis"""
    print("🔄 Two-Pass Analysis with InternVL3-14B")
    print("=" * 50)
    
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
    
    # Read image
    with open(png_path, 'rb') as f:
        image_bytes = f.read()
    
    img = Image.open(png_path)
    width, height = img.size
    print(f"🖼️  Image size: {width}x{height}")
    print(f"📦 PNG image: {len(image_bytes)} bytes")
    
    # Pass A: Layout sweep
    print(f"\n--- PASS A: Layout Sweep ---")
    pass_a_result = pass_a_layout_sweep(image_bytes, width, height)
    
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
    print(f"✅ Pass A found {len(containers)} containers")
    
    # Show containers found
    for i, container in enumerate(containers[:5], 1):
        container_type = container.get('type', 'unknown')
        text = container.get('text', '')[:30] + ('...' if len(container.get('text', '')) > 30 else '')
        bounds = container.get('bounds', [0,0,0,0])
        confidence = container.get('confidence', 0.0)
        print(f"  {i}. {container_type}: '{text}' bounds:{bounds} conf:{confidence:.2f}")
    
    if len(containers) > 5:
        print(f"  ... and {len(containers) - 5} more containers")
    
    # Pass B: Per-container detail for each container
    print(f"\n--- PASS B: Per-Container Detail ---")
    pass_b_results = []
    total_detail_time = 0
    
    for container in containers:
        container_id = container.get('id', 'unknown')
        container_bounds = container.get('bounds', [0, 0, width, height])
        
        # Skip if container is too small
        if len(container_bounds) == 4:
            x1, y1, x2, y2 = container_bounds
            if (x2 - x1) * (y2 - y1) < 1000:  # Skip tiny containers
                continue
        
        detail_result = pass_b_container_detail(image_bytes, container_id, container_bounds, width, height)
        
        if detail_result['success']:
            parsed_b = parse_json_response(detail_result['content'])
            
            if parsed_b['success']:
                elements = parsed_b['elements']
                print(f"✅ Container {container_id}: {len(elements)} detail elements")
                pass_b_results.append(elements)
                total_detail_time += detail_result['time']
            else:
                print(f"❌ Container {container_id}: JSON parse failed")
        else:
            print(f"❌ Container {container_id}: Analysis failed")
    
    # Merge all results
    print(f"\n--- MERGING RESULTS ---")
    all_elements = merge_results(containers, pass_b_results)
    
    # Final results
    total_time = pass_a_result['time'] + total_detail_time
    print(f"\n📋 FINAL RESULTS")
    print("=" * 30)
    print(f"Pass A time: {pass_a_result['time']:.1f}s")
    print(f"Pass B time: {total_detail_time:.1f}s")
    print(f"Total time: {total_time:.1f}s")
    print(f"Containers found: {len(containers)}")
    print(f"Total elements: {len(all_elements)}")
    
    # Save results
    results = {
        "approach": "two_pass_analysis",
        "model": "opengvlab/internvl3-14b",
        "pass_a_time": pass_a_result['time'],
        "pass_b_time": total_detail_time,
        "total_time": total_time,
        "image_size": {"width": width, "height": height},
        "containers_found": len(containers),
        "total_elements": len(all_elements),
        "elements": all_elements
    }
    
    results_path = "/tmp/two_pass_analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {results_path}")
    
    # Cleanup
    try:
        Path(ppm_path).unlink()
        Path(png_path).unlink()
    except:
        pass


if __name__ == "__main__":
    import os
    os.environ.pop('OPENROUTER_MODEL', None)
    
    run_two_pass_analysis()