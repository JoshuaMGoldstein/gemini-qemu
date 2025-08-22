#!/usr/bin/env python3
"""
Test arcee-ai/spotlight with different regions of the same image
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


def convert_ppm_to_png(ppm_path: str, png_path: str) -> bool:
    """Convert PPM to PNG without quality loss"""
    try:
        img = Image.open(ppm_path)
        img.save(png_path, format='PNG')
        return True
    except Exception as e:
        print(f"PPM to PNG conversion failed: {e}")
        return False


def analyze_region_json(image_bytes: bytes, region_name: str, x1: int, y1: int, x2: int, y2: int, image_width: int, image_height: int) -> dict:
    """
    Analyze a specific region using JSON-only prompt format
    """
    system_prompt = ""  # No system prompt needed
    
    user_prompt = f"""You are analyzing a desktop screenshot at {image_width}×{image_height} pixels. Focus ONLY on the rectangular region from ({x1},{y1}) to ({x2},{y2}). Do not rescale. Origin is top-left (0,0). Return only JSON with an array elements, each with: type, text, center:[x,y] (integers), bounds:[x1,y1,x2,y2] (integers), and confidence (0–1). Include every visible interactive or text element (windows, buttons, inputs, icons, menus, dialogs, files, folders) within this specific region. Example schema: {{"elements":[{{"type":"button","text":"OK","center":[640,742],"bounds":[612,728,668,756],"confidence":0.93}}]}}. If unsure, still output your best guess with lower confidence. No prose. Analyze region: {region_name}"""

    try:
        print(f"📡 Analyzing {region_name} region ({x1},{y1}) to ({x2},{y2})...")
        start_time = time.time()
        
        response = call_chat_vision(
            model="arcee-ai/spotlight",
            system_prompt=system_prompt,
            user_text=user_prompt,
            image_bytes=image_bytes,
            temperature=0.1,
            timeout=30,
            extra={
                "max_tokens": 4096
            }
        )
        
        elapsed = time.time() - start_time
        print(f"✅ {region_name} analysis completed in {elapsed:.1f} seconds")
        
        # Extract content from response
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            return {"success": True, "content": content, "time": elapsed, "region": region_name}
        else:
            return {"success": False, "error": "No content in response", "region": region_name}
            
    except Exception as e:
        return {"success": False, "error": str(e), "region": region_name}


def parse_json_response(content: str) -> dict:
    """Parse JSON response from the model - handle both array and object formats"""
    try:
        # Strip code fences if present
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        # Try to find JSON (either array or object)
        start_array = content.find('[')
        start_obj = content.find('{')
        
        # Determine which comes first
        if start_array != -1 and (start_obj == -1 or start_array < start_obj):
            # It's an array format
            end = content.rfind(']') + 1
            if end == 0:
                return {"success": False, "error": "No closing bracket found"}
            json_str = content[start_array:end]
            parsed = json.loads(json_str)
            return {"success": True, "elements": parsed}
            
        elif start_obj != -1:
            # It's an object format
            end = content.rfind('}') + 1
            if end == 0:
                return {"success": False, "error": "No closing brace found"}
            json_str = content[start_obj:end]
            parsed = json.loads(json_str)
            
            if "elements" in parsed:
                return {"success": True, "elements": parsed["elements"]}
            else:
                return {"success": False, "error": "No elements array in JSON"}
        else:
            return {"success": False, "error": "No JSON found in response"}
            
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON parse error: {e}", "raw_content": content[:500]}
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_multi_region_test():
    """Test multiple regions of the same image"""
    print("🔦 Spotlight Multi-Region Analysis Test")
    print("=" * 50)
    
    # Step 1: Take screenshot
    print("📸 Taking screenshot...")
    with tempfile.NamedTemporaryFile(suffix='.ppm', delete=False) as tmp_ppm:
        ppm_path = tmp_ppm.name
    
    if not take_screenshot(ppm_path):
        print("❌ Failed to take screenshot")
        return
    
    # Convert PPM to PNG
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_png:
        png_path = tmp_png.name
    
    if not convert_ppm_to_png(ppm_path, png_path):
        print("❌ Failed to convert PPM to PNG")
        return
    
    # Read PNG bytes
    with open(png_path, 'rb') as f:
        image_bytes = f.read()
    
    # Get image dimensions
    img = Image.open(png_path)
    width, height = img.size
    print(f"🖼️  Image size: {width}x{height}")
    print(f"📦 PNG image: {len(image_bytes)} bytes")
    
    # Define regions to test
    regions = [
        # Top area - window title bars and controls
        {"name": "Top Area", "x1": 0, "y1": 0, "x2": width, "y2": 200},
        
        # Left side - file/folder list area  
        {"name": "Left File List", "x1": 0, "y1": 100, "x2": width//2, "y2": 400},
        
        # Right side - file/folder list area
        {"name": "Right File List", "x1": width//2, "y1": 100, "x2": width, "y2": 400},
        
        # Center area - main content
        {"name": "Center Area", "x1": width//4, "y1": height//4, "x2": 3*width//4, "y2": 3*height//4},
        
        # Bottom area - taskbar and buttons
        {"name": "Bottom Area", "x1": 0, "y1": height-150, "x2": width, "y2": height},
    ]
    
    print(f"\n🔍 Testing {len(regions)} regions:")
    for region in regions:
        print(f"  - {region['name']}: ({region['x1']},{region['y1']}) to ({region['x2']},{region['y2']})")
    
    # Test each region
    all_results = []
    total_elements = 0
    total_time = 0
    
    for region in regions:
        print(f"\n--- {region['name']} ---")
        
        result = analyze_region_json(
            image_bytes, 
            region['name'],
            region['x1'], region['y1'], region['x2'], region['y2'],
            width, height
        )
        
        if result['success']:
            parsed = parse_json_response(result['content'])
            
            if parsed['success']:
                elements = parsed['elements']
                print(f"✅ Found {len(elements)} elements in {result['time']:.1f}s")
                
                # Show first few elements
                for i, element in enumerate(elements[:5], 1):
                    elem_type = element.get('type', 'unknown')
                    text = element.get('text', '')[:20] + ('...' if len(element.get('text', '')) > 20 else '')
                    center = element.get('center', [0, 0])
                    confidence = element.get('confidence', 0.0)
                    print(f"  {i}. {elem_type}: '{text}' center:{center} conf:{confidence:.2f}")
                
                if len(elements) > 5:
                    print(f"  ... and {len(elements) - 5} more")
                
                total_elements += len(elements)
                total_time += result['time']
                all_results.append({
                    "region": region,
                    "elements": elements,
                    "count": len(elements),
                    "time": result['time']
                })
            else:
                print(f"❌ JSON parse failed: {parsed['error']}")
                if 'raw_content' in parsed:
                    print(f"📄 Raw: {parsed['raw_content']}")
        else:
            print(f"❌ Analysis failed: {result['error']}")
    
    # Summary
    print(f"\n📋 FINAL SUMMARY")
    print("=" * 30)
    print(f"Total regions tested: {len(regions)}")
    print(f"Total elements found: {total_elements}")
    print(f"Total analysis time: {total_time:.1f} seconds")
    print(f"Average time per region: {total_time/len(regions):.1f} seconds")
    print(f"Average elements per region: {total_elements/len(regions):.1f}")
    
    # Save results
    summary = {
        "approach": "spotlight_multi_region",
        "model": "arcee-ai/spotlight", 
        "image_size": {"width": width, "height": height},
        "regions_tested": len(regions),
        "total_elements": total_elements,
        "total_time": total_time,
        "results": all_results
    }
    
    results_path = "/tmp/spotlight_regions_results.json"
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")
    
    # Cleanup
    try:
        Path(ppm_path).unlink()
        Path(png_path).unlink()
    except:
        pass


if __name__ == "__main__":
    # Unset environment variable for clean test
    import os
    os.environ.pop('OPENROUTER_MODEL', None)
    
    run_multi_region_test()