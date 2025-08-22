#!/usr/bin/env python3
"""
Test arcee-ai/spotlight model with JSON-only prompt format
Keep images in original format (no JPG conversion)
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


def analyze_screenshot_json(image_bytes: bytes) -> dict:
    """
    Analyze screenshot using JSON-only prompt format
    """
    system_prompt = ""  # No system prompt needed
    
    user_prompt = """You are analyzing a desktop screenshot at 800×600 pixels. Do not rescale. Origin is top-left (0,0). Return only JSON with an array elements, each with: type, text, center:[x,y] (integers), bounds:[x1,y1,x2,y2] (integers), and confidence (0–1). Include every visible interactive or text element (windows, buttons, inputs, icons, menus, dialogs). Example schema: {elements:[{type:'button',text:'OK',center:[640,742],bounds:[612,728,668,756],confidence:0.93}]}. If unsure, still output your best guess with lower confidence. No prose."""

    try:
        print(f"📡 Analyzing screenshot with arcee-ai/spotlight...")
        start_time = time.time()
        
        response = call_chat_vision(
            model="arcee-ai/spotlight",
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
        print(f"✅ Analysis completed in {elapsed:.1f} seconds")
        
        # Extract content from response
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            return {"success": True, "content": content, "time": elapsed}
        else:
            return {"success": False, "error": "No content in response"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


def parse_json_response(content: str) -> dict:
    """Parse JSON response from the model"""
    try:
        # Try to extract JSON from the response
        # Look for the first { and last } to handle any extra text
        start = content.find('{')
        end = content.rfind('}') + 1
        
        if start == -1 or end == 0:
            return {"success": False, "error": "No JSON found in response"}
        
        json_str = content[start:end]
        parsed = json.loads(json_str)
        
        if "elements" in parsed:
            return {"success": True, "elements": parsed["elements"]}
        else:
            return {"success": False, "error": "No elements array in JSON"}
            
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON parse error: {e}", "raw_content": content[:500]}
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_spotlight_test():
    """Main test function for Spotlight JSON analysis"""
    print("🔦 Spotlight JSON Analysis Test")
    print("=" * 40)
    
    # Step 1: Take screenshot
    print("📸 Taking screenshot...")
    with tempfile.NamedTemporaryFile(suffix='.ppm', delete=False) as tmp_ppm:
        ppm_path = tmp_ppm.name
    
    if not take_screenshot(ppm_path):
        print("❌ Failed to take screenshot")
        return
    
    print(f"✅ Screenshot saved: {ppm_path}")
    
    # Step 2: Convert PPM to PNG (no quality loss)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_png:
        png_path = tmp_png.name
    
    if not convert_ppm_to_png(ppm_path, png_path):
        print("❌ Failed to convert PPM to PNG")
        return
    
    # Read PNG bytes (no JPEG compression)
    with open(png_path, 'rb') as f:
        image_bytes = f.read()
    
    print(f"📦 PNG image: {len(image_bytes)} bytes")
    
    # Get image dimensions
    img = Image.open(png_path)
    width, height = img.size
    print(f"🖼️  Image size: {width}x{height}")
    
    # Step 3: Analyze using Spotlight
    analysis_result = analyze_screenshot_json(image_bytes)
    
    if not analysis_result['success']:
        print(f"❌ Analysis failed: {analysis_result.get('error', 'Unknown error')}")
        return
    
    # Step 4: Parse JSON response
    parsed_result = parse_json_response(analysis_result['content'])
    
    if not parsed_result['success']:
        print(f"❌ JSON parsing failed: {parsed_result.get('error', 'Unknown error')}")
        if 'raw_content' in parsed_result:
            print(f"📄 Raw response (first 500 chars):")
            print(parsed_result['raw_content'])
        else:
            print(f"📄 Full response:")
            print(analysis_result['content'][:800])
        return
    
    elements = parsed_result['elements']
    
    # Step 5: Results
    print(f"\n📋 RESULTS")
    print("=" * 30)
    print(f"Analysis time: {analysis_result['time']:.1f} seconds")
    print(f"Elements found: {len(elements)}")
    print(f"Image format: PNG (no compression)")
    print(f"Image size: {width}x{height}")
    
    if len(elements) > 0:
        print(f"\n🔍 Detected elements (first 15):")
        for i, element in enumerate(elements[:15], 1):
            elem_type = element.get('type', 'unknown')
            text = element.get('text', '')
            center = element.get('center', [0, 0])
            bounds = element.get('bounds', [0, 0, 0, 0])
            confidence = element.get('confidence', 0.0)
            
            print(f"  {i:2d}. {elem_type}: '{text}' center:{center} bounds:{bounds} conf:{confidence:.2f}")
        
        if len(elements) > 15:
            print(f"  ... and {len(elements) - 15} more elements")
    
    # Step 6: Save results
    results = {
        "approach": "spotlight_json_analysis",
        "model": "arcee-ai/spotlight",
        "analysis_time": analysis_result['time'],
        "image_format": "PNG",
        "image_size": {"width": width, "height": height},
        "elements_found": len(elements),
        "elements": elements,
        "raw_response": analysis_result['content']
    }
    
    results_path = "/tmp/spotlight_json_results.json"
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
    # Unset environment variable for clean test
    import os
    os.environ.pop('OPENROUTER_MODEL', None)
    
    run_spotlight_test()