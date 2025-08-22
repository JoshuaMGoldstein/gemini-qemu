#!/usr/bin/env python3
"""
Test a single region 200,200 to 500,500 with the full unmodified screenshot
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


def analyze_region(image_bytes: bytes) -> dict:
    """
    Analyze region 200,200 to 500,500 using the exact prompt provided
    """
    system_prompt = ""  # No system prompt
    
    user_prompt = """You are analyzing a desktop screenshot at 1280×800 pixels. Do not rescale. Origin is top-left (0,0). Return only JSON with an array elements, each with: type, text, center:[x,y] (integers), bounds:[x1,y1,x2,y2] (integers), and confidence (0–1). Include every visible interactive or text element (windows, buttons, inputs, icons, menus, dialogs) in the lower-right region starting from coordinates 400,400. Example schema: {elements:[{type:'button',text:'OK',center:[640,742],bounds:[612,728,668,756],confidence:0.93}]}. If unsure, still output your best guess with lower confidence. No prose."""

    try:
        print(f"📡 Analyzing lower-right region from 400,400...")
        start_time = time.time()
        
        response = call_chat_vision(
            model="opengvlab/internvl3-14b",
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
        print(f"✅ Analysis completed in {elapsed:.1f} seconds")
        
        # Extract content from response
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            return {"success": True, "content": content, "time": elapsed}
        else:
            return {"success": False, "error": "No content in response"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_single_region_test():
    """Test single region 200,200 to 500,500"""
    print("🔦 Lower-Right Region Test: 400,400 to bottom-right")
    print("=" * 45)
    
    # Take screenshot
    print("📸 Taking screenshot...")
    with tempfile.NamedTemporaryFile(suffix='.ppm', delete=False) as tmp_ppm:
        ppm_path = tmp_ppm.name
    
    if not take_screenshot(ppm_path):
        print("❌ Failed to take screenshot")
        return
    
    # Convert PPM to PNG (no quality loss)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_png:
        png_path = tmp_png.name
    
    if not convert_ppm_to_png(ppm_path, png_path):
        print("❌ Failed to convert PPM to PNG")
        return
    
    # Read PNG bytes (full unmodified screenshot)
    with open(png_path, 'rb') as f:
        image_bytes = f.read()
    
    # Get image dimensions
    img = Image.open(png_path)
    width, height = img.size
    print(f"🖼️  Full image size: {width}x{height}")
    print(f"📦 PNG image: {len(image_bytes)} bytes")
    print(f"🎯 Analyzing: Lower-right region from 400,400 to {width},{height}")
    
    # Analyze the region
    result = analyze_region(image_bytes)
    
    if not result['success']:
        print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
        return
    
    # Show results
    print(f"\n📋 RESULTS")
    print("=" * 20)
    print(f"Analysis time: {result['time']:.1f} seconds")
    print(f"Model: opengvlab/internvl3-14b")
    print(f"Image format: PNG (no compression)")
    
    print(f"\n📄 Full response:")
    print(result['content'])
    
    # Save results
    results = {
        "approach": "lower_right_region_400_400_to_bottom_right",
        "model": "opengvlab/internvl3-14b",
        "analysis_time": result['time'],
        "image_format": "PNG",
        "image_size": {"width": width, "height": height},
        "target_region": {"x1": 400, "y1": 400, "x2": "bottom_right", "y2": "bottom_right"},
        "raw_response": result['content']
    }
    
    results_path = "/tmp/single_region_results.json"
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
    
    run_single_region_test()