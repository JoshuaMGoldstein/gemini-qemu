#!/usr/bin/env python3
"""
Simple test to analyze a 300x300 box region to measure response time
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


def analyze_single_box(image_bytes: bytes, x1: int, y1: int, x2: int, y2: int) -> dict:
    """
    Analyze a single 300x300 box region
    """
    system_prompt = """You are a UI element detector. Analyze this FULL screenshot but ONLY examine the specified rectangular region.
    
For each UI element within ONLY that specific rectangular area, provide:
id|type|text|center_x,center_y|x1,y1,x2,y2|clickable|enabled|state

Where:
- center_x,center_y = coordinates to click (center of element)
- x1,y1,x2,y2 = bounding box of element
- clickable = 1 if clickable, 0 if not
- enabled = 1 if enabled, 0 if disabled
- state = n (normal), h (highlighted), d (disabled)"""
    
    user_prompt = f"""Analyze this FULL screenshot but ONLY look at the rectangular region from ({x1},{y1}) to ({x2},{y2}).

This is a 300x300 pixel area. Find ALL UI elements within this specific rectangle only.

Provide each element in this format:
element_id|element_type|text_content|center_x,center_y|x1,y1,x2,y2|clickable|enabled|state

Be extremely precise with coordinates. Ignore everything outside the specified region."""

    try:
        print(f"📡 Analyzing box region ({x1},{y1}) to ({x2},{y2})...")
        start_time = time.time()
        
        response = call_chat_vision(
            model="mistralai/pixtral-12b",
            system_prompt=system_prompt,
            user_text=user_prompt,
            image_bytes=image_bytes,
            temperature=0.1,
            timeout=30,
            extra={
                "max_tokens": 2048,
                "provider": {"order": ["mistral"]}
            }
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Box analysis completed in {elapsed:.1f} seconds")
        
        # Extract content from response
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            return {"success": True, "content": content, "time": elapsed}
        else:
            return {"success": False, "error": "No content in response"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


def parse_box_results(analysis_result: str) -> list:
    """Parse the box analysis result into list of element strings"""
    elements = []
    lines = analysis_result.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        # Look for lines that match the element format
        if '|' in line and line.count('|') >= 6:
            elements.append(line)
    
    return elements


def run_simple_box_test():
    """Main test function for simple box analysis"""
    print("🎯 Simple 300x300 Box Analysis Test")
    print("=" * 50)
    
    # Step 1: Take full screenshot
    print("📸 Taking full screenshot...")
    with tempfile.NamedTemporaryFile(suffix='.ppm', delete=False) as tmp:
        screenshot_path = tmp.name
    
    if not take_screenshot(screenshot_path):
        print("❌ Failed to take screenshot")
        return
    
    print(f"✅ Screenshot saved: {screenshot_path}")
    
    # Convert to JPEG for vision analysis
    jpeg_bytes = compress_image_to_jpeg(screenshot_path)
    print(f"📦 Compressed to {len(jpeg_bytes)} bytes")
    
    # Get image dimensions
    img = Image.open(screenshot_path)
    width, height = img.size
    print(f"🖼️  Image size: {width}x{height}")
    
    # Test a 300x300 box in the center of the screen
    center_x = width // 2
    center_y = height // 2
    x1 = center_x - 150  # 300x300 box centered
    y1 = center_y - 150
    x2 = center_x + 150
    y2 = center_y + 150
    
    print(f"🎯 Testing 300x300 box: ({x1},{y1}) to ({x2},{y2})")
    
    # Analyze the box
    analysis_result = analyze_single_box(jpeg_bytes, x1, y1, x2, y2)
    
    if not analysis_result['success']:
        print(f"❌ Analysis failed: {analysis_result.get('error', 'Unknown error')}")
        return
    
    # Parse elements
    elements = parse_box_results(analysis_result['content'])
    
    # Results
    print(f"\n📋 RESULTS")
    print("=" * 30)
    print(f"Analysis time: {analysis_result['time']:.1f} seconds")
    print(f"Elements found: {len(elements)}")
    print(f"Box area: 300x300 = 90,000 pixels")
    print(f"Elements per second: {len(elements) / analysis_result['time']:.1f}")
    
    if len(elements) > 0:
        print(f"\n🔍 Detected elements:")
        for i, element in enumerate(elements, 1):
            print(f"  {i:2d}. {element}")
    else:
        print("\n📄 Full response:")
        print(analysis_result['content'][:500] + "..." if len(analysis_result['content']) > 500 else analysis_result['content'])
    
    # Save results
    results = {
        "approach": "single_box_300x300",
        "model": "mistralai/pixtral-12b",
        "provider": "mistral",
        "analysis_time": analysis_result['time'],
        "box_region": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "elements_found": len(elements),
        "elements": elements,
        "raw_response": analysis_result['content']
    }
    
    results_path = "/tmp/simple_box_test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")
    
    # Cleanup
    try:
        Path(screenshot_path).unlink()
    except:
        pass


if __name__ == "__main__":
    # Unset environment variable for clean test
    import os
    os.environ.pop('OPENROUTER_MODEL', None)
    
    run_simple_box_test()