#!/usr/bin/env python3
"""
Test Pixtral with the full image analysis (no region restriction)
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


def analyze_full_image(image_bytes: bytes) -> dict:
    """
    Analyze the full image without region restrictions
    """
    system_prompt = """You are a UI element detector. Analyze this screenshot and identify ALL visible UI elements.
    
For each UI element, provide:
id|type|text|center_x,center_y|x1,y1,x2,y2|clickable|enabled|state

Where:
- center_x,center_y = coordinates to click (center of element)
- x1,y1,x2,y2 = bounding box of element
- clickable = 1 if clickable, 0 if not
- enabled = 1 if enabled, 0 if disabled
- state = n (normal), h (highlighted), d (disabled)"""
    
    user_prompt = """Analyze this screenshot and find ALL UI elements visible in the image.

List every button, icon, text field, file, folder, menu item, and any other interactive or display element you can see.

Provide each element in this exact format:
element_id|element_type|text_content|center_x,center_y|x1,y1,x2,y2|clickable|enabled|state

Be extremely precise with coordinates. Look at the ACTUAL screenshot content and give real coordinates for real elements."""

    try:
        print(f"📡 Analyzing full screenshot...")
        start_time = time.time()
        
        response = call_chat_vision(
            model="mistralai/pixtral-12b",
            system_prompt=system_prompt,
            user_text=user_prompt,
            image_bytes=image_bytes,
            temperature=0.1,
            timeout=60,
            extra={
                "max_tokens": 4096,
                "provider": {"order": ["mistral"]}
            }
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Full image analysis completed in {elapsed:.1f} seconds")
        
        # Extract content from response
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            return {"success": True, "content": content, "time": elapsed}
        else:
            return {"success": False, "error": "No content in response"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


def parse_full_results(analysis_result: str) -> list:
    """Parse the full analysis result into list of element strings"""
    elements = []
    lines = analysis_result.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        # Look for lines that match the element format
        if '|' in line and line.count('|') >= 6:
            # Skip headers
            if not 'element_id|element_type' in line:
                elements.append(line)
    
    return elements


def run_full_image_test():
    """Main test function for full image analysis"""
    print("🖼️  Full Image Analysis Test (Pixtral 12B)")
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
    
    # Analyze the full image
    analysis_result = analyze_full_image(jpeg_bytes)
    
    if not analysis_result['success']:
        print(f"❌ Analysis failed: {analysis_result.get('error', 'Unknown error')}")
        return
    
    # Parse elements
    elements = parse_full_results(analysis_result['content'])
    
    # Results
    print(f"\n📋 RESULTS")
    print("=" * 30)
    print(f"Analysis time: {analysis_result['time']:.1f} seconds")
    print(f"Elements found: {len(elements)}")
    print(f"Image area: {width}x{height} = {width*height:,} pixels")
    if analysis_result['time'] > 0:
        print(f"Elements per second: {len(elements) / analysis_result['time']:.1f}")
    
    if len(elements) > 0:
        print(f"\n🔍 Detected elements (first 20):")
        for i, element in enumerate(elements[:20], 1):
            print(f"  {i:2d}. {element}")
        
        if len(elements) > 20:
            print(f"  ... and {len(elements) - 20} more elements")
    else:
        print("\n📄 Full response (first 800 chars):")
        content = analysis_result['content']
        print(content[:800] + "..." if len(content) > 800 else content)
    
    # Save results
    results = {
        "approach": "full_image_analysis",
        "model": "mistralai/pixtral-12b",
        "provider": "mistral", 
        "analysis_time": analysis_result['time'],
        "image_size": {"width": width, "height": height},
        "elements_found": len(elements),
        "elements": elements,
        "raw_response": analysis_result['content']
    }
    
    results_path = "/tmp/full_image_pixtral_results.json"
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
    
    run_full_image_test()