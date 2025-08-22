#!/usr/bin/env python3
"""
Test script for single-call adaptive partitioning using Qwen 2.5 VL 32B's large context
This approach sends all partition regions in a single API call for faster analysis
"""

import sys
import json
import time
from pathlib import Path
from PIL import Image
import tempfile

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from adaptive_partitioning import (
    AdaptivePartitioner, HighDensityArea, BoundingBox, 
    parse_density_analysis
)
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


def analyze_density(image_bytes: bytes) -> str:
    """
    Pass 1: Analyze element density and identify high-density areas
    """
    system_prompt = """You are a UI density analyzer. Your job is to count elements and identify dense areas that need focused analysis."""
    
    user_prompt = """Analyze this screenshot and identify:

1. TOTAL number of distinct clickable UI elements you can see
2. Areas with HIGH element density (many elements clustered together)

For high-density areas, provide precise bounding box coordinates.

Format your response EXACTLY as:
TOTAL_ELEMENTS: [number]
HIGH_DENSITY_AREAS:
area_id|description|x1,y1,x2,y2|estimated_elements

Example:
TOTAL_ELEMENTS: 45
HIGH_DENSITY_AREAS:
setup_dialog|Dialog with 9 wizard buttons|130,140,620,600|12
taskbar|Bottom taskbar with multiple icons|0,570,800,600|8

IMPORTANT: 
- Only identify areas with 5+ clustered elements as high-density
- Provide exact pixel coordinates for bounding boxes
- If no high-density areas exist, just list: HIGH_DENSITY_AREAS: none"""

    try:
        response = call_chat_vision(
            model="qwen/qwen2.5-vl-32b-instruct",  # Using Qwen 2.5 VL 32B
            system_prompt=system_prompt,
            user_text=user_prompt,
            image_bytes=image_bytes,
            temperature=0.1,
            timeout=30,
            extra={
                "max_tokens": 1000,
                "provider": {"order": ["fireworks"]}  # Force Fireworks provider
            }
        )
        
        # Extract content from response
        if 'choices' in response and len(response['choices']) > 0:
            return response['choices'][0]['message']['content']
        else:
            return "Error: No content in response"
            
    except Exception as e:
        return f"Error: {str(e)}"


def analyze_all_partitions_single_call(image_bytes: bytes, partitions: list) -> dict:
    """
    Analyze all partitions in a single API call using Qwen 2.5 VL's large context
    """
    system_prompt = """You are a UI element detector. Analyze this FULL screenshot and examine EACH specified region.
    
For each region, report ALL visible UI elements within ONLY that specific rectangular area.
Format each element as: id|type|text|center_x,center_y|x1,y1,x2,y2|clickable|enabled|state

Where:
- center_x,center_y = coordinates to click (center of element)
- x1,y1,x2,y2 = bounding box of element
- clickable = 1 if clickable, 0 if not
- enabled = 1 if enabled, 0 if disabled
- state = n (normal), h (highlighted), d (disabled)"""
    
    # Build the prompt with all regions
    regions_prompt = "Analyze this screenshot region by region:\n\n"
    
    for i, partition in enumerate(partitions, 1):
        bounds = partition.bounds
        regions_prompt += f"""REGION {i}: {partition.description}
Boundaries: ({bounds.x1},{bounds.y1}) to ({bounds.x2},{bounds.y2})
Examine ONLY this rectangle and list ALL UI elements within it.

"""
    
    regions_prompt += """For each region, provide output in this format:
REGION [number] ELEMENTS:
element1|type|text|center_x,center_y|x1,y1,x2,y2|clickable|enabled|state
element2|type|text|center_x,center_y|x1,y1,x2,y2|clickable|enabled|state
...

Be extremely precise with coordinates. This will be used for UI automation."""

    try:
        print(f"📡 Sending single API call with {len(partitions)} regions...")
        start_time = time.time()
        
        response = call_chat_vision(
            model="qwen/qwen2.5-vl-32b-instruct",  # Using Qwen 2.5 VL 32B with Fireworks
            system_prompt=system_prompt,
            user_text=regions_prompt,
            image_bytes=image_bytes,
            temperature=0.1,
            timeout=60,  # Longer timeout for single large call
            extra={
                "max_tokens": 16384,  # Large output for all regions
                "provider": {"order": ["fireworks"]}  # Force Fireworks provider
            }
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Single API call completed in {elapsed:.1f} seconds")
        
        # Extract content from response
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            return {"success": True, "content": content, "time": elapsed}
        else:
            return {"success": False, "error": "No content in response"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


def parse_single_call_results(analysis_result: str) -> list:
    """Parse the single-call analysis result into list of element strings"""
    all_elements = []
    lines = analysis_result.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        # Look for lines that match the element format
        if '|' in line and line.count('|') >= 6:
            # Skip headers and region markers
            if not line.startswith('REGION') and not 'ELEMENTS:' in line:
                all_elements.append(line)
    
    return all_elements


def run_single_call_test():
    """Main test function for single-call approach"""
    print("🚀 Single-Call Adaptive Partitioning Test (Qwen 2.5 VL 32B)")
    print("=" * 60)
    
    # Create partitioner
    partitioner = AdaptivePartitioner(800, 600)
    
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
    
    # Step 2: Analyze density (still separate for planning)
    print("\n🧠 Analyzing element density...")
    start_density = time.time()
    density_result = analyze_density(jpeg_bytes)
    density_time = time.time() - start_density
    print(f"Density analysis completed in {density_time:.1f} seconds")
    print("Density Analysis Result:")
    print("-" * 30)
    print(density_result[:300] + "..." if len(density_result) > 300 else density_result)
    print("-" * 30)
    
    # Parse density analysis
    try:
        total_elements, high_density_areas = parse_density_analysis(density_result)
        print(f"\n📊 Parsed Results:")
        print(f"   Total elements detected: {total_elements}")
        print(f"   High-density areas: {len(high_density_areas)}")
        
        for area in high_density_areas:
            print(f"   - {area.area_id}: {area.description}")
            print(f"     Bounds: {area.bounds}")
    
    except Exception as e:
        print(f"❌ Failed to parse density analysis: {e}")
        return
    
    # Step 3: Generate partitions
    print(f"\n🔧 Generating screenshot partitions...")
    partitions = partitioner.partition_screen(high_density_areas)
    
    print(f"Generated {len(partitions)} partitions:")
    for i, partition in enumerate(partitions):
        print(f"  {i+1}. {partition.partition_type}: {partition.description}")
        print(f"     Bounds: {partition.bounds}")
    
    # Step 4: SINGLE API CALL for all partitions
    print(f"\n🔍 Analyzing all {len(partitions)} partitions in a SINGLE API call...")
    
    analysis_result = analyze_all_partitions_single_call(jpeg_bytes, partitions)
    
    if not analysis_result['success']:
        print(f"❌ Analysis failed: {analysis_result.get('error', 'Unknown error')}")
        return
    
    # Parse elements from the single response
    all_elements = parse_single_call_results(analysis_result['content'])
    
    # Step 5: Summary
    print(f"\n📋 FINAL RESULTS")
    print("=" * 60)
    print(f"Total API calls: 2 (density + single partition analysis)")
    print(f"Total time: {density_time + analysis_result['time']:.1f} seconds")
    print(f"  - Density analysis: {density_time:.1f}s")
    print(f"  - All partitions: {analysis_result['time']:.1f}s")
    print(f"Total elements detected: {len(all_elements)}")
    print(f"Expected elements: {total_elements}")
    
    if len(all_elements) > 0:
        print(f"\n🎯 Sample detected elements (first 10):")
        for i, element in enumerate(all_elements[:10], 1):
            print(f"  {i:2d}. {element}")
    
    # Save results
    results = {
        "approach": "single_call",
        "model": "qwen/qwen2.5-vl-32b-instruct",
        "total_api_calls": 2,
        "total_time": density_time + analysis_result['time'],
        "density_time": density_time,
        "partitions_time": analysis_result['time'],
        "total_partitions": len(partitions),
        "total_elements_detected": len(all_elements),
        "expected_elements": total_elements,
        "elements": all_elements
    }
    
    results_path = "/tmp/single_call_partitioning_results.json"
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
    
    run_single_call_test()