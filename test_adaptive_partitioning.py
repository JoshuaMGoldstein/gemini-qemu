#!/usr/bin/env python3
"""
Test script for adaptive partitioning system

This script tests the adaptive partitioning approach on the current Puppy desktop
without modifying the existing VNC tools system.
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


def create_partition_image(source_image_path: str, bounds: BoundingBox, 
                          output_path: str) -> bool:
    """Extract a partition from the source image"""
    try:
        img = Image.open(source_image_path)
        # Crop to partition bounds
        cropped = img.crop((bounds.x1, bounds.y1, bounds.x2, bounds.y2))
        cropped.save(output_path)
        return True
    except Exception as e:
        print(f"Failed to create partition image: {e}")
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
taskbar|Bottom taskbar with multiple icons|0,770,1280,800|8

IMPORTANT: 
- Only identify areas with 5+ clustered elements as high-density
- Provide exact pixel coordinates for bounding boxes
- If no high-density areas exist, just list: HIGH_DENSITY_AREAS: none"""

    try:
        response = call_chat_vision(
            model="qwen/qwen-vl-plus",
            system_prompt=system_prompt,
            user_text=user_prompt,
            image_bytes=image_bytes,
            temperature=0.1,
            timeout=30,
            extra={"max_tokens": 1000}  # Limited tokens for density analysis
        )
        
        # Extract content from response
        if 'choices' in response and len(response['choices']) > 0:
            return response['choices'][0]['message']['content']
        else:
            return "Error: No content in response"
            
    except Exception as e:
        return f"Error: {str(e)}"


def analyze_partition_region(image_bytes: bytes, partition_description: str, bounds: tuple) -> str:
    """
    Detailed analysis of a specific region within the full screenshot
    """
    x1, y1, x2, y2 = bounds
    system_prompt = f"""You are a UI element detector. Analyze this FULL screenshot but ONLY examine the region bounded by coordinates ({x1},{y1}) to ({x2},{y2}). 
    
CRITICAL: Only report elements that are actually visible within that specific rectangular region. Ignore everything outside those bounds."""
    
    user_prompt = f"""Analyze this FULL screenshot but ONLY look at the rectangular region from ({x1},{y1}) to ({x2},{y2}).

Region description: {partition_description}

Report ALL visible UI elements within ONLY that specific region.

Format each element as:
id|type|text|center_x,center_y|x1,y1,x2,y2|clickable|enabled|state

Where:
- center_x,center_y = coordinates to click (center of element)
- x1,y1,x2,y2 = bounding box of element
- clickable = 1 if clickable, 0 if not
- enabled = 1 if enabled, 0 if disabled
- state = n (normal), h (highlighted), d (disabled)

Example:
btn1|button|Language/Country|215,310|170,260,275,315|1|1|n
quit|button|Quit|590,585|570,565,610,605|1|1|n

Be extremely precise with coordinates. This partition will be combined with others."""

    try:
        response = call_chat_vision(
            model="qwen/qwen-vl-plus",
            system_prompt=system_prompt,
            user_text=user_prompt,
            image_bytes=image_bytes,
            temperature=0.1,
            timeout=30,
            extra={"max_tokens": 2000}  # More tokens for detailed analysis
        )
        
        if 'choices' in response and len(response['choices']) > 0:
            return response['choices'][0]['message']['content']
        else:
            return "Error: No content in response"
            
    except Exception as e:
        return f"Error: {str(e)}"


def parse_element_analysis(analysis_result: str) -> list:
    """Parse element analysis result into list of element strings"""
    elements = []
    lines = analysis_result.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        # Look for lines that match the element format: id|type|text|coords|bounds|...
        if '|' in line and line.count('|') >= 6:
            elements.append(line)
    
    return elements


def transform_partition_results(analysis_result: str, partition_bounds: BoundingBox, 
                              partitioner: AdaptivePartitioner) -> list:
    """Transform partition-local coordinates to global coordinates"""
    global_elements = []
    
    for line in analysis_result.strip().split('\n'):
        line = line.strip()
        if '|' in line and line.count('|') >= 6:
            parts = line.split('|')
            if len(parts) >= 7:
                try:
                    # Parse coordinates
                    center_coords = parts[3].split(',')
                    bbox_coords = parts[4].split(',')
                    
                    if len(center_coords) == 2 and len(bbox_coords) == 4:
                        # Transform center coordinates
                        local_center = (int(center_coords[0]), int(center_coords[1]))
                        global_center = partitioner.transform_coordinates(local_center, partition_bounds)
                        
                        # Transform bounding box coordinates
                        local_bbox = (int(bbox_coords[0]), int(bbox_coords[1]))
                        global_bbox_start = partitioner.transform_coordinates(local_bbox, partition_bounds)
                        
                        local_bbox_end = (int(bbox_coords[2]), int(bbox_coords[3]))
                        global_bbox_end = partitioner.transform_coordinates(local_bbox_end, partition_bounds)
                        
                        # Reconstruct line with global coordinates
                        global_line = (
                            f"{parts[0]}|{parts[1]}|{parts[2]}|"
                            f"{global_center[0]},{global_center[1]}|"
                            f"{global_bbox_start[0]},{global_bbox_start[1]},"
                            f"{global_bbox_end[0]},{global_bbox_end[1]}|"
                            f"{parts[5]}|{parts[6]}"
                        )
                        if len(parts) > 7:
                            global_line += "|" + "|".join(parts[7:])
                        
                        global_elements.append(global_line)
                        
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line '{line}': {e}")
                    continue
    
    return global_elements


def run_adaptive_partitioning_test():
    """Main test function"""
    print("🔬 Adaptive Partitioning Test")
    print("=" * 50)
    
    # Create partitioner
    partitioner = AdaptivePartitioner(1280, 800)
    
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
    
    # Step 2: Analyze density
    print("\n🧠 Analyzing element density...")
    density_result = analyze_density(jpeg_bytes)
    print("Density Analysis Result:")
    print("-" * 30)
    print(density_result)
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
            print(f"     Estimated elements: {area.estimated_elements}")
    
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
        print(f"     Size: {partition.bounds.width}x{partition.bounds.height}")
    
    # Step 4: Analyze each partition in parallel with rate limiting
    print(f"\n🔍 Analyzing partitions in parallel...")
    import concurrent.futures
    import threading
    import time
    
    all_elements = []
    
    def analyze_single_partition(i, partition):
        """Analyze a single partition and return results"""
        print(f"\nStarting partition {i+1}: {partition.description}")
        
        # Use full image with region bounds instead of cropping
        bounds = (partition.bounds.x1, partition.bounds.y1, partition.bounds.x2, partition.bounds.y2)
        analysis_result = analyze_partition_region(jpeg_bytes, partition.description, bounds)
        print(f"Partition {i+1} raw result (first 100 chars): {analysis_result[:100]}...")
        
        # Parse elements directly (coordinates should already be in global space)
        elements = parse_element_analysis(analysis_result)
        
        print(f"✅ Partition {i+1} found {len(elements)} elements")
        return elements
    
    # Launch all partition analyses with staggered start times
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
        futures = []
        
        for i, partition in enumerate(partitions):
            # Submit each task with a small delay to avoid rate limiting
            future = executor.submit(analyze_single_partition, i, partition)
            futures.append(future)
            time.sleep(0.5)  # 0.5 second delay between API calls
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                elements = future.result()
                all_elements.extend(elements)
            except Exception as e:
                print(f"❌ Partition analysis failed: {e}")
    
    # Step 5: Summary
    print(f"\n📋 FINAL RESULTS")
    print("=" * 50)
    print(f"Total partitions analyzed: {len(partitions)}")
    print(f"Total elements detected: {len(all_elements)}")
    print(f"Expected elements: {total_elements}")
    print(f"Coverage efficiency: {len(all_elements)}/{total_elements} = {len(all_elements)/max(1,total_elements)*100:.1f}%")
    
    print(f"\n🎯 All detected elements:")
    for i, element in enumerate(all_elements, 1):
        print(f"  {i:2d}. {element}")
    
    # Save results
    results = {
        "total_partitions": len(partitions),
        "total_elements_detected": len(all_elements),
        "expected_elements": total_elements,
        "partitions": [
            {
                "description": p.description,
                "bounds": [p.bounds.x1, p.bounds.y1, p.bounds.x2, p.bounds.y2],
                "type": p.partition_type
            } for p in partitions
        ],
        "elements": all_elements
    }
    
    results_path = "/tmp/adaptive_partitioning_test_results.json"
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
    
    run_adaptive_partitioning_test()