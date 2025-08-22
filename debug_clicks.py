#!/usr/bin/env python3
"""Debug script to visualize click points on screenshots"""

import sys
import tempfile
from PIL import Image, ImageDraw
from pathlib import Path

def calculate_click_point(x: int, y: int, w: int, h: int, element_type: str) -> tuple:
    """Calculate optimal click point for UI elements"""
    W, H = 800, 600  # Screen dimensions
    
    # Inclusive-style center removes +0.5px bias on even sizes
    cx = x + (w - 1) // 2
    cy = y + (h - 1) // 2
    
    # For title-bar controls, shift left to avoid right padding/border
    is_title_control = element_type in {'close_button', 'maximize_button', 'minimize_button'}
    is_top_right_small = (x > W * 0.65 and y < H * 0.20 and w <= 24 and h <= 24)
    
    if is_title_control or is_top_right_small:
        cx = max(x + 2, cx - 3)  # 3px left bias; keep within box
        cy = max(y + 2, min(y + h - 3, cy))
    
    # Keep away from screen edges by a pixel or two
    cx = min(max(1, cx), W - 2)
    cy = min(max(1, cy), H - 2)
    
    return cx, cy

def create_debug_image():
    """Create debug image with test click points"""
    from qmp_client import QMPClient
    
    # Take screenshot
    with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as tmp:
        screenshot_path = tmp.name
    
    try:
        client = QMPClient("127.0.0.1", 4444)
        client.connect()
        success = client.screendump(screenshot_path)
        client.disconnect()
        
        if not success:
            print("Screenshot failed")
            return
            
        # Open screenshot
        img = Image.open(screenshot_path)
        print(f"Debug image dimensions: {img.size}")
        print(f"Debug image mode: {img.mode}")
        draw = ImageDraw.Draw(img)
        
        # Test cases based on recent detection
        test_cases = [
            # close_button:X@770,30,20,20
            (770, 30, 20, 20, 'close_button', 'X'),
            # minimize_button:-@745,30,20,20  
            (745, 30, 20, 20, 'minimize_button', '-'),
            # maximize_button:□@720,30,20,20
            (720, 30, 20, 20, 'maximize_button', '□'),
        ]
        
        for x, y, w, h, element_type, text in test_cases:
            # Calculate click point
            click_x, click_y = calculate_click_point(x, y, w, h, element_type)
            
            # Draw bounding box
            draw.rectangle([x, y, x+w, y+h], outline='yellow', width=1)
            
            # Draw original center
            orig_cx = x + w // 2
            orig_cy = y + h // 2
            draw.ellipse([orig_cx-2, orig_cy-2, orig_cx+2, orig_cy+2], fill='blue')
            
            # Draw our calculated click point
            color = 'red' if 'close' in element_type else 'green'
            size = 3
            draw.line([(click_x-size, click_y), (click_x+size, click_y)], fill=color, width=2)
            draw.line([(click_x, click_y-size), (click_x, click_y+size)], fill=color, width=2)
            draw.ellipse([click_x-1, click_y-1, click_x+1, click_y+1], fill=color)
            
            print(f"{element_type}:{text}@{x},{y},{w},{h}")
            print(f"  Original center: ({orig_cx}, {orig_cy})")
            print(f"  Our click point: ({click_x}, {click_y})")
            print(f"  Adjustment: ({click_x-orig_cx}, {click_y-orig_cy})")
            print()
        
        # Save debug image
        debug_path = "/home/jacob/partition/qemu-tools/debug_clicks.png"
        img.save(debug_path)
        print(f"Debug image saved: {debug_path}")
        
    finally:
        Path(screenshot_path).unlink(missing_ok=True)

if __name__ == "__main__":
    create_debug_image()