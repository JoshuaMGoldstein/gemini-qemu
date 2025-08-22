#!/usr/bin/env python3
"""
Adaptive Image Partitioning for Vision-Language Models

This module implements intelligent screenshot partitioning to handle complex UI
interfaces by breaking them into manageable chunks for vision analysis.

Strategy:
1. Identify high-density areas that need focused analysis
2. Create efficient screenshot partitions that cover 100% of the screen
3. Minimize the number of API calls while maximizing element detection accuracy
"""

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class BoundingBox:
    """Represents a rectangular region with coordinates"""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
        
    @property
    def height(self) -> int:
        return self.y2 - self.y1
        
    @property
    def area(self) -> int:
        return self.width * self.height
        
    def __repr__(self) -> str:
        return f"BoundingBox({self.x1},{self.y1},{self.x2},{self.y2})"


@dataclass
class HighDensityArea:
    """Represents an area identified as having high element density"""
    area_id: str
    description: str
    bounds: BoundingBox
    estimated_elements: int


@dataclass
class ScreenshotPartition:
    """Represents a screenshot region to be analyzed"""
    bounds: BoundingBox
    partition_type: str  # "high_density" or "remainder"
    description: str
    priority: int = 1  # Higher priority = analyze first


class AdaptivePartitioner:
    """Handles intelligent partitioning of screenshots for vision analysis"""
    
    def __init__(self, screen_width: int = 1280, screen_height: int = 800, 
                 min_region_size: int = 50):
        self.screen_bounds = BoundingBox(0, 0, screen_width, screen_height)
        self.min_region_size = min_region_size
        
    def partition_screen(self, high_density_areas: List[HighDensityArea]) -> List[ScreenshotPartition]:
        """
        Create optimal screenshot partitions to cover entire screen
        
        Args:
            high_density_areas: List of areas requiring focused analysis
            
        Returns:
            List of screenshot partitions with complete screen coverage
        """
        partitions = []
        coverage_regions = [self.screen_bounds]
        
        # Process high-density areas first
        for area in high_density_areas:
            # Add dedicated screenshot for high-density area
            partitions.append(ScreenshotPartition(
                bounds=area.bounds,
                partition_type="high_density",
                description=f"High-density: {area.description}",
                priority=2  # Higher priority
            ))
            
            # Remove this area from coverage regions
            coverage_regions = self._subtract_box_from_regions(coverage_regions, area.bounds)
        
        # Merge remaining regions to minimize screenshots
        merged_regions = self._merge_adjacent_regions(coverage_regions)
        
        # Add remainder partitions
        for i, region in enumerate(merged_regions):
            if region.area >= self.min_region_size * self.min_region_size:
                partitions.append(ScreenshotPartition(
                    bounds=region,
                    partition_type="remainder",
                    description=f"Remainder region {i+1}",
                    priority=1
                ))
        
        return sorted(partitions, key=lambda p: p.priority, reverse=True)
    
    def _subtract_box_from_regions(self, regions: List[BoundingBox], 
                                 box_to_subtract: BoundingBox) -> List[BoundingBox]:
        """
        Remove a box from all regions, creating up to 4 remainder rectangles per region
        """
        new_regions = []
        
        for region in regions:
            # Check if box overlaps with region
            if not self._boxes_overlap(region, box_to_subtract):
                new_regions.append(region)
                continue
                
            # Create up to 4 remainder rectangles around the subtracted box
            remainder_rects = self._create_remainder_rectangles(region, box_to_subtract)
            new_regions.extend(remainder_rects)
        
        return new_regions
    
    def _create_remainder_rectangles(self, region: BoundingBox, 
                                   subtracted_box: BoundingBox) -> List[BoundingBox]:
        """
        Create remainder rectangles around a subtracted box within a region
        
        Creates up to 4 rectangles:
        - Top strip (above subtracted box)
        - Bottom strip (below subtracted box)  
        - Left strip (left of subtracted box)
        - Right strip (right of subtracted box)
        """
        remainders = []
        
        # Clamp subtracted box to region bounds
        sub_x1 = max(region.x1, subtracted_box.x1)
        sub_y1 = max(region.y1, subtracted_box.y1)
        sub_x2 = min(region.x2, subtracted_box.x2)
        sub_y2 = min(region.y2, subtracted_box.y2)
        
        # Top strip (above subtracted box)
        if region.y1 < sub_y1:
            remainders.append(BoundingBox(region.x1, region.y1, region.x2, sub_y1))
        
        # Bottom strip (below subtracted box)
        if sub_y2 < region.y2:
            remainders.append(BoundingBox(region.x1, sub_y2, region.x2, region.y2))
        
        # Left strip (left of subtracted box)
        if region.x1 < sub_x1:
            remainders.append(BoundingBox(region.x1, sub_y1, sub_x1, sub_y2))
        
        # Right strip (right of subtracted box)
        if sub_x2 < region.x2:
            remainders.append(BoundingBox(sub_x2, sub_y1, region.x2, sub_y2))
        
        return remainders
    
    def _merge_adjacent_regions(self, regions: List[BoundingBox]) -> List[BoundingBox]:
        """
        Merge adjacent regions to minimize the number of screenshots needed
        
        This is a simplified merge - in practice, more sophisticated algorithms
        could be used for optimal merging.
        """
        if not regions:
            return []
        
        # Simple merge: combine regions that can form larger rectangles
        merged = []
        remaining = regions.copy()
        
        while remaining:
            current = remaining.pop(0)
            
            # Try to merge with other regions
            merged_any = True
            while merged_any:
                merged_any = False
                for i, other in enumerate(remaining):
                    merged_rect = self._try_merge_rectangles(current, other)
                    if merged_rect:
                        current = merged_rect
                        remaining.pop(i)
                        merged_any = True
                        break
            
            merged.append(current)
        
        return merged
    
    def _try_merge_rectangles(self, rect1: BoundingBox, rect2: BoundingBox) -> BoundingBox:
        """
        Try to merge two rectangles if they form a valid larger rectangle
        Returns None if merge is not possible
        """
        # Check if rectangles can be merged horizontally
        if (rect1.y1 == rect2.y1 and rect1.y2 == rect2.y2 and 
            (rect1.x2 == rect2.x1 or rect2.x2 == rect1.x1)):
            return BoundingBox(
                min(rect1.x1, rect2.x1), rect1.y1,
                max(rect1.x2, rect2.x2), rect1.y2
            )
        
        # Check if rectangles can be merged vertically
        if (rect1.x1 == rect2.x1 and rect1.x2 == rect2.x2 and 
            (rect1.y2 == rect2.y1 or rect2.y2 == rect1.y1)):
            return BoundingBox(
                rect1.x1, min(rect1.y1, rect2.y1),
                rect1.x2, max(rect1.y2, rect2.y2)
            )
        
        return None
    
    def _boxes_overlap(self, box1: BoundingBox, box2: BoundingBox) -> bool:
        """Check if two bounding boxes overlap"""
        return not (box1.x2 <= box2.x1 or box2.x2 <= box1.x1 or 
                   box1.y2 <= box2.y1 or box2.y2 <= box1.y1)
    
    def transform_coordinates(self, local_coords: Tuple[int, int], 
                            partition_bounds: BoundingBox) -> Tuple[int, int]:
        """
        Transform coordinates from partition-local space to global screen space
        
        Args:
            local_coords: (x, y) coordinates within the partition
            partition_bounds: The bounds of the partition
            
        Returns:
            (x, y) coordinates in global screen space
        """
        local_x, local_y = local_coords
        global_x = partition_bounds.x1 + local_x
        global_y = partition_bounds.y1 + local_y
        return (global_x, global_y)


def parse_density_analysis(vision_response: str) -> Tuple[int, List[HighDensityArea]]:
    """
    Parse the vision model's density analysis response
    
    Expected format:
    TOTAL_ELEMENTS: 45
    HIGH_DENSITY_AREAS:
    setup_dialog|Puppy Setup wizard buttons|130,140,620,600|12
    taskbar|Bottom taskbar with icons|0,770,1280,800|8
    
    Returns:
        (total_elements, list_of_high_density_areas)
    """
    lines = vision_response.strip().split('\n')
    total_elements = 0
    high_density_areas = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('TOTAL_ELEMENTS:'):
            total_elements = int(line.split(':')[1].strip())
        elif '|' in line and not line.startswith('HIGH_DENSITY_AREAS'):
            # Parse high density area line
            parts = line.split('|')
            if len(parts) >= 4:
                area_id = parts[0].strip()
                description = parts[1].strip()
                coords = parts[2].strip().split(',')
                estimated_elements = int(parts[3].strip())
                
                if len(coords) == 4:
                    bounds = BoundingBox(
                        int(coords[0]), int(coords[1]),
                        int(coords[2]), int(coords[3])
                    )
                    high_density_areas.append(HighDensityArea(
                        area_id=area_id,
                        description=description,
                        bounds=bounds,
                        estimated_elements=estimated_elements
                    ))
    
    return total_elements, high_density_areas


# Example usage and testing
if __name__ == "__main__":
    # Test with Puppy Linux desktop scenario
    partitioner = AdaptivePartitioner(1280, 800)
    
    # Simulated high-density areas from vision analysis
    high_density_areas = [
        HighDensityArea("setup_dialog", "Puppy Setup wizard buttons", 
                       BoundingBox(130, 140, 620, 600), 12),
        HighDensityArea("taskbar", "Bottom taskbar with icons",
                       BoundingBox(0, 770, 1280, 800), 8)
    ]
    
    # Generate partitions
    partitions = partitioner.partition_screen(high_density_areas)
    
    print("Generated Screenshot Partitions:")
    print("=" * 50)
    for i, partition in enumerate(partitions):
        print(f"Partition {i+1}: {partition.partition_type}")
        print(f"  Description: {partition.description}")
        print(f"  Bounds: {partition.bounds}")
        print(f"  Size: {partition.bounds.width}x{partition.bounds.height}")
        print(f"  Priority: {partition.priority}")
        print()
    
    # Test coordinate transformation
    print("Coordinate Transformation Example:")
    print("=" * 50)
    setup_partition = partitions[0]  # Should be high-density setup dialog
    local_coords = (50, 100)  # Coordinates within the partition
    global_coords = partitioner.transform_coordinates(local_coords, setup_partition.bounds)
    print(f"Local coords {local_coords} in partition {setup_partition.bounds}")
    print(f"Transform to global coords: {global_coords}")