"""
Ghost Vision Grid System - Coordinate Translation

This module provides a grid-based coordinate system for translating
natural language element descriptions to screen coordinates.

Features:
    - Grid overlay for coarse positioning
    - Text-to-coordinate mapping
    - Intelligent element name parsing
    - Click target calculation
"""

import logging
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

logger = logging.getLogger("Grid")


@dataclass
class GridCell:
    """Represents a cell in the screen grid."""
    row: int
    col: int
    x_center: int
    y_center: int
    x_min: int
    x_max: int
    y_min: int
    y_max: int


class ScreenGrid:
    """
    Grid-based coordinate system for UI element positioning.
    
    Divides the screen into a grid (default 10x10) for coarse positioning
    when VLM-based detection is not precise enough.
    
    Example grid references:
    - "top left" -> (0, 0) cell
    - "center" -> (5, 5) cell  
    - "bottom right" -> (9, 9) cell
    - "A1" -> (0, 0) cell (spreadsheet style)
    """
    
    # Natural language position mappings
    POSITION_ALIASES = {
        # Vertical
        "top": (0, 1),
        "upper": (0, 2),
        "middle": (4, 6),
        "center": (4, 6),
        "lower": (7, 9),
        "bottom": (8, 10),
        
        # Horizontal  
        "left": (0, 2),
        "center_h": (4, 6),
        "right": (7, 10),
    }
    
    COMMON_LOCATIONS = {
        "top left": (1, 1),
        "top center": (1, 5),
        "top right": (1, 8),
        "center left": (5, 1),
        "center": (5, 5),
        "center right": (5, 8),
        "bottom left": (8, 1),
        "bottom center": (8, 5),
        "bottom right": (8, 8),
        
        # Common UI element locations
        "address bar": (0, 5),
        "search bar": (0, 5),
        "taskbar": (9, 5),
        "start menu": (9, 0),
        "system tray": (9, 9),
        "close button": (0, 9),
        "minimize button": (0, 8),
        "maximize button": (0, 8),
    }
    
    def __init__(self, screen_width: int, screen_height: int, grid_size: int = 10):
        """
        Initialize the grid system.
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            grid_size: Number of rows/columns (default 10x10)
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.grid_size = grid_size
        
        self.cell_width = screen_width // grid_size
        self.cell_height = screen_height // grid_size
        
        # Build grid
        self.cells: Dict[Tuple[int, int], GridCell] = {}
        for row in range(grid_size):
            for col in range(grid_size):
                x_min = col * self.cell_width
                x_max = x_min + self.cell_width
                y_min = row * self.cell_height
                y_max = y_min + self.cell_height
                
                self.cells[(row, col)] = GridCell(
                    row=row,
                    col=col,
                    x_center=(x_min + x_max) // 2,
                    y_center=(y_min + y_max) // 2,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max
                )
    
    def get_cell_coords(self, row: int, col: int) -> Tuple[int, int]:
        """
        Get the center coordinates of a grid cell.
        
        Args:
            row: Row index (0-based from top)
            col: Column index (0-based from left)
            
        Returns:
            (x, y) pixel coordinates of cell center
        """
        row = max(0, min(row, self.grid_size - 1))
        col = max(0, min(col, self.grid_size - 1))
        
        cell = self.cells.get((row, col))
        if cell:
            return cell.x_center, cell.y_center
        
        # Fallback calculation
        x = col * self.cell_width + self.cell_width // 2
        y = row * self.cell_height + self.cell_height // 2
        return x, y
    
    def parse_location(self, location_text: str) -> Optional[Tuple[int, int]]:
        """
        Parse natural language location to coordinates.
        
        Args:
            location_text: Description like "top right", "center", "A1"
            
        Returns:
            (x, y) pixel coordinates or None if not parsed
        """
        location_lower = location_text.lower().strip()
        
        # Check common locations first
        if location_lower in self.COMMON_LOCATIONS:
            row, col = self.COMMON_LOCATIONS[location_lower]
            return self.get_cell_coords(row, col)
        
        # Check for spreadsheet-style references (A1, B5, etc.)
        if len(location_lower) >= 2 and location_lower[0].isalpha() and location_lower[1:].isdigit():
            col = ord(location_lower[0]) - ord('a')
            row = int(location_lower[1:]) - 1
            if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
                return self.get_cell_coords(row, col)
        
        # Parse compound locations like "top right"
        words = location_lower.split()
        if len(words) == 2:
            v_word, h_word = words
            
            # Determine row range
            row_range = self.POSITION_ALIASES.get(v_word, (4, 6))
            col_range = self.POSITION_ALIASES.get(h_word, (4, 6))
            
            row = (row_range[0] + row_range[1]) // 2
            col = (col_range[0] + col_range[1]) // 2
            
            return self.get_cell_coords(row, col)
        
        return None
    
    def coords_to_cell(self, x: int, y: int) -> Tuple[int, int]:
        """
        Convert pixel coordinates to grid cell.
        
        Args:
            x: X coordinate in pixels
            y: Y coordinate in pixels
            
        Returns:
            (row, col) grid cell indices
        """
        col = min(x // self.cell_width, self.grid_size - 1)
        row = min(y // self.cell_height, self.grid_size - 1)
        return row, col
    
    def describe_location(self, x: int, y: int) -> str:
        """
        Get a natural language description of a location.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Description like "top left", "center right"
        """
        row, col = self.coords_to_cell(x, y)
        
        # Vertical position
        if row < 3:
            v_pos = "top"
        elif row < 7:
            v_pos = "center"
        else:
            v_pos = "bottom"
        
        # Horizontal position
        if col < 3:
            h_pos = "left"
        elif col < 7:
            h_pos = "center"
        else:
            h_pos = "right"
        
        if v_pos == "center" and h_pos == "center":
            return "center"
        
        return f"{v_pos} {h_pos}"


class ElementCoordinateResolver:
    """
    Resolves UI element descriptions to screen coordinates.
    
    Combines VLM-based detection with grid-based fallbacks.
    """
    
    def __init__(self, perceptor, grid: Optional[ScreenGrid] = None):
        """
        Initialize resolver.
        
        Args:
            perceptor: ScreenPerceptor instance for VLM-based detection
            grid: Optional ScreenGrid for fallback coordinate mapping
        """
        self.perceptor = perceptor
        self.grid = grid
        
    def resolve(self, element_description: str) -> Optional[Tuple[int, int]]:
        """
        Resolve element description to coordinates.
        
        Uses VLM first, falls back to grid-based mapping.
        
        Args:
            element_description: What to find (e.g., "Submit button", "search box")
            
        Returns:
            (x, y) coordinates or None if not found
        """
        # Try VLM-based detection first
        if self.perceptor and self.perceptor.vlm:
            element = self.perceptor.find_element(element_description)
            if element:
                return element.x, element.y
        
        # Fall back to grid-based location parsing
        if self.grid:
            coords = self.grid.parse_location(element_description)
            if coords:
                logger.info(f"Using grid fallback for '{element_description}': {coords}")
                return coords
        
        logger.warning(f"Could not resolve element: {element_description}")
        return None
    
    def resolve_or_default(self, element_description: str, default: Tuple[int, int] = (500, 500)) -> Tuple[int, int]:
        """
        Resolve element description to coordinates with fallback default.
        
        Args:
            element_description: What to find
            default: Default coordinates if not found
            
        Returns:
            (x, y) coordinates
        """
        result = self.resolve(element_description)
        return result if result else default
