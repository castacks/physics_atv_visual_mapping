"""
Sparse Global Costmap Implementation using PyTorch Sparse Tensors

This module provides a memory-efficient global costmap that:
1. Uses sparse tensor representation for memory efficiency
2. Does NOT move with the robot (robot moves inside the map)
3. Can grow unbounded (in practice, up to configured max size)
4. Designed for the global planner (ARA*) while local planner uses local costmap

The key design principle is MASKED OVERWRITE: when local costmap updates come in,
only KNOWN/OBSERVED cells in the local costmap are written to the global costmap.
This prevents unknown cells (due to occlusions or sensor FOV limits) from clearing
previously mapped obstacles when the robot returns to an area.
"""

import torch
from typing import Tuple, Dict


class SparseGlobalCostmap:
    """
    Sparse global costmap using PyTorch sparse tensors.
    
    The costmap uses a fixed world-frame origin and does not move with the robot.
    Data is stored as a sparse tensor for memory efficiency.
    
    IMPORTANT: This uses MASKED OVERWRITE semantics. When a local costmap update
    comes in, only KNOWN/OBSERVED cells are written to the global costmap. This:
    - Prevents unknown cells from clearing previously mapped obstacles
    - Allows dynamic obstacle clearing when the sensor actually observes free space
    - The global map accumulates history in areas never observed by current local map
    
    Coordinate System:
    - Origin is at (origin_x, origin_y) in world coordinates (meters)
    - Grid indices (i, j) correspond to world position:
        world_x = origin_x + i * resolution
        world_y = origin_y + j * resolution
    """
    
    def __init__(
        self,
        origin: Tuple[float, float] = (0.0, 0.0),
        resolution: float = 0.6,  # meters per cell (lower res than local map)
        max_size: Tuple[int, int] = (2000, 2000),  # max grid dimensions
        device: str = 'cuda',
        unknown_cost: float = 0.5,  # cost for unknown cells
    ):
        """
        Initialize sparse global costmap.
        
        Args:
            origin: World coordinates (x, y) of the bottom-left corner
            resolution: Meters per grid cell
            max_size: Maximum grid dimensions (rows, cols)
            device: PyTorch device ('cuda' or 'cpu')
            unknown_cost: Default cost for cells with no observations
        """
        self.origin = torch.tensor(origin, dtype=torch.float32, device=device)
        self.resolution = resolution
        self.max_size = max_size
        self.device = device
        self.unknown_cost = unknown_cost
        
        # Sparse storage: indices are (row, col) in grid coordinates
        # Values: cost for each observed cell (direct overwrite, no averaging)
        self._indices = torch.empty((2, 0), dtype=torch.long, device=device)
        self._costs = torch.empty(0, dtype=torch.float32, device=device)
        
        # Track bounds of observed area for efficient querying
        self._observed_min = torch.tensor([float('inf'), float('inf')], device=device)
        self._observed_max = torch.tensor([float('-inf'), float('-inf')], device=device)
        
        # Statistics
        self.total_updates = 0
        self.num_cells = 0
        
    @property
    def observed_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the bounds of observed area in grid coordinates."""
        return self._observed_min, self._observed_max
    
    @property
    def observed_world_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the bounds of observed area in world coordinates."""
        min_world = self._observed_min * self.resolution + self.origin
        max_world = self._observed_max * self.resolution + self.origin
        return min_world, max_world
    
    def world_to_grid(self, world_coords: torch.Tensor) -> torch.Tensor:
        """
        Convert world coordinates to grid indices.
        
        Args:
            world_coords: [N, 2] or [2] tensor of (x, y) world coordinates
            
        Returns:
            Grid indices [N, 2] or [2] as long tensor
        """
        grid_coords = (world_coords - self.origin) / self.resolution
        return grid_coords.long()
    
    def grid_to_world(self, grid_coords: torch.Tensor) -> torch.Tensor:
        """
        Convert grid indices to world coordinates (cell center).
        
        Args:
            grid_coords: [N, 2] or [2] tensor of grid indices
            
        Returns:
            World coordinates [N, 2] or [2] as float tensor
        """
        return grid_coords.float() * self.resolution + self.origin + self.resolution / 2
    
    def update_from_local_costmap(
        self,
        local_costs: torch.Tensor,
        local_origin: torch.Tensor,
        local_resolution: float,
        edge_margin: int = 5,
        known_mask: torch.Tensor = None,
    ):
        """
        Update global costmap from a local costmap observation.
        
        MASKED OVERWRITE: Only cells that are KNOWN/OBSERVED in the local costmap
        are written to the global costmap. This prevents unknown/unobserved cells
        (due to occlusions or sensor FOV limits) from clearing previously mapped
        obstacles in the global map.
        
        An edge margin is used to avoid writing edge cells, which often have
        boundary artifacts from terrain estimation or incomplete sensor coverage.
        
        Args:
            local_costs: [H, W] local costmap (0=free, 1=obstacle)
            local_origin: [2] origin of local map in world coordinates
            local_resolution: resolution of local map in meters
            edge_margin: Number of cells from the edge of local costmap to skip.
                         This avoids writing edge artifacts to the global map.
            known_mask: [H, W] boolean mask where True = cell is observed/known.
                        If None, all cells are treated as known (legacy behavior).
        """
        if local_costs.numel() == 0:
            return
            
        local_h, local_w = local_costs.shape
        
        # Skip if local map is too small to have an interior after margin
        if local_h <= 2 * edge_margin or local_w <= 2 * edge_margin:
            return
        
        # Create meshgrid of local grid indices
        local_i, local_j = torch.meshgrid(
            torch.arange(edge_margin, local_h - edge_margin, device=self.device),
            torch.arange(edge_margin, local_w - edge_margin, device=self.device),
            indexing='ij'
        )
        
        # Convert local grid indices to world coordinates
        local_world_x = local_origin[0] + local_i.float() * local_resolution + local_resolution / 2
        local_world_y = local_origin[1] + local_j.float() * local_resolution + local_resolution / 2
        local_world = torch.stack([local_world_x, local_world_y], dim=-1)  # [H-2*margin, W-2*margin, 2]
        
        # Get costs from interior region only
        interior_costs = local_costs[edge_margin:local_h - edge_margin, edge_margin:local_w - edge_margin]
        
        # Get known mask for interior region (if provided)
        if known_mask is not None:
            interior_known = known_mask[edge_margin:local_h - edge_margin, edge_margin:local_w - edge_margin]
        else:
            interior_known = None
        
        # Convert to global grid indices
        global_grid = self.world_to_grid(local_world.reshape(-1, 2))  # [N, 2]
        
        # Filter out-of-bounds indices
        valid_mask = (
            (global_grid[:, 0] >= 0) & (global_grid[:, 0] < self.max_size[0]) &
            (global_grid[:, 1] >= 0) & (global_grid[:, 1] < self.max_size[1])
        )
        
        # Also filter by known mask (only update observed cells)
        if interior_known is not None:
            valid_mask = valid_mask & interior_known.reshape(-1)
        
        new_indices = global_grid[valid_mask].T  # [2, N_valid]
        new_costs = interior_costs.reshape(-1)[valid_mask]  # [N_valid]
        
        if new_indices.shape[1] == 0:
            return
        
        # DIRECT OVERWRITE: merge new observations, replacing existing values
        self._direct_overwrite_merge(new_indices, new_costs)
        
        # Update observed bounds
        self._observed_min = torch.minimum(
            self._observed_min, 
            new_indices.min(dim=1).values.float()
        )
        self._observed_max = torch.maximum(
            self._observed_max,
            new_indices.max(dim=1).values.float()
        )
        
        self.total_updates += 1
        
    def _direct_overwrite_merge(
        self,
        new_indices: torch.Tensor,  # [2, N]
        new_costs: torch.Tensor,    # [N]
    ):
        """
        Merge new observations with existing sparse data using DIRECT OVERWRITE.
        
        For cells that exist in both old and new data, the NEW value wins.
        This allows dynamic obstacle clearing from the local map to propagate.
        """
        if new_indices.shape[1] == 0:
            return
            
        # Convert 2D indices to 1D raster indices
        new_raster = new_indices[0] * self.max_size[1] + new_indices[1]
        
        if self._indices.shape[1] == 0:
            # No existing data - just use new
            # Deduplicate new indices (take last value if duplicates)
            unique_raster, inverse = torch.unique(new_raster, return_inverse=True)
            # Use scatter with last-write-wins (reverse order trick)
            unique_costs = torch.zeros(unique_raster.shape[0], device=self.device)
            unique_costs.scatter_(0, inverse, new_costs)
            
            unique_i = unique_raster // self.max_size[1]
            unique_j = unique_raster % self.max_size[1]
            self._indices = torch.stack([unique_i, unique_j], dim=0)
            self._costs = unique_costs
        else:
            # Have existing data - need to merge with overwrite semantics
            existing_raster = self._indices[0] * self.max_size[1] + self._indices[1]
            
            # Find which existing cells are NOT being overwritten
            # Using set difference via searchsorted
            new_raster_sorted, sort_idx = torch.sort(new_raster)
            search_pos = torch.searchsorted(new_raster_sorted, existing_raster)
            search_pos = search_pos.clamp(max=new_raster_sorted.shape[0] - 1)
            is_overwritten = new_raster_sorted[search_pos] == existing_raster
            keep_existing = ~is_overwritten
            
            # Combine: kept existing + all new (deduplicated)
            kept_raster = existing_raster[keep_existing]
            kept_costs = self._costs[keep_existing]
            
            # Deduplicate new indices
            unique_new_raster, inverse = torch.unique(new_raster, return_inverse=True)
            unique_new_costs = torch.zeros(unique_new_raster.shape[0], device=self.device)
            unique_new_costs.scatter_(0, inverse, new_costs)
            
            # Combine all
            all_raster = torch.cat([kept_raster, unique_new_raster])
            all_costs = torch.cat([kept_costs, unique_new_costs])
            
            # Final sort and dedupe (shouldn't have dupes, but safety first)
            final_raster, inverse = torch.unique(all_raster, return_inverse=True)
            final_costs = torch.zeros(final_raster.shape[0], device=self.device)
            final_costs.scatter_(0, inverse, all_costs)
            
            # Convert back to 2D
            final_i = final_raster // self.max_size[1]
            final_j = final_raster % self.max_size[1]
            
            self._indices = torch.stack([final_i, final_j], dim=0)
            self._costs = final_costs
        
        self.num_cells = self._indices.shape[1]
        
    def get_cost_at_world(self, world_coords: torch.Tensor) -> torch.Tensor:
        """
        Get costs at specific world coordinates.
        
        Args:
            world_coords: [N, 2] world coordinates
            
        Returns:
            [N] costs (unknown_cost for unobserved cells)
        """
        grid_coords = self.world_to_grid(world_coords)
        return self.get_cost_at_grid(grid_coords)
    
    def get_cost_at_grid(self, grid_coords: torch.Tensor) -> torch.Tensor:
        """
        Get costs at specific grid indices.
        
        Args:
            grid_coords: [N, 2] grid indices
            
        Returns:
            [N] costs (unknown_cost for unobserved cells)
        """
        if self._indices.shape[1] == 0:
            return torch.full(
                (grid_coords.shape[0],), self.unknown_cost, 
                device=self.device
            )
        
        # Create raster indices for query points
        query_raster = grid_coords[:, 0] * self.max_size[1] + grid_coords[:, 1]
        existing_raster = self._indices[0] * self.max_size[1] + self._indices[1]
        
        # Find matches using searchsorted
        sorted_indices = torch.argsort(existing_raster)
        sorted_raster = existing_raster[sorted_indices]
        
        insert_positions = torch.searchsorted(sorted_raster, query_raster)
        
        # Check if positions actually match
        valid_positions = insert_positions < sorted_raster.shape[0]
        matches = torch.zeros_like(query_raster, dtype=torch.bool)
        matches[valid_positions] = (
            sorted_raster[insert_positions[valid_positions].clamp(max=sorted_raster.shape[0]-1)] 
            == query_raster[valid_positions]
        )
        
        # Return costs directly (no min_observations check - all observed cells are "known")
        costs = torch.full(
            (grid_coords.shape[0],), self.unknown_cost, 
            device=self.device
        )
        
        if matches.any():
            matched_sorted_idx = sorted_indices[insert_positions[matches]]
            costs[matches] = self._costs[matched_sorted_idx]
        
        return costs.clamp(0.0, 1.0)
    
    def to_dense_window(
        self,
        center_world: torch.Tensor,
        window_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract a dense costmap window centered at a world position.
        
        Args:
            center_world: [2] world coordinates of window center
            window_size: (rows, cols) size of output window
            
        Returns:
            (costmap, origin): dense [H, W] costmap and [2] origin in world coords
        """
        center_grid = self.world_to_grid(center_world)
        half_h, half_w = window_size[0] // 2, window_size[1] // 2
        
        # Window bounds in grid coordinates
        start_i = center_grid[0] - half_h
        start_j = center_grid[1] - half_w
        
        # Create grid of indices for the window
        window_i, window_j = torch.meshgrid(
            torch.arange(window_size[0], device=self.device) + start_i,
            torch.arange(window_size[1], device=self.device) + start_j,
            indexing='ij'
        )
        
        grid_coords = torch.stack([
            window_i.flatten(),
            window_j.flatten()
        ], dim=-1)
        
        costs = self.get_cost_at_grid(grid_coords)
        dense_map = costs.reshape(window_size)
        
        # Origin in world coordinates
        origin = self.grid_to_world(torch.tensor([start_i, start_j], device=self.device))
        origin = origin - self.resolution / 2  # bottom-left corner, not center
        
        return dense_map, origin
    
    def to_dense_full(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert the sparse map to a dense representation covering observed area.
        
        Returns:
            (costmap, origin, size): 
                - dense costmap covering observed region
                - origin in world coordinates
                - (rows, cols) of the dense map
        """
        if self._indices.shape[1] == 0:
            # Return small empty map if no observations
            return (
                torch.full((10, 10), self.unknown_cost, device=self.device),
                self.origin,
                torch.tensor([10, 10], device=self.device)
            )
        
        # Get observed bounds
        min_i, min_j = self._observed_min.long()
        max_i, max_j = self._observed_max.long()
        
        # Add padding
        padding = 5
        min_i = max(0, min_i - padding)
        min_j = max(0, min_j - padding)
        max_i = min(self.max_size[0] - 1, max_i + padding)
        max_j = min(self.max_size[1] - 1, max_j + padding)
        
        height = max_i - min_i + 1
        width = max_j - min_j + 1
        
        # Initialize with unknown cost
        dense_map = torch.full(
            (height, width), self.unknown_cost, device=self.device
        )
        
        # Map sparse indices to dense indices
        dense_i = self._indices[0] - min_i
        dense_j = self._indices[1] - min_j
        
        # Filter valid indices (all observed cells are valid - no min_observations check)
        valid = (
            (dense_i >= 0) & (dense_i < height) &
            (dense_j >= 0) & (dense_j < width)
        )
        
        if valid.any():
            dense_map[dense_i[valid], dense_j[valid]] = self._costs[valid].clamp(0.0, 1.0)
        
        # Origin in world coordinates
        origin = torch.tensor([min_i, min_j], dtype=torch.float32, device=self.device)
        origin = origin * self.resolution + self.origin
        
        return dense_map, origin, torch.tensor([height, width], device=self.device)
    
    def get_statistics(self) -> Dict:
        """Get costmap statistics for debugging."""
        if self._indices.shape[1] == 0:
            return {
                'num_cells': 0,
                'total_updates': self.total_updates,
                'memory_mb': 0.0,
            }
        
        # Estimate memory usage (only _indices and _costs now, no _hit_count)
        memory_bytes = (
            self._indices.numel() * 8 +  # long indices
            self._costs.numel() * 4       # float costs
        )
        
        return {
            'num_cells': self.num_cells,
            'num_known_cells': self.num_cells,  # All observed cells are "known" now
            'total_updates': self.total_updates,
            'memory_mb': memory_bytes / (1024 * 1024),
            'cost_mean': self._costs.mean().item() if self._costs.numel() > 0 else 0.0,
            'cost_max': self._costs.max().item() if self._costs.numel() > 0 else 0.0,
            'observed_area_cells': (
                (self._observed_max - self._observed_min + 1).prod().item()
                if self._observed_min[0] != float('inf') else 0
            ),
        }
    
    def to(self, device: str):
        """Move costmap to a different device."""
        self.device = device
        self.origin = self.origin.to(device)
        self._indices = self._indices.to(device)
        self._costs = self._costs.to(device)
        self._observed_min = self._observed_min.to(device)
        self._observed_max = self._observed_max.to(device)
        return self
    
    def reset(self):
        """Clear all observations."""
        self._indices = torch.empty((2, 0), dtype=torch.long, device=self.device)
        self._costs = torch.empty(0, dtype=torch.float32, device=self.device)
        self._observed_min = torch.tensor([float('inf'), float('inf')], device=self.device)
        self._observed_max = torch.tensor([float('-inf'), float('-inf')], device=self.device)
        self.num_cells = 0
        self.total_updates = 0
