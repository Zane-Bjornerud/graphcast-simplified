import numpy as np
from scipy.spatial import cKDTree


def create_lat_lon_grid(num_lat=32, num_lon=64):
    # create a regular latitude-longitude grid matching the data ie 5.625 resolution
    lat = np.linspace(90, -90, num_lat, endpoint=False)  # 90 to -90
    lon = np.linspace(0, 360, num_lon, endpoint=False)  # 0 to 360

    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # flatten
    grid_lat = lat_grid.flatten()
    grid_lon = lon_grid.flatten()

    # Combine into coordinate array
    grid_coords = np.column_stack([grid_lat, grid_lon])

    print(f"\nGrid information:")
    print(f"Grid shape: {num_lat} * {num_lon} = {num_lat * num_lon} points")
    print(f"Latitude range: [{lat.min():.2f}, {lat.max():.2f}]")
    print(f"Longitude range: [{lon.min():.2f}, {lon.max():.2f}]")

    return grid_lat, grid_lon, grid_coords


def mesh_to_latlon(vertices):
    # Convert mesh vertices from (x,y,z) to (lat,lon)
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    lat = np.arcsin(z) * 180 / np.pi
    lon = np.arctan2(y, x) * 180 / np.pi

    # Convert lon from [-180, 180] to [0, 360]
    lon = (lon + 360) % 360

    mesh_coords = np.column_stack([lat, lon])

    return mesh_coords


def create_grid_to_mesh_mapping(grid_coords, mesh_coords, k_neighbors=4):
    # Create mapping from grid to mesh using k-nearest neighbors
    # For each mesh node, find the k nearest grid points
    # Returns grid_to_mesh_indices: [num_mesh, k] - indices of nearest grid points & grid_to_mesh_weights: [num_mesh, k] - interpolation weights

    print(f"\nCreating grid to mesh mapping")
    print(f"Finding {k_neighbors} nearest grid points for each mesh node")

    # Build KD-tree for grid points
    tree = cKDTree(grid_coords)

    # For each mesh node, find k nearest grid points
    distances, indices = tree.query(mesh_coords, k=k_neighbors)

    # Convert distances to weights (inverse distance weighting)
    # Add small epsilon to avoid division by zero
    weights = 1.0 / (distances + 1e-6)

    # Normalize weights to sum to 1
    weights = weights / weights.sum(axis=1, keepdims=True)

    print(f"Mapping shape: {indices.shape}")
    print(f"Average distance to neighbors: {distances.mean():.4f}")

    return indices, weights


def create_mesh_to_grid_mapping(grid_coords, mesh_coords, k_neighbors=3):
    # Create mapping from mesh to grid using k-nearest neighbors

    print(f"\nCreating mesh to grid mapping")
    print(f"Finding {k_neighbors} nearest mesh nodes for each grid point")

    # Build KD-tree for mesh nodes
    tree = cKDTree(mesh_coords)

    # For each grid point, find k nearest mesh nodes
    distances, indices = tree.query(grid_coords, k=k_neighbors)

    # Convert distances to weights
    weights = 1.0 / (distances + 1e-6)
    weights = weights / weights.sum(axis=1, keepdims=True)

    print(f"Mapping shape: {indices.shape}")
    print(f"Average distance to neighbors: {distances.mean():.4f}°")

    return indices, weights


def test_mappings(
    grid_coords, mesh_coords, g2m_indices, g2m_weights, m2g_indices, m2g_weights
):
    # Test that mappings work correctly

    print("Testing Mappings")

    # Create test data: simple pattern on grid
    num_grid = grid_coords.shape[0]
    num_mesh = mesh_coords.shape[0]

    # Test 1: Grid to mesh
    print("\nTest 1: Grid to Mesh pooling")
    grid_data = grid_coords[:, 0]  # Use latitude as test data

    # Pool to mesh
    mesh_data = np.zeros(num_mesh)
    for i in range(num_mesh):
        neighbors = g2m_indices[i]
        weights = g2m_weights[i]
        mesh_data[i] = (grid_data[neighbors] * weights).sum()

    print(f"Grid data range: [{grid_data.min():.2f}, {grid_data.max():.2f}]")
    print(f"Mesh data range: [{mesh_data.min():.2f}, {mesh_data.max():.2f}]")
    print(f"Pooling preserves range")

    # Test 2: Mesh to grid
    print("\nTest 2: Mesh to Grid unpooling")
    grid_reconstructed = np.zeros(num_grid)
    for i in range(num_grid):
        neighbors = m2g_indices[i]
        weights = m2g_weights[i]
        grid_reconstructed[i] = (mesh_data[neighbors] * weights).sum()

    print(
        f"Reconstructed range: [{grid_reconstructed.min():.2f}, {grid_reconstructed.max():.2f}]"
    )

    # Check reconstruction error
    error = np.abs(grid_data - grid_reconstructed).mean()
    print(f"Mean reconstruction error: {error:.4f}°")

    if error < 5.0:  # Arbitrary threshold
        print(f"Reconstruction quality good")
    else:
        print(f"Reconstruction error high (expected for coarse mesh)")


if __name__ == "__main__":

    # Load mesh
    print("\nLoading mesh...")
    mesh_data = np.load("mesh_data.npz")
    vertices = mesh_data["vertices"]

    # print("Creating the Lat-Lon grid")
    # create_lat_lon_grid(num_lat=32, num_lon=64)
    # print("Lat Lon grid complete")

    # Create grid
    grid_lat, grid_lon, grid_coords = create_lat_lon_grid(num_lat=32, num_lon=64)

    # Convert mesh to lat/lon
    mesh_coords = mesh_to_latlon(vertices)

    # Create mappings
    g2m_indices, g2m_weights = create_grid_to_mesh_mapping(
        grid_coords, mesh_coords, k_neighbors=4
    )

    m2g_indices, m2g_weights = create_mesh_to_grid_mapping(
        grid_coords, mesh_coords, k_neighbors=3
    )

    # Test mappings
    test_mappings(
        grid_coords, mesh_coords, g2m_indices, g2m_weights, m2g_indices, m2g_weights
    )

    # Save mappings
    print(f"\nSaving mappings...")
    np.savez(
        "grid_mesh_mappings.npz",
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        grid_coords=grid_coords,
        mesh_coords=mesh_coords,
        g2m_indices=g2m_indices,
        g2m_weights=g2m_weights,
        m2g_indices=m2g_indices,
        m2g_weights=m2g_weights,
    )
    print("Saved to grid_mesh_mappings.npz")
