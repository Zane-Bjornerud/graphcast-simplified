import numpy as np
import trimesh


def create_icosphere(subdiv=2):
    # subdiv = number of subdivision levels
    print(f"\nCreating icosphere with {subdiv} subdivisions...")

    # Create icosphere using trimesh
    sphere = trimesh.creation.icosphere(subdivisions=subdiv, radius=1.0)

    vertices = sphere.vertices  # [num_nodes, 3]
    faces = sphere.faces  # [num_faces, 3]
    # Extract edges from faces
    edges = sphere.edges_unique  # [num_edges, 2]

    print(f"Created mesh:")
    print(f"Vertices: {vertices.shape[0]}")
    print(f"Faces: {faces.shape[0]}")
    print(f"Edges: {edges.shape[0]}")

    return vertices, faces, edges


def cartesian_to_latlon(vertices):
    # Convert (x, y, z) Cartesian coordinates to (lat, lon)
    # vertices: [num_nodes, 3] - (x, y, z) on unit sphere
    # Returns: lat: [num_nodes] - latitude in degrees [-90, 90] & lon: [num_nodes] - longitude in degrees [-180, 180]
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    # Latitude: arcsin(z)
    lat = np.arcsin(z) * 180 / np.pi

    # Longitude: arctan2(y, x)
    lon = np.arctan2(y, x) * 180 / np.pi

    return lat, lon


def visualize_mesh(vertices, edges=None, faces=None):
    # Print mesh statistics and basic visualization info
    # vertices: [num_nodes, 3]
    # edges: [num_edges, 2] (optional)
    # faces: [num_faces, 3] (optional)

    lat, lon = cartesian_to_latlon(vertices)

    print(f"\nMesh Statistics:")
    print(f"  Number of nodes: {vertices.shape[0]}")

    if edges is not None:
        print(f"  Number of edges: {edges.shape[0]}")

        # Average degree (edges per node)
        degrees = np.bincount(edges.flatten())
        avg_degree = degrees.mean()
        print(f"  Average node degree: {avg_degree:.2f}")

    if faces is not None:
        print(f"  Number of faces: {faces.shape[0]}")

    print(f"\nGeographic distribution:")
    print(f"Latitude range: [{lat.min():.2f}°, {lat.max():.2f}°]")
    print(f"Longitude range: [{lon.min():.2f}°, {lon.max():.2f}°]")

    # Count nodes by latitude bands
    tropics = np.sum((lat >= -23.5) & (lat <= 23.5))
    mid_lat = np.sum(((lat < -23.5) & (lat >= -66.5)) | ((lat > 23.5) & (lat <= 66.5)))
    polar = np.sum((lat < -66.5) | (lat > 66.5))

    print(f"\nNodes by region:")
    print(f"Tropics (+-23.5): {tropics} ({tropics/len(lat)*100:.1f}%)")
    print(f"Mid-latitudes: {mid_lat} ({mid_lat/len(lat)*100:.1f}%)")
    print(f"Polar (>66.5): {polar} ({polar/len(lat)*100:.1f}%)")


def create_mesh_graph(vertices, edges):
    # Create graph structure for GNN
    # Returns: mesh_nodes: [num_nodes, 3] - node features, mesh_edges: [2, num_edges] - edge indices (for PyTorch Geometric), edge_features: [num_edges, 1] - edge lengths

    # Node features: xyz coordinates
    mesh_nodes = vertices

    # Edge indices: transpose for PyTorch Geometric format
    mesh_edges = edges.T  # [2, num_edges]

    # Edge features: distances between connected nodes
    edge_features = []
    for i, j in edges:
        dist = np.linalg.norm(vertices[i] - vertices[j])
        edge_features.append(dist)

    edge_features = np.array(edge_features).reshape(-1, 1)

    print(f"\nGraph structure:")
    print(f"Node features shape: {mesh_nodes.shape}")
    print(f"Edge indices shape: {mesh_edges.shape}")
    print(f"Edge features shape: {edge_features.shape}")
    print(f"Average edge length: {edge_features.mean():.4f}")

    return mesh_nodes, mesh_edges, edge_features


if __name__ == "__main__":
    print("Creating Mesh")

    # create mesh
    vertices, faces, edges = create_icosphere(subdiv=2)

    # Visualize
    visualize_mesh(vertices, edges, faces)

    # Create graph structure
    mesh_nodes, mesh_edges, edge_features = create_mesh_graph(vertices, edges)

    # Save mesh
    print("\nSaving mesh...")
    np.savez(
        "mesh_data.npz",
        vertices=vertices,
        faces=faces,
        edges=edges,
        mesh_nodes=mesh_nodes,
        mesh_edges=mesh_edges,
        edge_features=edge_features,
    )

    print("Saved to mesh_data.npz")
