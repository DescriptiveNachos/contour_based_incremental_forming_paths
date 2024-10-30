from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Polyhedron_3 import Polyhedron_modifier, Polyhedron_3, ABSOLUTE_INDEXING
import pyvista as pv
import numpy as np

def pvmesh_2_cgalph3(pv_mesh):
    #extract points and faces from pyvista mesh
    points = pv_mesh.points
    faces = pv_mesh.faces.reshape((-1, 4))[:, 1:]
    # Step 2: Create the CGAL Polyhedron_3 object
    polyhedron = Polyhedron_3()

    # Step 3: Create a modifier and begin adding vertices and faces
    m = Polyhedron_modifier()

    # First, add vertices to the modifier
    m.begin_surface(len(points), len(faces))

    # Add all vertices from the pyvista mesh to the modifier
    for point in points:
        m.add_vertex(Point_3(float(point[0]), float(point[1]), float(point[2])))

    # Now add the faces (triangles or quads) to the modifier
    for face in faces:
        m.begin_facet()
        for vertex_idx in face:
            m.add_vertex_to_facet(int(vertex_idx))  # Add the vertex index to the face
        m.end_facet()

    # End surface creation
    m.end_surface()

    # Step 4: Delegate the construction of the polyhedron to the modifier
    polyhedron.delegate(m)

    # Optional: Verify the size of the polyhedron (vertices, facets, edges)
    print(f"(vertices, facets, edges) = {polyhedron.size_of_vertices()}, {polyhedron.size_of_facets()}, {polyhedron.size_of_halfedges() // 2}")

    # Optional: Ensure the polyhedron is valid
    assert polyhedron.is_valid()

    # The polyhedron now contains the mesh data from the pyvista PolyData object

    return polyhedron

def get_cgal_ph3_vf(polyhedron):
    """get cgal Polyhedron 3 object vertices and faces"""
    # Step 1: Manually create an index map for vertices
    vertex_index_map = {}
    points = []

    # Iterate through all vertices in the polyhedron and map them to an index
    for i, vertex in enumerate(polyhedron.vertices()):
        point = vertex.point()
        points.append([point.x(), point.y(), point.z()])
        vertex_index_map[vertex] = i  # Assign an index to each vertex

    # Step 2: Extract faces using the vertex indices from the map
    faces = []
    for face in polyhedron.facets():
        halfedge = face.halfedge()
        face_vertices = []
        start = halfedge
        while True:
            face_vertices.append(vertex_index_map[halfedge.vertex()])  # Use the mapped index
            halfedge = halfedge.next()
            if halfedge == start:
                break
        faces.append(face_vertices)

    # Convert points to numpy array
    points_np = np.array(points)

    # Convert faces to pyvista format
    faces_np = []
    for face in faces:
        faces_np.append(len(face))  # Number of vertices in the face
        faces_np.extend(face)       # Add the vertex indices
    faces_np = np.array(faces_np)

    return points,faces


# cad_path = 'simpleDome.stl'
# #out_path = 'simpleDomeWraped.stl'

# dome_mesh = pv.PolyData(cad_path)

# ph3 = pvmesh_2_cgalph3(dome_mesh)
# print(ph3)