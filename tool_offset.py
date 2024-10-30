import CGAL
from CGAL.CGAL_Alpha_wrap_3 import alpha_wrap_3
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from CGAL.CGAL_Kernel import Point_3
import meshio as mio
import numpy as np
import pyvista as pv
from cgal_custom_utils import pvmesh_2_cgalph3, get_cgal_ph3_vf
from utils import polyline_from_points


def faces_to_pv_faces(faces):
    # Convert faces to pyvista format
    # PyVista expects a flat array where each face is preceded by the number of vertices it contains
    faces_pv = []
    for face in faces:
        faces_pv.append(len(face))  # Number of vertices in the face
        faces_pv.extend(face)       # Add the vertex indices
    faces_pv = np.array(faces_pv)
    return faces_pv

def generate_offset_mesh(base_mesh,offset,alpha):
    #generate offset mesh by alpha wrapping the base mesh
    output_mesh = Polyhedron_3()
    input_mesh = pvmesh_2_cgalph3(base_mesh)
    alpha_wrap_3(input_mesh, alpha, offset,output_mesh)
    #convert the alpha wrap mesh to a pyvista polydata object
    out_points,out_faces = get_cgal_ph3_vf(output_mesh)
    out_faces = faces_to_pv_faces(out_faces)
    offset_mesh = pv.PolyData(out_points,out_faces)
    return offset_mesh

def project_path_from_mesh_to_mesh(path,base_mesh,target_mesh,flip_normal_direction=False):
    base_mesh.compute_normals(cell_normals=False,inplace=True,split_vertices=True)
    path = path.interpolate(base_mesh,strategy='mask_points')
    if flip_normal_direction:
        path.point_data['Normals'] = -path.point_data['Normals']
   
    #use multi ray tracing to project the path onto the offset mesh along the point normals
    points, rays, cells = target_mesh.multi_ray_trace(path.points,path.point_data['Normals'],first_point=True)
    sorted_points = np.full((len(points),3),[0,0,0])
    for point, id in zip(points,rays):
        sorted_points[id] = point

    projected_path = pv.PolyData(sorted_points,lines=path.lines)
    return projected_path

def offset_paths_master_slave(path_points,target_surface,tool_diameter,material_thickness,flip_normals=False):
    tool_offset = (tool_diameter+material_thickness)/2
    pathspline = polyline_from_points(path_points)
    target_surface.compute_normals(cell_normals=False,inplace=True,split_vertices=True,flip_normals=flip_normals)
    pathspline = pathspline.interpolate(target_surface,strategy='closest_point')
    normals = align_normals_along_z(pathspline['Normals'])
    normals = pathspline['Normals']
    master_path_points = path_points + normals * tool_offset
    slave_path_points =  - normals * tool_offset
    slave_path_points_visu = path_points - normals * tool_offset
    return master_path_points, slave_path_points, slave_path_points_visu

def align_normals_along_z(normals):
    new_normals = []
    for normal in normals:
        if normal[2]<0:
            new_normal = -normal
            new_normals.append(new_normal)
            continue
        elif normal[2] == 0:
            print('WARNING horizontal face. correct normal computation can not be guaranteed')
        new_normals.append(normal)
    return np.array(new_normals)


# cad_path = 'IOTest2.stl'
# cad_path = 'simpleDome.stl'

# dome_mesh = pv.PolyData(cad_path)
# dome_mesh['elevation'] = dome_mesh.points[:,2]
# dome_mesh.set_active_scalars('elevation')
# dome_mesh.compute_normals(cell_normals=True,inplace=True,split_vertices=False)
# contours = dome_mesh.contour(10)
# paths = pv.PolyData(contours.points,contours.lines)

# path = contours.sample(dome_mesh)


# offset_contours,offset_mesh = offset_path_on_mesh(contours,dome_mesh,5,flip_normal_direction=False,alpha=5)

# p1 = pv.Plotter()
# p1.add_mesh(offset_contours,color='red')
# p1.add_mesh(offset_contours.points,color='blue')
# p1.add_mesh(offset_mesh)
# p1.show()