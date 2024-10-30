import math as m
import numpy as np
import pyvista as pv
from pykdtree.kdtree import KDTree
import potpourri3d as pp3d
from matplotlib import cm

def order_points_in_chain(points, dir='cw'):
        even = points[0::2]
        uneven = np.flip(points[1::2],axis=0)
        points = np.concatenate((even,uneven))
        if dir == 'ccw':
            np.flip(points,axis=0)
        return points

def order_open_contour_points(contour):

    end_point_idxs = []

    list_of_cells = contour.cells_dict[3].flatten()
    list_of_unique_cells = np.unique(list_of_cells)
    for cell in list_of_unique_cells:
        cells_containing_cell = np.nonzero(list_of_cells==cell)
        if cells_containing_cell[0].size == 1:
            end_point_idxs.append(cell)

    points = contour.points

    smaller_endpoint = min(end_point_idxs)
    larger_endpoint = max(end_point_idxs)
    if smaller_endpoint % 2 == 0:
        points_to_endpoint = np.insert(np.arange(0,smaller_endpoint+2,2),1,1)
        points_in_opposite_direction = np.arange(3,smaller_endpoint+1,2)
    else: 
        points_to_endpoint = np.arange(3,smaller_endpoint,2)
        points_in_opposite_direction = np.insert(np.arange(0,smaller_endpoint+1,2),1,1)
    
    left_over_points = np.arange(smaller_endpoint+1,larger_endpoint)
    point_idxs = np.concatenate((np.flip(points_to_endpoint),points_in_opposite_direction))
    point_idxs = np.concatenate((point_idxs,left_over_points))
    
    ordered_points = points[point_idxs]
    return ordered_points

def comp_geodesic(surf, inds):

    faces = surf.faces
    faces = faces.reshape((-1,4))
    faces = faces[:,1:]

    distances = pp3d.compute_distance_multisource(surf.points, faces, inds)

    return distances

def geo_dist_from_outer_edge(mesh):
    outer_edge = mesh.extract_feature_edges(non_manifold_edges=False, feature_edges=False, manifold_edges=False)
    _, outer_edge_indices = KDTree(mesh.points).query(outer_edge.points,k=1)
    geo_dist = comp_geodesic(mesh,list(outer_edge_indices))
    return geo_dist


def load_init_mesh(path=None,mesh=None,scalars='elevation'):
    """Takes a path or a mesh to load it and calculate elevation or geodesic distance (geo_dist) on the mesh and returns it
    """
    if not mesh and path: 
        try: mesh = pv.PolyData(path)
        except FileNotFoundError:
            print('No mesh found at this path')
            return None
    elif not mesh and not path: 
        print('nothing provided to load')
        return None
    if scalars == 'elevation': mesh["elevation"] = mesh.points[:,2]
    elif scalars == 'geo_dist': mesh['geo_dist'] = geo_dist_from_outer_edge(mesh)
    return mesh

def levels_from_layer_height(mesh,target_layer_height,scalars='elevation',offset=0.5):
    """ take layerheight and find integer number of layers that approximates it given the mesh dimensions
    optionally set the bottom offset that determines how far above the bottom of the mesh the first contour will be placed """
    min_height = np.min(mesh[scalars]) + offset
    max_height = np.max(mesh[scalars]) - offset
    num_of_layers = (max_height-min_height)/target_layer_height
    num_of_layers = m.floor(num_of_layers)
    layer_height = (max_height-min_height)/num_of_layers
    levels = np.arange(min_height,max_height+layer_height,layer_height,dtype='float32')
    levels = np.unique(np.round(levels,4))
    return layer_height, num_of_layers, levels

def create_morph_vectors(mesh,divisions):
    """adds morph_vectors to mesh: at every point of the mesh create a vector that points up and is of length elevation/division"""
    elevations = -mesh['elevation']/divisions
    morph_vectors = np.array([[0,0,z] for z in elevations])
    mesh['morph_vectors'] = morph_vectors
    return mesh

def individualize_contours(contours):
        contour_set = contours.connectivity('all')
        ids = np.unique(contour_set["RegionId"])
        contour_list = [contour_set.threshold([identifier-0.001, identifier+0.001],"RegionId") for identifier in ids]
        return contour_list, ids

def choose_objects_from_list(list,selection_property,selection_value):
        """returns sublist of a list of objects containing those objects that 
        have the specified selection value of the specified selection property"""
        return [item for item in list if vars(item)[selection_property] == selection_value]

def check_and_split(original_list, condition):
    """splits the input list wherever an item fulfills the condition with the item being
    its own sublist. I.e. if c condition true [a,b,c,d,e] = [[a,b],[c],[d,e]]"""
    result = []
    current_sublist = []

    for item in original_list:
        if condition(item):
            # If the condition is true, append the current sublist to the result (if not empty)
            if current_sublist:
                result.append(current_sublist)
                current_sublist = []
            # Append the item that fulfills the condition as a single-element list
            result.append([item])
        else:
            # Otherwise, add the item to the current sublist
            current_sublist.append(item)

    # Append any remaining items in the current sublist to the result
    if current_sublist:
        result.append(current_sublist)

    return result

def resample_curve(points,num_of_points):
    """resamples the given points to equally spaced number of points\n
    Assuming that 'data' is rows x dims (where dims is the dimensionality)"""
    x,y,z = points.T
    xd = np.diff(x)
    yd = np.diff(y)
    zd = np.diff(z)
    dist = np.sqrt(xd**2+yd**2+zd**2)
    u = np.cumsum(dist)
    u = np.hstack([[0],u])

    t = np.linspace(0,u.max(),num_of_points)
    xn = np.interp(t, u, x)
    yn = np.interp(t, u, y)
    zn = np.interp(t, u, z)
    return np.array([xn,yn,zn]).T

def roll_to_closest_kdtree(points, reference_point):
    """Reorders points to start with the point closest to the reference point"""
    # Build the KD-Tree from the points
    tree = KDTree(points)
    # Query the tree to find the index of the closest point to the reference point
    _, closest_idx = tree.query(reference_point[None, :], k=1)
    roll = len(points) - closest_idx[-1] - 1
    points = np.roll(points,roll,axis=0)
    # Roll the array so that the closest point is the first
    return points

def polyline_from_points(points):
    """turns points into a pyvista polyline"""
    poly = pv.PolyData()
    poly.points = points
    n_points = len(points)
    the_cell = np.arange(0, n_points, dtype=np.int_)
    the_cell = np.insert(the_cell, 0, n_points)
    poly.lines = the_cell
    return poly

def colors_for_list(list):
     """generate a list with one color for each item in the list"""
     return cm.rainbow(np.linspace(0, 1,len(list)))

def add_retract_points(path_points,retract_dist):
    path_points = np.vstack((np.append(path_points[0][:2],retract_dist),path_points))
    path_points = np.vstack((path_points,np.append(path_points[-1][:2],retract_dist)))
    return path_points

def resample_path_points(path_points,point_spacing):
    pathspline = polyline_from_points(path_points)
    path_length = pathspline.compute_arc_length()['arc_length'][-1]
    number_of_points = int(np.floor(path_length/point_spacing))
    if number_of_points <=1:
        return []
    resampled_path_points = resample_curve(path_points,number_of_points)
    return resampled_path_points
