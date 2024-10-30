import numpy as np
import pyvista as pv
from utils import order_open_contour_points, order_points_in_chain, roll_to_closest_kdtree, resample_curve, polyline_from_points, check_and_split

def preprocess_branch(branch,spin='cw'):
    """Prepares branch for spiraling by ordering and aligning all contours """
    for augmented_contour in branch:
        augmented_contour.points = order_points_in_chain(augmented_contour.points,dir=spin)
        
    #Initialize the result list with the first line unchanged
    last_line = branch[0].points

    # Start aligning from the second line
    for i in range(1, len(branch)):
        last_point = last_line[-1]  # Last point of the previous line
        current_line = branch[i].points
        # Roll the current line using KD-Tree to start with the point closest to the last point
        rolled_line = roll_to_closest_kdtree(current_line, last_point)
        #update branch with rolled line
        branch[i].points = rolled_line
        last_line = rolled_line
    return branch

def generate_spiral(branch):
    """assumes a branch section of closed and sorted contours. Direction, spin and starting Point of the contours are also already set during preprocessing."""

    spiral = []
    for current_contour,next_contour in zip(branch[:-1],branch[1:]):

        current_contour = current_contour.points
        next_contour = next_contour.points

        dists = []
        slopes = []

        dist = 0

        number_resample_points = max([len(current_contour),len(next_contour)])

        #to be able to get a bijective map between the contours they are both resampled to have the same amount of points
        current_contour = resample_curve(current_contour,number_resample_points)
        next_contour = resample_curve(next_contour,number_resample_points)
        
        for i in range(0,len(current_contour)):
            dists.append(dist)
            dist = dist + 1
            slope = next_contour[i] - current_contour[i]
            slopes.append(slope)
          
        total_dist = dists[-1]
        #generate spiral by moving each inner contour point out according to the distance traveled along the contour
        for point,slope,dist in zip(current_contour,slopes,dists):
            spoint = point + slope * dist/total_dist
            spiral.append(spoint)
    return np.array(spiral)

def spiral_paths_from_contour_tree(contour_tree,point_spacing,retract_dist,spin='cw',include_open_contours=True):
    """Takes a contour tree and generates paths from it. Branch segments of multiple closed contours will be spiraled. The rest is traversed normally\n
    Specify the spin of the resulting path i.e. if it is traversed cw or ccw, the point spacing of the resulting paths and the 
    retract distance for movements between path segments\n
    Returns a list of all path segments as np.arrays of points"""
    path_segments = []
    for branch in contour_tree:
        #open contours have less cells than points, so this property is used
        is_open_contour = lambda x: x.contour.n_points != x.contour.n_cells
        sectioned_branch = check_and_split(branch,is_open_contour)
        for branch_section in sectioned_branch:
            if len(branch_section) > 1:
                #prepare the section to be spiraled
                branch_section = preprocess_branch(branch_section,spin=spin)
                #generate spiral
                spiral_points = generate_spiral(branch_section)
                #add initial and final contour as a complete pass
                path_points = np.concatenate((branch_section[0].points,spiral_points,branch_section[-1].points))
            elif is_open_contour(branch_section[0]):
                if not include_open_contours: continue
                path_points = order_open_contour_points(branch_section[0].contour)
            else:
                path_points = order_points_in_chain(branch_section[0].points,dir=spin)

            path_segments.append(path_points)

    return path_segments
    