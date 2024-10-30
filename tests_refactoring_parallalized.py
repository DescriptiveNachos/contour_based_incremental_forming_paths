import concurrent.futures
import numpy as np
import pyvista as pv
import time
from utils import load_init_mesh, create_morph_vectors, levels_from_layer_height, colors_for_list, add_retract_points, resample_path_points, polyline_from_points
from contour_extension_classes import Slicer, plot_connection_states
from forest import generate_contour_tree
from spiral_paths import spiral_paths_from_contour_tree
from tool_offset import offset_paths_master_slave

def process_slicer(args):
    """parallel processing of intermediate shapes"""

    Slicer_of_Step, target_layer_height, scalars, bottom_offset, retract_dist, branch_dir, \
    point_spacing, spin, include_open_contours, tool_diameter, material_thickness, \
    flip_normals, length_thresh = args
    
    Slicer_of_Step.slice(target_layer_height,
                         scalars=scalars,
                         offset=bottom_offset)
    
    pos_retract_distance = np.max(Slicer_of_Step.levels) + retract_dist
    neg_retract_distance = np.min(Slicer_of_Step.levels) - retract_dist
      

    contour_tree = generate_contour_tree(Slicer_of_Step.slices,Slicer_of_Step.contours,dir=branch_dir)
    # p1 = pv.Plotter()
    # for branch in contour_tree:
    #     plot_connection_states(branch,p1)
    # p1.show()

    raw_path_segments = spiral_paths_from_contour_tree(contour_tree,point_spacing,retract_dist,spin=spin,include_open_contours=include_open_contours)

    offset_path_segments = []
    for path_points in raw_path_segments:
        path_points = resample_path_points(path_points,point_spacing)
        if len(path_points)==0:
            print('WARNING contour length <= point spacing -> contour ommited')
            continue
        master_path,slave_path,slave_path_visu = offset_paths_master_slave(path_points,Slicer_of_Step.mesh,tool_diameter,material_thickness,flip_normals=flip_normals)
        paths = [master_path,slave_path,slave_path_visu]
        offset_path_segments.append(paths)

    final_path_segments = []
    for segment_paths in offset_path_segments:
        paths = []
        for i in range(len(segment_paths)):
            path_points = segment_paths[i]
            if i>0: signed_retract_dist = neg_retract_distance
            else: signed_retract_dist = pos_retract_distance
            path_points = add_retract_points(path_points,signed_retract_dist)
            pathspline = polyline_from_points(path_points)
            pathspline["scalars"] = np.arange(pathspline.n_points)
            path_length = pathspline.compute_arc_length()['arc_length'][-1]
            if path_length<length_thresh:
                print('WARNING path length below threshhold -> contour ommited')
                continue
            paths.append(pathspline)

        final_path_segments.append(paths)

    return final_path_segments

if __name__ == '__main__':
    start_time = time.time()
    #######################################---main---###########################################################
    cad_path = 'immerwilder.stl'
    cad_path = 'IOTest2.stl'
    cad_path = 'Britenbiegeteil.stl'
    #cad_path = 'immerwilder.stl'
    cad_path = 'SuperNeuesBiegeteil.stl'

    tool_diameter = 10
    material_thickness = 0

    include_open_contours = True
    length_thresh = 15 #mm

    #this is the value by which the total height of the part will be devided to form the intermediate shapes
    #it can be interpreted as the maximum height difference between to successive forming stept
    #set to -1 to disable intermediate shapes
    division_layer_height = 5

    #when target layer height is larger than division_layer_height/2 the intermediate meshes smaller than twice the target layer height are skipped
    target_layer_height = 10


    bottom_offset = 0.01
    distance_filter_thresh = 0
    point_spacing = 5 #mm
    retract_dist = 100 #mm #relative to the lowest and highest point of the processed geometry

    scalars = 'geo_dist'
    scalars = 'elevation'

    flip_normals = False

    spin = 'cw'
    branch_dir = 'up'

    robot_type = 'ABB'
    robot_pose_config = 'A 0, B 0, C 0, S 2, T 2'
    ########################################################################################################

    mesh = load_init_mesh(cad_path)

    robtarget_strings = []
    list_of_path_instructions = []
    layers_in_path = []

    util_slicer = Slicer(mesh)

    if division_layer_height <= 0: divisions = 1
    else:
        act_div_layer_height, divisions,_ = levels_from_layer_height(mesh,division_layer_height,offset=0)
        print("mesh divided into %i substeps at an actual layerheight of %f mm"%(divisions,act_div_layer_height))

    mesh = create_morph_vectors(mesh,divisions)

    Slicers = []
    steps = range(1,divisions+1)

    step_time = time.time()
    for step in steps:
        #morphs the input mesh by the vectors pointing straigth up. The first morphed mesh is the one closest to the flat configuration, the last one is the final shape
        morphed_mesh = mesh.warp_by_vector('morph_vectors',factor=divisions-step)
        morphed_mesh = load_init_mesh(morphed_mesh)
        #check if shape height larger layer height
        if (np.max(morphed_mesh[scalars])-np.min(morphed_mesh[scalars])) < (2 * (target_layer_height + bottom_offset)): continue
        #initiate the slicer with the morphed mesh
        Slicers.append(Slicer(morphed_mesh))
    print("--- %s seconds for processing meshes ---" % (time.time() - step_time))

   # Prepare arguments for each call to process_slicer
    args = [
        (slicer, target_layer_height, scalars, bottom_offset, retract_dist, branch_dir, 
         point_spacing, spin, include_open_contours, tool_diameter, material_thickness, 
         flip_normals, length_thresh) for slicer in Slicers
    ]
    
    # Use a ProcessPoolExecutor for parallel processing
    step_time = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Pass args to process_slicer without lambda
        results = list(executor.map(process_slicer, args))
    print("--- %s seconds for slicer processing ---" % (time.time() - step_time))

    # Optionally, visualize or further process the results
    # for slicer_id, final_path_segments in enumerate(results):
    #     p1 = pv.Plotter()
    #     p1.add_mesh(Slicers[slicer_id].mesh)
    #     colors = colors_for_list(final_path_segments[0])
    #     for color_id, segment in enumerate(final_path_segments):
    #         for color_id, path in enumerate(segment):
    #             p1.add_mesh(path, color=colors[color_id])
    #     # Uncomment to show the plot for each slicer if needed
    #     p1.show()

    print("fin")