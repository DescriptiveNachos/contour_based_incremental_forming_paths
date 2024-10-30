import time
import numpy as np
import pyvista as pv
from pykdtree.kdtree import KDTree
from utils import levels_from_layer_height, individualize_contours, choose_objects_from_list

class augmented_contour:
    def __init__(self,contour,scalars) -> None:
        self.contour = contour
        self.points = contour.points
        self.anker_point = contour.points[0]
        self.level = np.round(contour[scalars][0],4)
        self.id = contour['RegionId'][0]
        self.level_id = None
        self.connection_state = 'free'
        self.parents = set()
        self.children = set()

    def find_closest_contour(self,other_contours):
            all_dists = [KDTree(other_contour.points).query(self.points,k=1)[0] for other_contour in other_contours]
            dists = [np.min(dist_list) for dist_list in all_dists]
            dist = np.min(dists)
            idx = np.argmin(dists)
            return dist,idx

class Slicer:
    def __init__(self,mesh) -> None:
    
        self.mesh = mesh
        self.mesh["elevation"] = self.mesh.points[:,2]

        #getting appended by slicing
        self.contours = []
        self.ids = []
        #getting set by slicing
        self.layer_height = None
        self.num_of_layers = None
        self.levels = []
        self.slices = []

    def slice(self,target_layer_height,scalars='elevation',offset=0.3,length_thresh=2,include_open_contours=True):
            """slices the base mesh of the slicer object into equaly spaced slices on the specified scalarfield which are lists of all contours with a specific isovalue
            specify
                - target_layer_heigth / distance between slices
                - scalars / the scalars on which the iso-contours are generated (elevation or geodesic_distance)
                - offset / offset of the first slice from the minimum value
                - length_thresh / filter contours with less then this points
                - include_open_contours / set to false to filter out open contours
            """

            start_time = time.time()
            self.layer_height, self.num_of_layers, self.levels = levels_from_layer_height(self.mesh,target_layer_height,offset=offset,scalars=scalars)
            contours = self.mesh.contour(self.levels,scalars=scalars)  
            self.levels = np.unique(np.round(contours[scalars],4))

            #individualize and filter contours and save them in a list
            contour_list, self.ids = individualize_contours(contours)

            self.contours = [augmented_contour(contour,scalars) for contour in contour_list]
            for contour in self.contours: contour.level_id = np.nonzero(self.levels == contour.level)[0][0]
            #update levels in case some levels only had short or open contours
            self.levels = np.unique(np.array([contour.level for contour in self.contours]))
            #sorts in ascending order
            self.slices = self.sort_contours_into_slices(self.contours)
    
    def sort_contours_into_slices(self,contours):
        return [choose_objects_from_list(contours,'level',level) for level in self.levels]

def plot_connection_states(augmented_contours,plotter):
    for contour in augmented_contours:
        if contour.connection_state == 'root': color = 'green'
        elif contour.connection_state == 'leaf': color = 'red'
        elif contour.connection_state == 'connected': color = 'blue'
        elif contour.connection_state == 'lonely_root': color = 'orange'
        plotter.add_points(contour.points,color=color)
   
