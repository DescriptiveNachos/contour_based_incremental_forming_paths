import numpy as np
import copy
import networkx as nx
import shapely.geometry as sg 
from utils import choose_objects_from_list,order_points_in_chain


def build_graph(slices,graph,direction='up'):
    """builds a graph connecting the closest contour 
    in the specified direction until no free contour is found in the next slice, 
    then starts at the highest free contour again"""
    oriented_slices = copy.deepcopy(slices)
    if direction == 'down':
        oriented_slices.reverse()
    while True:
        active_contour = None
        for current_contours, next_contours in zip(oriented_slices[:-1],oriented_slices[1:]):
            
            #active contours are chosen from all current contours by always choosing the first free contour
            if active_contour == None:
                free_contours = choose_objects_from_list(current_contours,'connection_state','free')
                #if the current slice contains free contours the first one is chosen as the root
                if free_contours:
                    active_contour = free_contours[0]
                    active_contour.connection_state = 'root' 
                #otherwise the loop continues to the next slice until all slices have been checked for free contours
                else: continue
                        
            #all contours in the next slice are checked to find the the closest contour to the currently active one
            i = next_contours[0]
            dist,idx = active_contour.find_closest_contour(next_contours)
            closest_contour = next_contours[idx]
            #the closest contour is connected to the active one
            id_active = active_contour.id
            id_closest = closest_contour.id
            graph.add_edge(id_active, id_closest, weight=dist)
            #continue in the next slice with the now connected contour
            closest_contour.connection_state = 'connected'
            active_contour = closest_contour
        if active_contour == None: break
    return graph

def prune_join_tree(J,augmented_contours):
    """""The geometric containment of the contours is used to remove edges from the join tree.
    Non simply connected nodes are investigated
    Edges are removed where connected contours do not contain each other
    And multiple contours on the same level with a common neighbour are reduced by keeping only the connection
    of the outer most contour 
    """
    #Optimization potential if this deepcopy can be avoided
    new_J = copy.deepcopy(J)
    
    #All nodes that are not simply connected i.e. lonely roots, roots/leafs with exactly one edge or connected nodes with two edges
    #are investigated.
    for node in J.nodes:
        
        #levels are all levels of nodes that are connected to the current node (i.e. its neighbours)
        levels = [augmented_contours[neighbour].level for neighbour in J.neighbors(node)]
        unique_levels = np.unique(levels)
        edges = J.edges(node)
        if node_simply_connected(edges,unique_levels): continue
        #the levels where one unique level contains multiple of the neighbours
        levels_of_multiple_contours = np.unique([level for level in levels if levels.count(level)>1])

        #from the geometric containment properties of the contours further information on the possibility on parent-child 
        #relationships can be extracted: 
        for level in levels_of_multiple_contours:
            #multiple nodes connected to the currently investigated node on the same level
            nodes_of_level = [neighbour for neighbour in J.neighbors(node) if augmented_contours[neighbour].level == level]
            #the node from the top level for loop is referred to as the parent here. we find its contour and anker point
            #and convert them to spacial geometry objects
            sg_parent_contour = sg.Polygon(order_points_in_chain(augmented_contours[node].contour.points[:,0:2]))
            sg_parent_anker = sg.Point(augmented_contours[node].anker_point[:2])
            #the nodes of the currently investigated level of multiple neighbours are used to generate a list of contours and
            #anker points
            sg_contour_list = [sg.Polygon(order_points_in_chain(augmented_contours[this_node].contour.points[:,0:2])) for this_node in nodes_of_level]
            sg_anker_list = [sg.Point(augmented_contours[this_node].anker_point[:2]) for this_node in nodes_of_level]
            #now we find out which of the connections are geometrically possible
            for current_node,current_anker,current_contour in zip(nodes_of_level,sg_anker_list,sg_contour_list):
                #if two contours are geometrically disjoined i.e. do not contain each other there can not be a true connection
                #so these edges are removed
                if not (current_anker.within(sg_parent_contour) or sg_parent_anker.within(current_contour)): 
                    if new_J.has_edge(current_node,node): new_J.remove_edge(current_node,node)
                #for contours on the same level that contain each other, only the conection of the outermost contour to the parent
                #is kept
                parent_list = [current_anker.within(other_contour) for node_of_contour,other_contour in zip(nodes_of_level,sg_contour_list) if node_of_contour!=current_node]
                if any(parent_list):
                    if new_J.has_edge(current_node,node): new_J.remove_edge(current_node,node)
    return new_J

def node_simply_connected(edges,unique_levels):
        if len(edges)<=1: return True
        if len(edges)==2: 
            if not len(unique_levels) == 1:
                return True
        return False

def extract_split_merge_nodes(J,augmented_contours):
    """extracts all non simply connected nodes"""
    split_merge_nodes = []
    for node in J.nodes:
        edges = J.edges(node)
        levels = np.array([augmented_contours[neighbour].level for neighbour in J.neighbors(node)])
        unique_levels = np.unique(levels)
        if not node_simply_connected(edges,unique_levels): split_merge_nodes.append(node)
    return split_merge_nodes

def prune_split_tree(S,J,augmented_contours):
    """removes all edges from the split tree where at least one of the nodes is not simply connected in the join tree"""
    #extract all nodes that are merges or splits in the join tree
    split_merge_nodes = extract_split_merge_nodes(J,augmented_contours)
    for split_merge_node in split_merge_nodes:
        #we find those levels where the split_merge_node has multiple children
        levels = [augmented_contours[neighbour].level for neighbour in J.neighbors(split_merge_node)]
        levels_of_multiple_contours = np.unique([level for level in levels if levels.count(level)>1])
        #we remove those connections from the split tree 
        for level in levels_of_multiple_contours:
            nodes_of_level = [neighbour for neighbour in J.neighbors(split_merge_node) if augmented_contours[neighbour].level == level]
            for node_of_level in nodes_of_level:
                if S.has_edge(node_of_level,split_merge_node): S.remove_edge(node_of_level,split_merge_node)
    return S

def set_connection_state(contour, graph, augmented_contours):
    neighbors = [neigh for neigh in graph.neighbors(contour.id)]
    num_of_neighbors = len(neighbors)
    if num_of_neighbors == 0: contour.connection_state = 'lonely_root'
    elif num_of_neighbors > 1: contour.connection_state = 'connected'
    elif num_of_neighbors == 1:
        level_of_neighbor = augmented_contours[neighbors[0]].level
        if contour.level < level_of_neighbor: contour.connection_state = 'leaf'
        elif contour.level > level_of_neighbor: contour.connection_state = 'root'
    return contour

def sort_branch_by_level(branch,augmented_contours,dir='up'):
    """sort contours in branch by level. defaults to sort to increasing order. specify dir=down to invert"""
    levels_in_branch = [augmented_contours[node].level for node in branch]
    branch = list(branch)
    new_branch = [branch[sort_index] for sort_index in np.argsort(levels_in_branch)]
    if dir == 'down': new_branch = new_branch[::-1]
    return new_branch
    
def nodal_to_contour_tree(nodal_tree,augmented_contours,graph):
    """translates branches of nodes to branches of contours"""
    contour_tree = []
    for branch in nodal_tree:
        contour_branch = []
        for node in branch:
            nodal_contour = augmented_contours[node]
            nodal_contour = set_connection_state(nodal_contour,graph,augmented_contours)
            contour_branch.append(nodal_contour)
        contour_tree.append(contour_branch)
    return contour_tree

def generate_contour_tree(slices,augmented_contours,dir='up'):
    """Takes slices and the underlying contours and constructs a tree where every branch is a sorted list of contours
    that form a trivially connected subcomponent,i.e. no contour splits or merges are part of the component.
    Specify a direction to sort the branches in decending or ascending order of the contours levels.
    """

    U = nx.Graph() #Up Tree Graph
    D = nx.Graph() #Down Tree Graph

    U = build_graph(slices,U,direction='up')
    D = build_graph(slices,D,direction='down')

    J = nx.compose(D,U) #Naive Join Tree
    J = nx.minimum_spanning_tree(J)
    J = prune_join_tree(J,augmented_contours)

    S = nx.intersection(D,U) #Naive Split Tree
    S = prune_split_tree(S,J,augmented_contours)

    #the final nodal tree consists of the connected components of the final pruned split tree
    nodal_tree = list(nx.connected_components(S))

    #the nodes in each branch are sorted by level of the corresponding contour
    #the nodal tree is therefore converted to a list from a set
    nodal_tree = [sort_branch_by_level(list(branch),augmented_contours,dir=dir) for branch in nodal_tree]

    contour_tree = nodal_to_contour_tree(nodal_tree,augmented_contours,S)

    return contour_tree