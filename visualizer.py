import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.colors import ListedColormap




def get_lineage_pos(G):
    pos = {}
    cells = list(G.nodes())
    cells.sort()

    left_pos = 0
    for cell in cells:
        if cell not in pos:
            make_pos(G, cell, pos, left_pos, 1)
            left_pos += 1
    return pos


def make_pos(G, node, pos, left_pos, width):
   pos[node] = (left_pos + width/2, -1 * node.frame)
   children_nodes =  list(G.successors(node))
   children_nodes.sort()
   if len(children_nodes) == 0 : return 
   slice_width = width / len(children_nodes)
   for i in range(len(children_nodes)):
      node = children_nodes[i]
      make_pos(G, node, pos, left_pos + i * slice_width, slice_width)



def tag_type(G):
    tag_dict = {"death": set(), "birth": set(), "split": set(), "merge":set()}     
    cells = G.nodes()
    for cell in cells:
        outgoing = len(list(G.successors(cell)))
        incoming = len(list(G.predecessors(cell)))
        if outgoing == 0 and incoming > 0:
            tag_dict["death"].add(cell)
        elif incoming == 0 and outgoing > 0:
            tag_dict["birth"].add(cell)
        
        if incoming > 1:
            # merge
            edges = G.in_edges(cell)
            tag_dict["merge"].update(set(edges))
        if outgoing > 1 :
            # split
            edges = G.out_edges(cell)
            tag_dict["split"].update(set(edges))
    return tag_dict



def mark_cell_style(G,tag_dict):
    
    node_colors = {}
    node_sizes = {}
    edge_colors = {}

    # Make default node style
    for cell in G.nodes():
        node_colors[cell] = "black"
        node_sizes[cell] = 0
    # Make default edge style
    for edge in G.edges():
        edge_colors[edge] = "grey"
    # Make special cell style
    for cell in tag_dict["death"]:
        node_colors[cell] = "grey"
        node_sizes[cell] = 20
    for cell in tag_dict["birth"]:
        node_colors[cell] = "blue"
        node_sizes[cell] = 20
    # Make special edge style
    for edge in tag_dict["merge"]:
        edge_colors[edge] = "red"
    for edge in tag_dict["split"]:   
        edge_colors[edge] = "blue"

    return node_colors, node_sizes, edge_colors

# Rest are pure visiualize related code, with styling stuff 

def crop_to_square_bounding_box(mask, label_value, padding=0):
    # Find the coordinates of the mask where the value equals label_value
    y_indices, x_indices = np.where(mask == label_value)
    
    # If no such value is found, return None
    if not len(y_indices) or not len(x_indices):
        return None
    
    # Determine the bounding box
    y_min, x_min = y_indices.min(), x_indices.min()
    y_max, x_max = y_indices.max(), x_indices.max()
    
    # Compute the current width and height of the bounding box
    current_width = x_max - x_min
    current_height = y_max - y_min
    
    # Determine the size of the square (the max of width and height)
    square_size = max(current_width, current_height) + 2 * padding
    
    # Calculate the center of the old bounding box
    center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
    
    # Determine the new square bounding box coordinates
    x_min = max(center_x - square_size // 2, 0)
    y_min = max(center_y - square_size // 2, 0)
    x_max = x_min + square_size
    y_max = y_min + square_size
    
    # Ensure the bounding box is within the bounds of the mask
    x_min = max(0, min(x_min, mask.shape[1] - square_size))
    y_min = max(0, min(y_min, mask.shape[0] - square_size))
    x_max = min(mask.shape[1], x_min + square_size)
    y_max = min(mask.shape[0], y_min + square_size)
    
    # Crop the mask to this square bounding box
    cropped_mask = mask[y_min:y_max, x_min:x_max]
    
    return cropped_mask


def plot_error_masks(mask, error):
    num_images = len(error)  
    num_rows = math.ceil(num_images / 5)

    # Create a large figure to hold all subplots
    fig, axs = plt.subplots(num_rows, 5, figsize=(15, 3 * num_rows))

    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    cmap = ListedColormap([(0.9, 0.9, 1, 0.15), (1, 0, 0, 1)])
    # Loop through images and plot them
    for idx, item in enumerate(error):
        frame_index = item[0]
        mask_label = item[1]
        cropped_mask = crop_to_square_bounding_box(mask[frame_index], mask_label, padding=10) 
        axs[idx].imshow(cropped_mask, cmap='gray')
        axs[idx].imshow(cropped_mask== item[1], cmap=cmap)
        axs[idx].set_title(f"Frame: {frame_index}, Mask Label: {mask_label}")

    # Turn off any unused subplots
    for ax in axs[num_images:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()



