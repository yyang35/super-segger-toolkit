import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.colors import ListedColormap
from PIL import Image
import string
import cv2
import os
import matplotlib.patches as patches
import glob
from superseggertoolkit.link_composer import LinkComposer
import networkx as nx
import matplotlib.ticker as ticker

from cell_event import CellEvent, CellType, CellDefine


CELL_EVENT_COLOR = {
    CellEvent.SPLIT:"blue", 
    CellEvent.MERGE:"green", 
    CellEvent.DIE:"red", 
    CellEvent.BIRTH: "purple"
}

CELL_TYPE_COLOR = {
    CellType.REGULAR: "#878787", 
    CellType.SPLIT:"#1500FF", 
    CellType.SPLITED:"#756BE1", 
    CellType.MERGE:"#30FF00", 
    CellType.MERGED:"#8DE279", 
    CellType.DIE:"#FF0000", 
    CellType.BIRTH: "#FF00ED",
    CellType.UNKOWN: "#878787",
}


# ===================lineage related ==================================== #

# get each node lineage's position, get each node's location
# cells set input used for reasonable pos depending on specifc set, could use for      
def get_lineage_pos(G, cells = None):
    pos = {}
    cells = list(G.nodes()) if cells is None else cells
    cells.sort()

    left_pos = 0
    for cell in cells:
        if cell not in pos:
            make_pos(G, cell, pos, left_pos, 1)
            left_pos += 1
    return pos



# helper function for get_lineage_pos
# Deep Fist Search to label all horizontal position of each cell
def make_pos(G, node, pos, left_pos, width):
   pos[node] = (left_pos + width/2, -1 * node.frame)
   children_nodes =  list(G.successors(node))
   children_nodes.sort()
   if len(children_nodes) == 0 : return 
   slice_width = width / len(children_nodes)
   for i in range(len(children_nodes)):
      node = children_nodes[i]
      make_pos(G, node, pos, left_pos + i * slice_width, slice_width)



# for lineage, return a set of special edges and nodes, which used to shown overlap on normal lineage
def tag_type(G):
    tag_dict = {CellEvent.DIE: set(), CellEvent.BIRTH: set(), CellEvent.SPLIT: set(), CellEvent.MERGE:set()}     
    cells = set(G.nodes())
    for cell in cells:
        define = CellDefine(G, cell)
        # add special nodes:
        if define.die: tag_dict[CellEvent.DIE].add(cell)
        elif define.birth: tag_dict[CellEvent.BIRTH].add(cell)
        # add special edges:
        if define.merge:
            edges = G.in_edges(cell)
            tag_dict[CellEvent.MERGE].update(set(edges))
        if define.split:
            edges = G.out_edges(cell)
            tag_dict[CellEvent.SPLIT].update(set(edges))
    return tag_dict


# This plot lineage make some default highlight infomation on lineage: include cell events and basically statstic information
def quick_lineage(G):
    tag = tag_type(G)
    pos = get_lineage_pos(G)

    node_special = {CELL_EVENT_COLOR[CellEvent.BIRTH]: tag[CellEvent.BIRTH], CELL_EVENT_COLOR[CellEvent.DIE]: tag[CellEvent.DIE]}
    edge_special = {CELL_EVENT_COLOR[CellEvent.SPLIT]: tag[CellEvent.SPLIT], CELL_EVENT_COLOR[CellEvent.MERGE]: tag[CellEvent.MERGE]}

    plot_lineage(G, pos, with_background = True , nodes_special = node_special, edges_special = edge_special)


# 
def plot_lineage(G, pos, with_background, nodes_special = None, edges_special = None, show_stat = True, figsize = (15,12), arrow = False):
    fig, ax = plt.subplots(figsize=figsize)
    ax =  subplot_lineage(ax, G, pos, with_background, nodes_special, edges_special, show_stat, figsize, arrow)
    plt.show()


# plot lineage on a ax, this be factor out for any use of subplot, 
def subplot_lineage(ax, G, pos, with_background = False, nodes_special = None, edges_special = None, show_stat = True, figsize = (15,12), arrow = False):
    node_list = list(G.nodes())
    node_list.sort()

    # draw background
    if with_background: 
        nx.draw(G, pos, node_size = 0,  width=1, edge_color="grey", arrows = arrow, ax=ax)
    # draw special nodes, and edges that need be highlight
    if nodes_special != None:
        for color, nodes in nodes_special.items():
            nx.draw_networkx_nodes(G, pos, nodelist=list(nodes), node_size=20, node_color=color, ax=ax)
        for color, edges in edges_special.items():
            nx.draw_networkx_edges(G, pos, edgelist=edges , width=2, edge_color=color, arrows= arrow, ax=ax)
    # show statstic infomation 
    if show_stat:
        text = get_graph_stats_text(G)
        ax.text(0.95, 0.95,  text, transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')

    # styling below
    ax.set_frame_on(False)
    ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)

    def format_fn(tick_val, tick_pos):
        return f"frame {int(abs(tick_val))}" if tick_val % 1 == 0 else ""

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_fn))

    limits=plt.axis('on')
    ax.grid(True, linestyle='--', alpha=0.5)

    return ax




# function decide what to show on lineage slice for each frame
# extract all the node appear on this frame, get all edges/nodes it connected to
def get_single_frame_lineage_info(G, frame, tag = None):
    composer = LinkComposer(G.nodes())

    cells_center = composer.cells_frame_dict[frame]
    cell_s = {"red": cells_center}

    connected_edges = set()
    connected_nodes = set()

    for cell in cells_center:
        # Get incoming and outgoing edges for each cell
        in_edges = G.in_edges(cell)
        out_edges = G.out_edges(cell)

        # Update the sets of connected edges and nodes
        connected_edges.update(in_edges)
        connected_edges.update(out_edges)

        # Extract nodes from edges
        for edge in in_edges:
            connected_nodes.add(edge[0])  # Source node of the incoming edge
        for edge in out_edges:
            connected_nodes.add(edge[1])  # Target node of the outgoing edge
    
    edge_s = {"grey": connected_edges}
    cell_s = {"grey": connected_nodes, "blue": cells_center}

    if tag is not None:
        edge_split = tag[CellEvent.SPLIT].intersection(connected_edges)
        edge_merge = tag[CellEvent.MERGE].intersection(connected_edges)

        node_birth = tag[CellEvent.BIRTH].intersection(connected_nodes.union(cells_center))
        node_death = tag[CellEvent.DIE].intersection(connected_nodes.union(cells_center))

        edge_s[CELL_EVENT_COLOR[CellEvent.SPLIT]] = edge_split
        edge_s[CELL_EVENT_COLOR[CellEvent.MERGE]] = edge_merge

        cell_s[CELL_EVENT_COLOR[CellEvent.BIRTH]] = node_birth 
        cell_s[CELL_EVENT_COLOR[CellEvent.DIE]] = node_death

    pos = get_lineage_pos(G, list(connected_nodes.union(cells_center)))

    return  cell_s, edge_s, pos


# 
def get_graph_stats_text(G):

    composer = LinkComposer(G.nodes())
    max_frame = composer.frame_num - 1

    merge = 0
    split = 0
    birth = 0
    death = 0 
    ghost = 0 
    irregular_death = 0

    cells = set(G.nodes())

    for cell in cells:
        define = CellDefine(G,cell)
        merge += define.merge
        split += define.split
        birth += define.birth
        death += define.die
        ghost += define.ghost
        if define.die and cell.frame != max_frame:
            irregular_death += 1

    coverage_rate = (len(composer.cells) - stary) / len(composer.cells)
    frame_index = sorted(composer.cells_frame_dict)
    last_frame_cells = composer.cells_frame_dict[frame_index[-1]]
    cell_num = len(last_frame_cells)
    text = f"Max frame: {max_frame} \n Coverage rate : {coverage_rate:.0%} \n  last frame cell num:{cell_num} \n  Merge: {merge}, split: {split}, birth:{birth}, death: {death} \n  ghost: { ghost},irregular death: {irregular_death}"
    return text



# ======================== video related ==================================== #

# for images visualization, label the cell label and it's type 
# this cell label only for reability and represent relative relationship, no strict label be applied
def get_label_info(G, cells = None, alphabet_label = False):
    cells = list(G.nodes()) if cells == None else cells
    cells.sort()

    max_frame = cells[-1].frame 

    info = {}
    cell_id = 0

    if alphabet_label:
        alphabet = string.ascii_uppercase

    def get_new_label():
        nonlocal cell_id 
        label = alphabet[cell_id % len(alphabet)] if alphabet_label else cell_id
        cell_id = cell_id + 1
        return label

    for cell in cells:
        if cell not in info:
            define = CellDefine(G, cell)
            incoming = len(list(G.predecessors(cell)))
            # label birth/death, notice they are not conflict with merge/split
            # label birth/death first, so merge/split have higher priority, since will keep the label of latest asssigned
            if define.birth:
                label = get_new_label()
                info[cell] = (label, CellType.BIRTH)
            elif define.die:
                if cell.frame == max_frame:
                    info[cell] = (info[mother][0],CellType.UNKOWN)
                else:
                    info[cell] = (info[mother][0],CellType.DIE)

            # lable all other events. regular(1 to 1), split, and merge are parallel structure. 
            if incoming > 0:
                mother = list(G.predecessors(cell))[0]
                if define.regular: 
                    info[cell] = (info[mother][0], CellType.REGULAR)
                elif define.split:
                    # label itself 
                    info[cell] = (info[mother][0], CellType.SPLIT)
                    # label it's outgoing cells 
                    cell_list = list(G.successors(cell))
                    cell_list.sort()
                    for i in range(len(cell_list)):
                        outgoing_cell = cell_list[i]
                        # give a label 
                        label = get_new_label() if not alphabet_label else info[mother][0] + str(i)
                        info[outgoing_cell] = (label,CellType.SPLITED)
                elif define.merge:
                    # label itself
                    label = get_new_label()
                    info[cell] = (label, CellType.MERGE)
                    # label it's incoming cells 
                    cell_list = list(G.predecessors(cell))
                    cell_list.sort()
                    for i in range(len(cell_list)):
                        income_cell = cell_list[i]
                        assert income_cell in info, "the cell frame before current cell haven't be labeled"
                        label = info[cell][0]+ str(i) if alphabet_label else info[income_cell][0]
                        info[income_cell] = (label,CellType.MERGED)         

    return info



# Master function of draw polygon and cell id label on raw image
def get_single_track_visualization(G,base_dir, info, circle_label = False, representative_point = False):
    video_dir = os.path.join(base_dir, "video")
    os.makedirs(video_dir, exist_ok=True)

    composer = LinkComposer(set(G.nodes()))
    # found all initial images.
    filename = "*.tif"
    file_path = os.path.join(base_dir, filename)

    files = glob.glob(file_path)
    files.sort()

    for frame in range(len(files)):
        # Load and convert the image
        file = files[frame]
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        output_path = os.path.join(video_dir, f"frame{frame:05d}.png")
        # Create a new figure and axis
        fig, ax = plt.subplots()

        ax = get_single_frame_visualization(ax, image, composer.cells_frame_dict, info, frame, circle_label, representative_point)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)

    image_to_tif_sequence( video_dir )
        


# Draw polygon and cell id label on each raw image 
def get_single_frame_visualization(ax, image, cells_frame_dict, info, frame, circle_label = False, representative_point = False ):
    # Display the image
    ax.imshow(image)

    for cell in cells_frame_dict[frame]:
        color_dict = CELL_TYPE_COLOR
        if cell in info:
            # Ensure this is a list of (x, y) tuples
            polygon_vertices = cell.polygon.exterior.coords

            color = color_dict[info[cell][1]]
            label = info[cell][0]

            # Half transparent
            facecolor = color + "60"

            # Create a polygon patch
            polygon = patches.Polygon(polygon_vertices, closed=True, edgecolor=color, facecolor=facecolor)

            # Add the polygon patch to the axis
            ax.add_patch(polygon)

            if representative_point: 
                 # this is not center of cell, but a point that guaranteed in polygon
                centroid_x = cell.polygon.representative_point().x
                centroid_y = cell.polygon.representative_point().x
            else:
                # Add text annotation
                centroid_x = cell.polygon.centroid.x
                centroid_y = cell.polygon.centroid.y

            if circle_label:
                ax.text(centroid_x, centroid_y, str(label), color='black', fontweight='bold', bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle'), fontsize=5, horizontalalignment='center', verticalalignment='center')
            else:
                ax.text(centroid_x, centroid_y, str(label), color='white', fontweight='bold', fontsize=5, horizontalalignment='center', verticalalignment='center')

    # Optionally, set the axis limits based on the image size
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)

    ax.set_axis_off()
    ax.set_frame_on(False)

    return ax 


def image_to_tif_sequence(images_folder):
    tif_images = [f for f in os.listdir(images_folder) if f.lower().endswith('.png')]
    tif_images.sort()

    # Output TIF "video" filename
    output_tif_sequence = images_folder  + 'time_sequences.tif'

    images = []

    for image_filename in tif_images:
        image = Image.open(os.path.join(images_folder,image_filename))
        images.append(image)

    # Save the sequence as a multi-page TIF file
    images[0].save(
        output_tif_sequence,
        save_all=True,
        append_images=images[1:],
        resolution=100.0,  # Set the resolution (DPI)
        compression='tiff_lzw'  # Set the compression method
    )

    print("TIF sequence created successfully.")



# ======================== plot error mask related ====================== #
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



