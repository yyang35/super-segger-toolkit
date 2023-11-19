import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import re
import cv2
from shapely.geometry import Polygon
from matplotlib.patches import Polygon as pltPolygon
import pandas as pd
from natsort import natsorted
import warnings

from PIL import Image
import numpy as np


from cell import Cell



# Mask reader which can only read supersegger-omnipose produced mask. It use time frame with prefix "t" to sort frame
def get_omnipose_mask_dict(foldername: str):
    mask_dict = {}
    npzFiles = glob.glob(foldername)
    for filename in npzFiles:
        img = Image.open(filename)
        img = img.convert('L')
        mask = np.array(img) 

        pattern = r't(\d+)'
        match = re.search(pattern, filename)

        if match:
            t_value = match.group(1)
            t_with_prefix = 't' + t_value
        else:
            raise Exception("No time prefix match found.")
        
        mask_dict[t_with_prefix] = mask

    return mask_dict



# general mask reader which sort file by nature order
# check natsorted for more info 
def get_mask_dict(foldername: str):
    filenames = glob.glob(foldername)
    # Using nature storted name here, for let name like 't1' < 't10' 
    sorted_filenames = natsorted(filenames)
    mask_dict = {}
    for i in range(len(sorted_filenames)):
        filename = sorted_filenames[i]

        img = Image.open(filename)
        img = img.convert('L')
        mask = np.array(img) 

        mask_dict[i] = mask

    return mask_dict


def get_folder_info(foldername: str):
    filenames = glob.glob(foldername)
    assert len(filenames) > 0, f"no image in {foldername}"
    # Using nature storted name here, for let name like 't1' < 't10' 
    frame_count = len(filenames)
    img = Image.open(filenames[0])
    first_frame_shape = np.array(img).shape

    return frame_count, first_frame_shape


def read_tiff_sequence(filename):
    mask_dict = {}
    with Image.open(filename) as img:
        i = 0
        while True:
            # Convert image to grayscale ('L')
            img.seek(i)
            img_converted = img.convert('L')
            mask = np.array(img_converted)

            mask_dict[i] = mask

            i += 1
            try:
                img.seek(i)
            except EOFError:
                # Exit loop when end of file is reached
                break

    return mask_dict


def get_tiff_info(filename):
    with Image.open(filename) as img:
        # Get the shape of the first frame
        img.seek(0)
        first_frame_shape = np.array(img).shape
        
        # Count the total number of frames
        frame_count = 1
        while True:
            try:
                img.seek(frame_count)
                frame_count += 1
            except EOFError:
                break

    return frame_count, first_frame_shape



def get_cells_set_by_mask_dict(mask_dict, force = False):
    cells_set = set()
    frame_keys = sorted(mask_dict)
    error = []
    for frame_index in range(len(frame_keys)):
        mask = mask_dict[frame_keys[frame_index]]
        # start from label = 1, label = 0 is background
        for mask_label in range(1,np.max(mask)+1):
            n_pixels = np.sum(mask == mask_label)
            if n_pixels > 0:
                try:
                    cell_mask = mask == mask_label
                    polygon = single_cell_mask_to_polygon(cell_mask)
                    cells_set.add(Cell(frame = frame_index, label = mask_label, polygon = polygon))
                except AssertionError as e:
                    if force:
                        error.append([frame_index, mask_label])
                        warnings.warn(f"{e}")
                    else:
                        raise
                except Exception as e:
                    # this is for some dirty manually changed mask, some un-noticed little picels might be not earsed
                    error.append([frame_index, mask_label])
                    print(f"Frame:{frame_index}, Mask label:{mask_label}. Pixels number = {n_pixels}. cannot make polygon. {e}")
    
    return cells_set, error



def single_cell_mask_to_polygon(cell_mask):
    # convert 1,0 binary mask to 255,0 mask, thus it's easier for extract polygon
    cell_mask = ( cell_mask  * 255).astype(np.uint8)
    contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [Polygon(contour.reshape(-1, 2)) for contour in contours]
    assert len(polygons) == 1 , "Disconnected multi-pieces found on single mask/cell label"
    return polygons[0]
