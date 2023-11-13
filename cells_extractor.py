import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import re
import cv2
from shapely.geometry import Polygon
from matplotlib.patches import Polygon as pltPolygon
import pandas as pd


def get_omnipose_mask_dict(filename: str):
    mask_dict = {}
    npzFiles = glob.glob(filename)
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

def get_polygon_2d_dict(filename: str, threshold = 5):
    mask_dict = get_omnipose_mask_dict(filename)
    polygons_dict = {}
    for key, mask in mask_dict.items():
        polygons_dict[key]= mask_matrix_to_ploygon_list(mask, threshold=threshold)

    return polygons_dict


def mask_matrix_to_ploygon_list(mask, threshold = 0):
    frame_dict={}
    for i in range(1,np.max(mask)+1):
        if np.sum(mask == i) > threshold:
            cell_mask = mask == i 
            cell_mask = ( cell_mask  * 255).astype(np.uint8)
            contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons = [Polygon(contour.reshape(-1, 2)) for contour in contours]
            assert len(polygons) == 1 , "Disconnected multi-pieces found on single mask/cell label"
            frame_dict[i] = polygons[0]
    
    return frame_dict


def get_cell_info_by_mask(labels_mask):
    label_info_df = pd.DataFrame(columns = ['label', 'x_mean', 'y_mean', 'area'])
    regs_label = labels_mask
    for i in range(1,np.max(regs_label)+1):
        row_indices, col_indices = np.where(regs_label == i)
        if len(row_indices) > 0:
            new_row = {'label': i, 'x_mean': np.average(col_indices), 'y_mean': np.average(row_indices), 'area': len(row_indices)}
            label_info_df.loc[len(label_info_df)] = new_row

    return label_info_df


def get_cell_info_by_ploygon(frame_polygons_dict):
    label_info_df = pd.DataFrame(columns = ['label', 'x_mean', 'y_mean', 'area'])
    for key, value in frame_polygons_dict.items():
        # value should in Shaply polygon class
        new_row = {'label': key, 'x_mean': value.centroid.x, 'y_mean': value.centroid.y, 'area':value.area}
        label_info_df.loc[len(label_info_df)] = new_row

    return label_info_df
