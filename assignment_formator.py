import pandas as pd
from scipy.io import savemat
import numpy as np
import glob
import os
import re
from scipy.io import loadmat
import warnings

from cells_extractor import get_polygon_2d_dict
from cells_extractor import get_cell_info_by_ploygon


def get_manual_link_dict(excel_path):
    assert excel_path.endswith('.xlsx'), "File must be an Excel file with a .xlsx extension"

    excel_file = pd.ExcelFile(excel_path)
    sheet_names = excel_file.sheet_names

    manual_link_master_dict = {}
    for time_sheet in sheet_names:
        df = excel_file.parse(time_sheet)
        df = df.astype(str)
        [source, target] = df.columns

        target_max_id = 0
        source_max_id = 0

        time_dict = {}
       
        for index, row in df.iterrows():
            try:
                mother = int(row[source])
            except ValueError:
                mother = 0
            daughter = row[target].lower()
            if 'x' in daughter:
                cell_list = [0]
            else:
                cell_list = daughter.split(',')

            time_dict[mother] = {int(x) for x in cell_list}
            target_max_id = max(np.max(list(time_dict[mother])), target_max_id)
            source_max_id = max(mother, source_max_id)

        manual_link_master_dict[time_sheet] = linking_dict_to_linking_matrix(time_dict)

    return manual_link_master_dict


def get_supersegger_file_info_and_tracker_result_new(foldername):
    npzFiles = glob.glob(foldername)
    supperSegger_dict = {}

    for f in npzFiles:
        pattern = r't(\d+)'
        match = re.search(pattern, f)

        if match:
            time_value = int(match.group(1)) 
            frame_index = f"t{time_value:05d}"
        else:
            raise ValueError('No time prefix on filename.')

        data = loadmat(f)
        label = data['regs']['regs_label'][0][0]
        track_result = data['regs']['map'][0][0]['f'][0][0][0]
        
        track_dict = {}
        for i in range(len(track_result)):
            if len(track_result[i][0]) > 0:
               track_dict[i+1] = set(track_result[i][0])
        
        #Last frame don't have linking array
        if len(track_dict) == 0 :
            continue

        supperSegger_dict[frame_index] = linking_dict_to_linking_matrix(track_dict)

    return supperSegger_dict


def linking_dict_to_linking_matrix(dict):
    source_max_id = np.max(list(dict.keys()))
    flattened_values = [item for value_set in dict.values() for item in value_set]
    target_max_id = max(flattened_values)

    matrix = np.zeros((source_max_id+1, target_max_id+1))

    for key, value in dict.items():
        for item in value:
            matrix[key][item] = 1
    
    return matrix


def match_trackmate_cell_id_to_mask_label(spots_filename, mask_foldername, UNIT_CONVERT_COEFF = 1):

    # Read top 4 line as header by trackmate dataformat
    spots = pd.read_csv(spots_filename, header=[0, 1, 2, 3])
    # Only use the first row header for convenient
    spots.columns = spots.columns.get_level_values(0)

    mask_polygons_dict = get_polygon_2d_dict(mask_foldername)
    assert len(mask_polygons_dict) == np.max(spots["FRAME"])+1, "The number of masks and trackmate frame number is inconsist,  contents of folders don't match up"
    sorted_polygon_frame_key = sorted(mask_polygons_dict)

    trackmate_frame_index = spots["FRAME"].unique()
    trackmate_frame_index.sort()

    label_correlated_dic = {}

    for frame in trackmate_frame_index:
        cells_in_frame = spots[spots["FRAME"] == frame]
        mask_filename_time_index = sorted_polygon_frame_key[frame]
        mask_cells = get_cell_info_by_ploygon(mask_polygons_dict[ mask_filename_time_index ])
        for index, row in cells_in_frame.iterrows():
            # Trackmate using timeframe start from 0
            cell_id = row["ID"]
            # Trackmate use different unit of position, check trackmate document / tracks excel 
            trackmate_x = row["POSITION_X"]*UNIT_CONVERT_COEFF
            trackmate_y = row["POSITION_Y"]*UNIT_CONVERT_COEFF

            candidates =  mask_cells.loc[ ((abs( mask_cells['x_mean'] - trackmate_x)  + abs( mask_cells['y_mean'] - trackmate_y)) < 2 ) ]

            # if matched candidates is not prefect 1 to 1, means the matching result could have some uncertainty, notify
            if len(candidates) == 0 or len(candidates) > 1:
                if len(candidates) == 0:
                    warnings.warn(f"Trackmate cell:{cell_id} match back to mask is inaccute, matching to the nearest cell!")
                else:
                    warnings.warn(f"Trackmate cell:{cell_id} around by dense cells, multiple candidates, matching to the nearest cell!")
                min_value = min(( mask_cells['x_mean'] - trackmate_x)**2 + ( mask_cells['y_mean'] - trackmate_y)**2)
                # Use boolean indexing to filter rows where the expression equals the minimum value
                label =  mask_cells.loc[(( mask_cells['x_mean'] - trackmate_x)**2 + ( mask_cells['y_mean'] - trackmate_y) ** 2) == min_value].iloc[0]['label']
            else:
                label = candidates.iloc[0]['label']

            spots.loc[index, "mask_label"] = int(label)
            spots.loc[index, "supersegger_time_index"] = mask_filename_time_index

    return spots


def abstract_trackmate_linking_result(spots_filename, edge_filename, mask_foldername):
    spots = match_trackmate_cell_id_to_mask_label(spots_filename, mask_foldername)
    track_dict =  abstact_tackmate_single_frame_assignment_matrix(spots, edge_filename, mask_foldername)
    return track_dict
    

def abstact_tackmate_single_frame_assignment_matrix(spots_df, edge_filename, mask_foldername):
    # Read top 4 line as header by trackmate dataformat
    tracks = pd.read_csv(edge_filename, header=[0, 1, 2, 3])
    # Only use the first row header for convenient
    tracks.columns =  tracks.columns.get_level_values(0)

    trackmate_track_index = tracks["EDGE_TIME"].unique()
    trackmate_track_index.sort()

    spots_reduced = spots_df[['ID', 'mask_label']]
    assert spots_reduced.isna().sum().sum() == 0
    spots_reduced['mask_label'] = pd.to_numeric(spots_reduced['mask_label'], errors='coerce').fillna(0).astype(int)

    tracks['mask_label'] = "" # for later suffixes convenient

    tracks = tracks.merge(
        spots_reduced[['ID', 'mask_label']],
        left_on='SPOT_SOURCE_ID',
        right_on='ID',
        suffixes=('', '_source')
    ).drop('ID', axis=1)

    # Second join on 'SPOT_TARGET_ID'
    tracks = tracks.merge(
        spots_reduced[['ID', 'mask_label']],
        left_on='SPOT_TARGET_ID',
        right_on='ID',
        suffixes=('', '_target')
    ).drop('ID', axis=1)

    tracks = tracks.drop('mask_label', axis=1)

    trackmate_linking_dict = {}
    for index in trackmate_track_index:
        frame_linking = tracks[tracks["EDGE_TIME"] == index]
        assignment_matrix = np.zeros((np.max(frame_linking["mask_label_source"]) + 1, np.max(frame_linking["mask_label_target"]) + 1))
        for key, row in frame_linking.iterrows():
            assignment_matrix[row["mask_label_source"]][row["mask_label_target"]] = 1
        trackmate_linking_dict[index] = assignment_matrix

    return trackmate_linking_dict

print("Loaded assignment_formator, contains:", dir())

