from cell import Cell
import pandas as pd
import numpy as np
import glob
import os
import re
import warnings
import networkx as nx
from typing import Set
import sys


class LinkComposer:



    def __init__(self, cells: Set[Cell]):
        self.cells = cells
        self.cells_frame_dict = self.get_cells_frame_dict(cells)
        self.frame_num = len(self.cells_frame_dict)



    def get_cells_frame_dict(self, cells: Set[Cell]) -> dict:
        cells_frame_dict = {}
        for cell in cells:
            if cell.frame not in cells_frame_dict:
                cells_frame_dict[cell.frame] = {cell}
            else:
                cells_frame_dict[cell.frame].add(cell)

        return cells_frame_dict
    


    def make_new_dircted_graph(self):
        G = nx.DiGraph()
        for cell in self.cells:
            G.add_node(cell)

        return G


    
    def link(self, G, cell1, cell2):
        assert cell1 in self.cells, "source cell not in cells"
        assert cell2 in self.cells, "target cell not in cells"
        G.add_edge(cell1, cell2)



    def get_manual_link_dict(self, excel_path):
        assert excel_path.endswith('.xlsx'), "File must be an Excel file with a .xlsx extension"

        G = self.make_new_dircted_graph()

        excel_file = pd.ExcelFile(excel_path)
        sheet_names = excel_file.sheet_names

        assert self.frame_num == len(sheet_names) + 1, "Linking frames count not same with masks"

        for frame_start in range(sheet_names):
            frame_end = frame_start + 1

            time_sheet = sheet_names[frame_start]
            df = excel_file.parse(time_sheet)
            df = df.astype(str)
            [source, target] = df.columns
            for index, row in df.iterrows():
                try:
                    mother = int(row[source])
                except ValueError:
                    # new born will be automatically handled by graph
                    continue
                daughter = row[target].lower()
                if 'x' in daughter:
                    # death  will be automatically handled by graph
                    continue
                else:
                    cell_list = daughter.split(',')
                
                assert Cell(frame_start, mother) in G.nodes()
                for cell in cell_list:
                    assert Cell(frame_end,cell) in G.nodes()
                    G.add_edge(Cell(frame_start, mother), Cell(frame_end,cell))

        return G



    def get_supersegger_file_info_and_tracker_result(self,foldername):
        
        G = self.make_new_dircted_graph()

        npzFiles = glob.glob(foldername)
        npzFiles.sort()
        assert self.frame_num == len(npzFiles), "Linking frames count not same with masks"
        # Don't need last file, there no linking 
        npzFiles.pop()
    
        for frame_start in npzFiles:
            f = npzFiles[frame_start]
            frame_end = frame_start + 1
            data = loadmat(f)
            label = data['regs']['regs_label'][0][0]
            track_result = data['regs']['map'][0][0]['f'][0][0][0]
            
            track_dict = {}
            for i in range(len(track_result)):
                if len(track_result[i][0]) > 0:
                    # Matlab index start at 1, but python start with 0
                    assert(Cell(frame_start, i+1) in G)
                    target_cells = set(track_result[i][0])

                    for target_cell in target_cells:
                        assert(Cell(frame_end, target_cell) in G)
                        G.add_edge(Cell(frame_start, i+1), Cell(frame_end,target_cell))

        return G
    


    def get_trackmate_linking_result(self,spots_filename, edge_filename,  UNIT_CONVERT_COEFF = 1):
        spots = self._match_trackmate_cell_id_to_mask_label(spots_filename,  UNIT_CONVERT_COEFF =  UNIT_CONVERT_COEFF )
        G =  self._abstact_tackmate_assignment_by_edges_file(spots, edge_filename)
        return G
    


    def private_method(func):
        def wrapper(*args, **kwargs):
            print(f"Warning: {func.__name__} is a private method and should not be accessed directly.")
            return func(*args, **kwargs)
        return wrapper



    @private_method
    def _match_trackmate_cell_id_to_mask_label(self, spots_filename, UNIT_CONVERT_COEFF = 1):

        # Read top 4 line as header by trackmate dataformat
        spots = pd.read_csv(spots_filename, header=[0, 1, 2, 3])
        # Only use the first row header for convenient
        spots.columns = spots.columns.get_level_values(0)

        # trackmate also have frame start with 0
        assert self.frame_num == np.max(spots["FRAME"])+1, "The number of masks and trackmate frame number is inconsist,  contents of folders don't match up"

        trackmate_frame_index = spots["FRAME"].unique()
        trackmate_frame_index.sort()

        for frame_index in range(len(trackmate_frame_index)):
            frame = trackmate_frame_index[frame_index]
            cells_in_frame = spots[spots["FRAME"] == frame]
            cells_in_mask = self.cells_frame_dict[frame_index]

            for index, row in cells_in_frame.iterrows():
                # Trackmate using timeframe start from 0
                cell_id = row["ID"]
                # Trackmate use different unit of position, check trackmate document / tracks excel 
                trackmate_x = row["POSITION_X"]*UNIT_CONVERT_COEFF
                trackmate_y = row["POSITION_Y"]*UNIT_CONVERT_COEFF

                shorest_distance = sys.maxsize
                matched_cell = None
                total_good_candidates = 0

                for cell in cells_in_mask:
                    distance = (cell.polygon.centroid.x - trackmate_x) ** 2 + (cell.polygon.centroid.y - trackmate_y) ** 2 
                    if distance < 2 : total_good_candidates += 1 
                    if distance < shorest_distance:
                        shorest_distance = distance
                        matched_cell = cell

                if total_good_candidates == 0:
                    warnings.warn(f"Trackmate cell:{cell_id} match back to mask is inaccute, assigned to the nearest cell.")
                if total_good_candidates > 1:
                    warnings.warn(f"Trackmate cell:{cell_id} around by dense cells, multiple candidates, assigned to the nearest cell.")

                spots.loc[index, "cell"] = matched_cell

        return spots
        

    @private_method
    def _abstact_tackmate_assignment_by_edges_file(self, spots_df, edge_filename):
        # Read top 4 line as header by trackmate dataformat
        tracks = pd.read_csv(edge_filename, header=[0, 1, 2, 3])
        # Only use the first row header for convenient
        tracks.columns =  tracks.columns.get_level_values(0)

        G = self.make_new_dircted_graph()

        spots_reduced = spots_df[['ID', 'cell']]
        assert spots_reduced.isna().sum().sum() == 0

        tracks['cell'] = "" # for later suffixes convenient

        tracks = tracks.merge(
            spots_reduced[['ID', 'cell']],
            left_on='SPOT_SOURCE_ID',
            right_on='ID',
            suffixes=('', '_source')
        ).drop('ID', axis=1)

        # Second join on 'SPOT_TARGET_ID'
        tracks = tracks.merge(
            spots_reduced[['ID', 'cell']],
            left_on='SPOT_TARGET_ID',
            right_on='ID',
            suffixes=('', '_target')
        ).drop('ID', axis=1)

        tracks = tracks.drop('cell', axis=1)

        for index, row in tracks.iterrows():
            G.add_edge(row["cell_source"], row["cell_target"])

        return G

