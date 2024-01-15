from .link_composer import LinkComposer


def silly_linker(composer: LinkComposer):
    G = composer.make_new_dircted_graph()
    dict = composer.cells_frame_dict
    sorted_frame = sorted(dict)

    for i in range(1, len(sorted_frame)):
        source_frame = dict[sorted_frame[i-1]]
        target_frame =  dict[sorted_frame[i]]
        for cell in target_frame:
            max_IoU = 0
            best_candidate = None
            for cell_candicate in source_frame:
                intersect = cell.polygon.intersection(cell_candicate.polygon).area 
                union = cell.polygon.union(cell_candicate.polygon).area 
                IoU = intersect * 1.0 / union
                if IoU  > max_IoU:
                    max_IoU  = IoU
                    best_candidate = cell_candicate
            if best_candidate is not None:
                G.add_edge(best_candidate, cell)

        for cell in source_frame:
            if G.out_degree(cell) == 0:
                max_IoU = 0
                best_candidate = None
                for cell_candicate in target_frame:
                    intersect = cell.polygon.intersection(cell_candicate.polygon).area 
                    union = cell.polygon.union(cell_candicate.polygon).area 
                    IoU = intersect * 1.0 / union
                    if IoU  > max_IoU:
                        max_IoU  = IoU
                        best_candidate = cell_candicate
                if best_candidate is not None:
                    G.add_edge(cell, best_candidate)

    return G