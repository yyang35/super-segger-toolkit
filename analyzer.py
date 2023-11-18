import numpy as np
import pandas as pd



# this function is not useful after change approach from assignment matrix of labels to graph of 'Cell' class
# this function be delete once be proved useless
def compare_single_frame_matrix(matrix1, matrix2):
    # Determine the common shape
    common_shape = (min(matrix1.shape[0], matrix2.shape[0]), 
                    min(matrix1.shape[1], matrix2.shape[1]))

    # Slice both matrices to the common shape
    sub_matrix1 = matrix1[:common_shape[0], :common_shape[1]]
    sub_matrix2 = matrix2[:common_shape[0], :common_shape[1]]

    # Compare and count overlapping elements
    overlap_count = np.sum(sub_matrix1 * sub_matrix2)

     # Slice both matrices to the common shape
    sub_matrix1_inner = matrix1[1:common_shape[0], 1:common_shape[1]]
    sub_matrix2_inner = matrix2[1:common_shape[0], 1:common_shape[1]]

    # Compare and count overlapping elements
    overlap_count_without_born_death  = np.sum(sub_matrix1_inner * sub_matrix2_inner)
    
    return {"overlap": overlap_count,
            "left": np.sum(sub_matrix1), 
            "right": np.sum(sub_matrix2),  
            "overlap_pure_link": overlap_count_without_born_death,
            "left_pure_link": np.sum(sub_matrix1_inner),
            "right_pure_link": np.sum(sub_matrix2_inner),
            }

# this function is not useful after change approach from assignment matrix of labels to graph of 'Cell' class
# this function be delete once be proved useless
def compare_multiframe(dict1, dict2, only_link = True):
    assert len(dict1) == len(dict2), "Dimention of two assignment result dictionary not same "
    sorted_keys1 = sorted(dict1.keys())
    sorted_keys2 = sorted(dict2.keys())

    master_arr = np.zeros(len(dict1))

    for i in range(len(sorted_keys1)):
        assert isinstance(dict1[sorted_keys1[i]], np.ndarray)
        assert isinstance(dict2[sorted_keys2[i]], np.ndarray)
        result = compare_single_frame_matrix(dict1[sorted_keys1[i]], dict2[sorted_keys2[i]])
        if only_link:
            master_arr[i] = result["overlap"] * 1.0 / (result["left"] + result["right"] - result["overlap"])
        else:
            master_arr[i] = result["overlap_pure_link"] * 1.0 / (result["left_pure_link"] + result["right_pure_link"] - result["overlap_pure_link"])

    return master_arr

# this function is not useful after change approach from assignment matrix of labels to graph of 'Cell' class
# this function be delete once be proved useless
def compare_multiple_result(dict_of_multiple_assignment_dict, reference_dict):

    reference_length = len(reference_dict)

    master_dict = {}
    for key, value in dict_of_multiple_assignment_dict.items():
        # assert each assignment algo has same number frames
        assert len(value) == reference_length, f"Assignment {key} algo, have different number of frame than others" 
        master_dict[key] = compare_multiframe(value,reference_dict)

    return pd.DataFrame(master_dict)



