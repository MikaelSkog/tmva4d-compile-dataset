import glob
import os
import random

import numpy as np
import open3d as o3d


TRAIN_SET_SPLIT = 0.5  # Ratio of the training set split.
VAL_SET_SPLIT   = 0.25 # Ratio of the validation set split.
# The rest is assigned to the test set.

# Get the minimums and maximums for the range, power, and Doppler values as a dictionary.
# 'pcd_dir' specifies the path to the directory containing subdirectories each representing
# a sequence, and containing PCD files.
def get_pcd_stats(pcd_dir):
    min_range = float('inf')
    max_range = float('-inf')
    min_power = float('inf')
    max_power = float('-inf')
    min_doppler = float('inf')
    max_doppler = float('-inf')

    # Get the sequences in the point cloud input directory.
    sequences = [d for d in os.listdir(pcd_dir) if os.path.isdir(os.path.join(pcd_dir, d))]

    # For each sequence:
    for seq in sequences:
        pcd_file_paths = glob.glob(os.path.join(pcd_dir, seq, '*.pcd'))
        # For each point cloud in the sequence directory:
        for pcd_file_path in pcd_file_paths:
            # Read the point cloud and get its points (x, y, and z for each point).
            point_cloud = o3d.io.read_point_cloud(pcd_file_path)
            points = np.asarray(point_cloud.points) # x (azimuth), y (elevation), z(range).
            
            range_values = points[:, 2] # Read from the 'z' field.

            # Read the PCD file and extract power and doppler values for each point.
            power_values = []
            doppler_values = []
            with open(pcd_file_path, 'r') as f:
                lines = f.readlines()
                flag = False
                # Get all data lines.
                for line in lines:
                    if line.startswith("DATA"):
                        flag = True
                        continue
                    if flag:
                        # Get the power and doppler values for the points of this point cloud.
                        values = line.split()
                        power = float(values[3])  # Assuming the power value of the point is at in field of index 3.
                        power_values.append(power)
                        doppler = float(values[4])  # Assuming the Doppler value of the point is in the field of index 4.
                        doppler_values.append(doppler)

            # Update the minimum and maximum range values (range==0 indicates invalid point, thus disregard these points).
            min_range = np.min(np.append(range_values[range_values != 0], min_range))
            max_range = np.max(np.append(range_values[range_values != 0], max_range))

            # Update the minimum and maximum power values (power==0 indicates invalid point, thus disregard these points).
            min_power = np.min(np.append(power_values[power_values != 0], min_power))
            max_power = np.max(np.append(power_values[power_values != 0], max_power))

            # Update the minimum and maximum Doppler values.
            min_doppler = np.min(np.append(doppler_values, min_doppler))
            max_doppler = np.max(np.append(doppler_values, max_doppler))
    
    pcd_stats = {
        'range'  : {'min': min_range,   'max': max_range},
        'power'  : {'min': min_power,   'max': max_power},
        'doppler': {'min': min_doppler, 'max': max_doppler}
    }
    return pcd_stats

# Get the minimum, maximum, mean and standard devation for the power values in a specified view,
# across the whole dataset (all instances) as a dictionary.
def get_stats_for_view(dataset_dir, view_subdir):
    # Get the sequences in the dataset directory.
    sequences = [d for d in os.listdir(dataset_dir) if (os.path.isdir(os.path.join(dataset_dir, d)) and d != 'samples')]

    # NumPy array that will end up being all instances of the view concatenated.
    all_instances_of_view = None

    # For each sequence:
    for seq in sequences:
        # Get the file paths to all NumPy array files for instances of the specified view in the current
        # sequence.
        dir_with_instances_for_view = os.path.join(dataset_dir, seq, view_subdir)
        npy_file_paths = glob.glob(os.path.join(dir_with_instances_for_view, '*.npy'))

        # Load the NumPy arrays and then concatenate them into one single, large NumPy array.
        numpy_arrays = [np.load(path) for path in npy_file_paths]

        # Initialize the NumPy array if not yet done.
        if all_instances_of_view is None:
            all_instances_of_view = np.concatenate(numpy_arrays, axis=0)
        # Else, simply concatenate all NumPy arrays for this view and sequence to the NumPy array for
        # all instances of this view.
        all_instances_of_view = np.concatenate((all_instances_of_view, *numpy_arrays), axis=0)

    stats_for_view = {
        'mean'   : np.mean(all_instances_of_view),
        'std'    : np.std(all_instances_of_view),
        'min_val': np.min(all_instances_of_view),
        'max_val': np.max(all_instances_of_view)
    }
    return stats_for_view

# Get the weights of each class ('background' and 'person'), based on their inverted prevalence in the
# masks, as a dictionary.
def get_weights_for_ea_view(dataset_dir):
    # Get the sequences in the dataset directory.
    sequences = [d for d in os.listdir(dataset_dir) if (os.path.isdir(os.path.join(dataset_dir, d)) and d != 'samples')]

    # NumPy array that will end up being all masks of the 'person' class for elevation-azimuth view concatenated.
    all_ea_masks_person = None

    # For each sequence:
    for seq in sequences:
        # Get the file paths to all NumPy array files for all masks in the elevation-azimuth view in the
        # current sequence.
        dir_with_ea_masks = os.path.join(dataset_dir, seq, 'annotations', 'dense')
        npy_file_paths = glob.glob(os.path.join(dir_with_ea_masks, '*', 'elevation_azimuth.npy'))

        # Load the masks in this sequence corresponding to the 'person' class ([1]).
        numpy_arrays = [np.load(path)[1] for path in npy_file_paths]

        # Initialize the NumPy array if not yet done.
        if all_ea_masks_person is None:
            all_ea_masks_person = np.concatenate(numpy_arrays, axis=0)
        # Else, simply concatenate all NumPy arrays for this sequence to the NumPy array for
        # all instances of this view.
        all_ea_masks_person = np.concatenate((all_ea_masks_person, *numpy_arrays), axis=0)

    total_num_of_elements = np.size(all_ea_masks_person) # Number of all elements across all elevation-azimuth masks.
    num_person_elements = np.sum(all_ea_masks_person == 1) # Number of 'person' elements across all elevation-azimuth masks.
    num_background_elements = np.sum(all_ea_masks_person == 0) # Number of 'backgroud' elements across all elevation-azimuth masks.
    
    weights = {
        'background': num_person_elements / total_num_of_elements,
        'person'    : num_background_elements / total_num_of_elements
    }
    return weights

# Get a dictionary where the keys are the sequence names and the values are lists of their frames.
def get_dataset_sequence_frames(dataset_dir):
    # Initialize dictionary.
    dataset_sequence_frames = {}

    # Get the sequences in the dataset directory.
    sequences = [d for d in os.listdir(dataset_dir) if (os.path.isdir(os.path.join(dataset_dir, d)) and d != 'samples')]

    # For each sequence:
    for seq in sequences:
        # dataset_dir contains subdirectories whose names are each one of the frame names.
        all_frames_dir = os.path.join(dataset_dir, seq, 'annotations', 'dense')

        # Get the sequences in the dataset directory.
        frames = [d for d in os.listdir(all_frames_dir) if os.path.isdir(os.path.join(all_frames_dir, d))]

        # Add this sequence, along with a list of its frames, to the dictionary.
        dataset_sequence_frames[seq] = frames

    return dataset_sequence_frames

# Get a dictionary where the keys are the sequences and values specify the split ('Train', 'Validation', 'Test')
# of each sequence. The values in the dictionary are in the format of key-value pairs, e.g., 'split': 'Train'.
def get_dataset_sequence_splits(dataset_dir):
    # Initialize dictionary.
    dataset_sequence_splits = {}

    # Get the sequences in the dataset directory.
    sequences = [d for d in os.listdir(dataset_dir) if (os.path.isdir(os.path.join(dataset_dir, d)) and d != 'samples')]

    # Set all sequences as keys of the dictionary.
    dataset_sequence_splits = {seq: None for seq in sequences}

    # Shuffle the order of the sequences so that the splits will be randomized.
    random.shuffle(sequences)

    # Calculate the indices at which the splits are performed, based on the split ratios specified.
    train_index = int(len(sequences) * TRAIN_SET_SPLIT)
    val_index   = int(len(sequences) * (TRAIN_SET_SPLIT + VAL_SET_SPLIT))

    # Split the list into training, validation, and test sets.
    train_set = sequences[:train_index]
    val_set   = sequences[train_index:val_index]

    # Set the data value of each sequence in the dictionary based on which split it was assigned to.
    for seq in sequences:
        if seq in train_set:
            dataset_sequence_splits[seq] = {'split': 'Train'}
        elif seq in val_set:
            dataset_sequence_splits[seq] = {'split': 'Validation'}
        else:
            dataset_sequence_splits[seq] = {'split': 'Test'}

    return dataset_sequence_splits