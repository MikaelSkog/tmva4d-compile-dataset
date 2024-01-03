import argparse
import glob
import json
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scripts.yolov8_predictor import MaskPredictor
import scripts.metadata as metadata
from scripts.project_points_ea import project_points_to_ea_view
from scripts.view_instance_generator import ViewInstancesGenerator

NUM_SAMPLES_PER_SEQ = 3 # Number of samples selected per sequence.

def get_args():
    # Handle command line arguments.
    parser = argparse.ArgumentParser(description="Create a 4D radar dataset from input point cloud data files and image files.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the input directory.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory.")
    parser.add_argument('--yolov8_model', type=str, required=True, help="Path to the YOLOv8 model")
    args = parser.parse_args()

    # Check that all necessary folders exist.
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        exit(1)
    if not os.path.exists(os.path.join(args.input_dir, 'img')):
        print(f"Error: Input image directory '{os.path.join(args.input_dir, 'img')}' does not exist.")
        exit(1)
    if not os.path.exists(os.path.join(args.input_dir, 'pcd')):
        print(f"Error: Input point cloud directory '{os.path.join(args.input_dir, 'pcd')}' does not exist.")
        exit(1)
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory '{args.output_dir}' does not exist.")
        exit(1)

    # Check that the specified file is a model (extension .pt).
    if os.path.splitext(args.yolov8_model)[1] != '.pt':
        print(f"'{args.yolov8_model}' is not a YOLOv8 model (extension .pt)")
        exit(1)

    # Check that the necessary YOLOv8 model exists
    if not os.path.exists(args.yolov8_model):
        print(f"Model '{args.yolov8_model}' does not exist.")
        exit(1)

    return args.input_dir, args.output_dir, args.yolov8_model

# Get the timestamp based on the filename of a specified file.
def get_timestamp(file_path):
    filename = os.path.basename(file_path)
    timestamp_str = filename[0].replace('_', '.')
    return int(timestamp_str)


# COMPILE THE 4D RADAR DATASET.
if __name__ == "__main__":
    input_dir, output_dir, yolov8_model_path = get_args()

    pcd_dir = os.path.join(input_dir, 'pcd')
    img_dir = os.path.join(input_dir, 'img')
    dataset_dir = os.path.join(output_dir, 'Dataset4d')

    # Remove the dataset directory in the output directory (if it exists). Then create a new one.
    if os.path.exists(dataset_dir):
        try:
            shutil.rmtree(dataset_dir)
        except Exception as e:
            print(f"Error when replacing the old dataset directory: {e}")
    os.makedirs(dataset_dir)

    # Get the sequences in the point cloud input directory.
    sequences = [d for d in os.listdir(pcd_dir) if os.path.isdir(os.path.join(pcd_dir, d))]

    print("Gathering statistics on the input point cloud data files.")
    # Get the minimums and maximums for the range, power, and Doppler values as a dictionary, then save it.
    pcd_stats = metadata.get_pcd_stats(pcd_dir)
    with open(os.path.join(dataset_dir, 'pcd_stats.json'), 'w') as f:
        json.dump(pcd_stats, f, indent=2)

    # Initialize the YOLOv8 predictor, using the model at the specified path.
    mask_predictor = MaskPredictor(yolov8_model_path)

    # Create a new directory in the dataset directory, with visualizations for a few sample frames.
    dataset_samples_dir = os.path.join(dataset_dir, 'samples')
    os.makedirs(dataset_samples_dir)

    # PREPARE THE DATASET:
    # For each sequence:
    for seq in sequences:
        print(f"Processing sequence '{seq}'.")
        # Get the image sequence directories in the 'pcd' and 'img' directories for this sequence.
        pcd_seq_dir = os.path.join(pcd_dir, seq)
        img_seq_dir = os.path.join(img_dir, seq)

        # If the directory of this sequence is missing from the 'img' directory, skip it.
        if not os.path.exists(img_seq_dir):
            print(f"Warning: Sequence '{seq}' missing in the img folder; skipping it.")
            continue
        
        # Get every point cloud data file and image file of the sequence from their respective directories.
        pcd_file_paths = glob.glob(os.path.join(pcd_seq_dir, '*.pcd'))
        img_file_paths = [file for ext in ('jpg', 'jpeg', 'png') for file in glob.glob(os.path.join(img_seq_dir, f'*.{ext}'))]

        # If no images were found for the sequence, skip this sequence:
        if len(img_file_paths) == 0:
            print(f"Warning: No images found for sequence '{seq}'; skipping it.")
            continue

        # If no point cloud data files were found for the sequence, skip this sequence:
        if len(pcd_file_paths) == 0:
            print(f"Warning: No point cloud data files found for sequence '{seq}'; skipping it.")
            continue

        # Create a new directory in the dataset directory corresponding to this sequence, along with subdirectories.
        dataset_seq_dir             = os.path.join(dataset_dir, seq)
        dataset_seq_img_dir         = os.path.join(dataset_seq_dir, 'camera_images')
        dataset_seq_annotations_dir = os.path.join(dataset_seq_dir, 'annotations', 'dense')
        dataset_seq_ea_dir          = os.path.join(dataset_seq_dir, 'elevation_azimuth_processed')
        dataset_seq_er_dir          = os.path.join(dataset_seq_dir, 'elevation_range_processed')
        dataset_seq_ed_dir          = os.path.join(dataset_seq_dir, 'elevation_doppler_processed')
        dataset_seq_ra_dir          = os.path.join(dataset_seq_dir, 'range_azimuth_processed')
        dataset_seq_da_dir          = os.path.join(dataset_seq_dir, 'doppler_azimuth_processed')
        os.makedirs(dataset_seq_dir)
        os.makedirs(dataset_seq_img_dir)
        os.makedirs(dataset_seq_annotations_dir)
        os.makedirs(dataset_seq_ea_dir)
        os.makedirs(dataset_seq_er_dir)
        os.makedirs(dataset_seq_ed_dir)
        os.makedirs(dataset_seq_ra_dir)
        os.makedirs(dataset_seq_da_dir)

        # Create a directory corresponding to this sequence in the samples directory. All samples for this sequence
        # will be placed here.
        dataset_samples_seq_dir = os.path.join(dataset_samples_dir, seq)
        os.makedirs(dataset_samples_seq_dir)

        # Initialize the frame counter for this sequence.
        frame = 0
        
        # For each point cloud in the sequence directory:
        for pcd_file_path in pcd_file_paths:
            # Turn the frame number into a 6 digit string with leading zeros. This will be the frame's name.
            frame_name = str(frame).zfill(6)

            # Get the path of the image temporally closest to this point cloud.
            closest_img_to_pcd_path = min(
                img_file_paths, key=lambda img_file_path: abs(get_timestamp(pcd_file_path) - get_timestamp(img_file_path)))
            
            # Get and save the camera image. Also get its resolution.
            img_save_filename = frame_name + os.path.splitext(closest_img_to_pcd_path)[1] # File extension.
            shutil.copy(closest_img_to_pcd_path, os.path.join(dataset_seq_img_dir, img_save_filename))
            img = Image.open(closest_img_to_pcd_path)
            img_resolution = img.size

            # Get (predict) and save the YOLOv8 prediction mask(s) for the image. Also get a visualization of the predictions.
            predictions_img, masks = mask_predictor.predict(img)
            os.makedirs(os.path.join(dataset_seq_annotations_dir, frame_name))
            np.save(os.path.join(dataset_seq_annotations_dir, frame_name, 'elevation_azimuth'), masks)

            # Get and save the instance for each view.
            view_instance_generator = ViewInstancesGenerator(pcd_file_path, pcd_stats, img_resolution)
            ea_instance = view_instance_generator.get_ea_instance() # Elevation-azimuth.
            er_instance = view_instance_generator.get_er_instance() # Elevation-range.
            ed_instance = view_instance_generator.get_ed_instance() # Elevation-Doppler.
            ra_instance = view_instance_generator.get_ra_instance() # Range-azimuth.
            da_instance = view_instance_generator.get_da_instance() # Doppler-azimuth.
            np.save(os.path.join(dataset_seq_ea_dir, frame_name), ea_instance) # Elevation-azimuth.
            np.save(os.path.join(dataset_seq_er_dir, frame_name), er_instance) # Elevation-range.
            np.save(os.path.join(dataset_seq_ed_dir, frame_name), ed_instance) # Elevation-Doppler.
            np.save(os.path.join(dataset_seq_ra_dir, frame_name), ra_instance) # Range-azimuth.
            np.save(os.path.join(dataset_seq_da_dir, frame_name), da_instance) # Doppler-azimuth.

            # Potentially get visualizations for this frame.
            viz_part_int = round(len(pcd_file_paths) / (NUM_SAMPLES_PER_SEQ + 1)) # Visualization partitioning interval.
            for factor in range(1, NUM_SAMPLES_PER_SEQ + 1):
                # If this is one of the frames chosen as samples:
                if viz_part_int * factor == frame + 1:
                    # Create a directory corresponding to this frame in the current sequence.
                    dataset_samples_seq_frame_dir = os.path.join(dataset_samples_seq_dir, frame_name)
                    os.makedirs(dataset_samples_seq_frame_dir)

                    # Save an image showing the segmentation masks and bounding boxes overlayed on top of the input image.
                    cv2.imwrite(os.path.join(dataset_samples_seq_frame_dir, 'seg_pred.png'), predictions_img)

                    # For each view, save a visualization of the frame's instance of the view.
                    # Elevation-azimuth.
                    plt.imshow(ea_instance, cmap='viridis', aspect='equal')
                    plt.ylabel('Elevation')
                    plt.xlabel('Azimuth')
                    plt.xticks([])
                    plt.yticks([])
                    plt.savefig(os.path.join(dataset_samples_seq_frame_dir, 'ea.png'), bbox_inches='tight', pad_inches=0.1, dpi=600)
                    plt.close()
                    # Elevation-range.
                    plt.imshow(er_instance, cmap='viridis', aspect='equal')
                    plt.ylabel('Elevation')
                    plt.xlabel('Range')
                    plt.xticks([])
                    plt.yticks([])
                    plt.savefig(os.path.join(dataset_samples_seq_frame_dir, 'er.png'), bbox_inches='tight', pad_inches=0.1, dpi=600)
                    plt.close()
                    # Elevation-Doppler.
                    plt.imshow(ed_instance, cmap='viridis', aspect='equal')
                    plt.ylabel('Elevation')
                    plt.xlabel('Doppler')
                    plt.xticks([])
                    plt.yticks([])
                    plt.savefig(os.path.join(dataset_samples_seq_frame_dir, 'ed.png'), bbox_inches='tight', pad_inches=0.1, dpi=600)
                    plt.close()
                    # Range-azimuth.
                    plt.imshow(np.flipud(ra_instance), cmap='viridis', aspect='equal')
                    plt.ylabel('Range')
                    plt.xlabel('Azimuth')
                    plt.xticks([])
                    plt.yticks([])
                    plt.savefig(os.path.join(dataset_samples_seq_frame_dir, 'ra_flipud.png'), bbox_inches='tight', pad_inches=0.1, dpi=600)
                    plt.close()
                    # Doppler-azimuth.
                    plt.imshow(np.flipud(ra_instance), cmap='viridis', aspect='equal')
                    plt.ylabel('Doppler')
                    plt.xlabel('Azimuth')
                    plt.xticks([])
                    plt.yticks([])
                    plt.savefig(os.path.join(dataset_samples_seq_frame_dir, 'da_flipud.png'), bbox_inches='tight', pad_inches=0.1, dpi=600)
                    plt.close()

                    # Project and save a visualization of the points projected to the elevation-azimuth view, and the grid that defines the
                    # boundaries of the cells.
                    ea_projection_plot = project_points_to_ea_view(pcd_file_path, img_resolution)
                    ea_projection_plot.savefig(os.path.join(dataset_samples_seq_frame_dir, 'ea_projection.png'), bbox_inches='tight', pad_inches=0.1, dpi=600)

            # Increment the frame counter for this sequence.
            frame += 1
    
    # GENERATE METADATA FOR THE DATASET:
    print("Generating dataset metadata.")
    # Get and save statistics (mean, standard deviation, minimum, maximum) for all elements in all instances of
    # each view.
    ea_stats = metadata.get_stats_for_view(dataset_dir, 'elevation_azimuth_processed')
    er_stats = metadata.get_stats_for_view(dataset_dir, 'elevation_range_processed')
    ed_stats = metadata.get_stats_for_view(dataset_dir, 'elevation_doppler_processed')
    ra_stats = metadata.get_stats_for_view(dataset_dir, 'range_azimuth_processed')
    da_stats = metadata.get_stats_for_view(dataset_dir, 'doppler_azimuth_processed')
    with open(os.path.join(dataset_dir, 'ea_stats_all.json'), 'w') as f:
        json.dump(ea_stats, f, indent=2)
    with open(os.path.join(dataset_dir, 'er_stats_all.json'), 'w') as f:
        json.dump(er_stats, f, indent=2)
    with open(os.path.join(dataset_dir, 'ed_stats_all.json'), 'w') as f:
        json.dump(ed_stats, f, indent=2)
    with open(os.path.join(dataset_dir, 'ra_stats_all.json'), 'w') as f:
        json.dump(ra_stats, f, indent=2)
    with open(os.path.join(dataset_dir, 'da_stats_all.json'), 'w') as f:
        json.dump(da_stats, f, indent=2)

    # Get and save the weights for the classes 'background' and 'person' to compensate for the class imbalance.
    ea_weights = metadata.get_weights_for_ea_view(dataset_dir)
    with open(os.path.join(dataset_dir, 'ea_weights.json'), 'w') as f:
        json.dump(ea_weights, f, indent=2)

    # Get and save a dictionary where each key is a sequences, and the values are a list of each sequence's frames.
    frames_in_each_sequence = metadata.get_dataset_sequence_frames(dataset_dir)
    with open(os.path.join(dataset_dir, 'light_dataset_frame_oriented.json'), 'w') as f:
        json.dump(frames_in_each_sequence, f, indent=2)

    # Get and save a dictionary where each key is a sequences, and the values specify the split ('Train', 'Validation', 'Test')
    # of each sequence. The values in the dictionary are in the format of key-value pairs, e.g., 'split': 'Train'.
    frames_in_each_sequence = metadata.get_dataset_sequence_splits(dataset_dir)
    with open(os.path.join(dataset_dir, 'data_seq_ref.json'), 'w') as f:
        json.dump(frames_in_each_sequence, f, indent=2)

    print("Dataset compilation completed!")