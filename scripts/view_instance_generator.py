import cv2
import numpy as np
import open3d as o3d

# Number of bins in the elevation and azimuth dimension (before resizing).
# NOTE: These should be set such that the "grid" representing the bin divisions in the
# elevation-azimuth view fits the grid-like distribution of points projecte
ELEVATION_SIZE = 28
AZIMUTH_SIZE = 44

# Number of bins in the range and Doppler dimensions.
RANGE_SIZE = 256
DOPPLER_SIZE = 256

class ViewInstancesGenerator():
    def __init__(self, point_cloud_file_path, pcd_stats, image_resolution):
        image_width, image_height = image_resolution

        # Read the point cloud data file and get its points.
        point_cloud = o3d.io.read_point_cloud(point_cloud_file_path)
        points = np.asarray(point_cloud.points)

        # Extract the azimuth, elevation, and range values (x, y, z) from the point cloud.
        azimuth_values = points[:, 0]
        elevation_values = points[:, 1]
        range_values = points[:, 2]

        # Read the point cloud data file and extract the power and Doppler values.
        power_values = []
        doppler_values = []
        with open(point_cloud_file_path, 'r') as f:
            lines = f.readlines()
            flag = False
            # Get all data lines.
            for line in lines:
                if line.startswith("DATA"):
                    flag = True
                    continue
                if flag:
                    # Get the power and Doppler values for the points of this point cloud.
                    values = line.split()
                    power = float(values[3])  # Assuming the power value of the point is at in field of index 3.
                    power_values.append(power)
                    doppler = float(values[4])  # Assuming the Doppler value of the point is in the field of index 4.
                    doppler_values.append(doppler)

        # Convert the power and Doppler values to NumPy arrays.
        power_values = np.array(power_values)
        doppler_values = np.array(doppler_values)

        # Initialize the matrices, each corresponding to an instance in a distinct view.
        self.ea_matrix = np.zeros([ELEVATION_SIZE, AZIMUTH_SIZE])
        self.er_matrix = np.zeros([ELEVATION_SIZE, RANGE_SIZE])
        self.ed_matrix = np.zeros([ELEVATION_SIZE, DOPPLER_SIZE])
        self.ra_matrix = np.zeros([RANGE_SIZE, AZIMUTH_SIZE])
        self.da_matrix = np.zeros([DOPPLER_SIZE, AZIMUTH_SIZE])
        
        # Update (potentially) the matrices based on the point specified by the point index.
        def update_matrices(point_idx):
            # Get the point's bin in each dimension.
            azimuth_bin_idx = round(azimuth_values[point_idx] * (AZIMUTH_SIZE-1) / (image_width-1))
            elevation_bin_idx = round(elevation_values[point_idx] * (ELEVATION_SIZE-1) / (image_height-1))
            range_bin_idx = round((range_values[point_idx] - pcd_stats['range']['min']) * 
                                      (RANGE_SIZE-1) / (pcd_stats['range']['max'] - pcd_stats['range']['min']))
            doppler_bin_idx = round((doppler_values[point_idx] - pcd_stats['doppler']['min']) * 
                                      (DOPPLER_SIZE-1) / (pcd_stats['doppler']['max'] - pcd_stats['doppler']['min']))
            power = power_values[point_idx] - pcd_stats['power']['min']

            # If the point does not fall within the camera when projected to the camera view, skip it.
            if (azimuth_bin_idx < 0 or azimuth_bin_idx > AZIMUTH_SIZE-1 or
                elevation_bin_idx < 0 or elevation_bin_idx > ELEVATION_SIZE-1):
                return
            
            # If the range bin index or the power is negative (after subtracting the minimum power), then the point is
            # invalid; skip it.
            if (range_bin_idx < 0 or power < 0):
                return

            # If, somehow (most likely due to an error in the code to get the pcd stats or when calculating the bins)
            # points that are supposed to be included are excluded due to their range or Doppler values being slighly
            # too large: include them anyway.
            if range_bin_idx > RANGE_SIZE-1:
                range_bin_idx = RANGE_SIZE-1
            if doppler_bin_idx > DOPPLER_SIZE-1:
                doppler_bin_idx = DOPPLER_SIZE-1
            # Do the same if the Doppler value is too small.
            if doppler_bin_idx < 0:
                doppler_bin_idx = 0
            
            # Whatever cell this point corresponds to in each view, update that cell
            # if this point has a higher power than the previous occupant of the cell
            if power > self.ea_matrix[elevation_bin_idx][azimuth_bin_idx]:
                self.ea_matrix[elevation_bin_idx][azimuth_bin_idx] = power
            if power > self.er_matrix[elevation_bin_idx][range_bin_idx]:
                self.er_matrix[elevation_bin_idx][range_bin_idx] = power
            if power > self.ed_matrix[elevation_bin_idx][doppler_bin_idx]:
                self.ed_matrix[elevation_bin_idx][doppler_bin_idx] = power
            if power > self.ra_matrix[range_bin_idx][azimuth_bin_idx]:
                self.ra_matrix[range_bin_idx][azimuth_bin_idx] = power
            if power > self.da_matrix[doppler_bin_idx][azimuth_bin_idx]:
                self.da_matrix[doppler_bin_idx][azimuth_bin_idx] = power

        # For each point, update (potentially) the matrices.
        for i in range(len(azimuth_values)):
            update_matrices(i)

        # Resize the instances.
        self.ea_matrix = cv2.resize(self.ea_matrix, dsize=(128, 128), interpolation=cv2.INTER_LINEAR).astype('float64')
        self.er_matrix = cv2.resize(self.er_matrix, dsize=(256, 128), interpolation=cv2.INTER_LINEAR).astype('float64')
        self.ed_matrix = cv2.resize(self.ed_matrix, dsize=(256, 128), interpolation=cv2.INTER_LINEAR).astype('float64')
        self.ra_matrix = cv2.resize(self.ra_matrix, dsize=(128, 256), interpolation=cv2.INTER_LINEAR).astype('float64')
        self.da_matrix = cv2.resize(self.da_matrix, dsize=(128, 256), interpolation=cv2.INTER_LINEAR).astype('float64')

    def get_ea_instance(self):
        return self.ea_matrix
    
    def get_er_instance(self):
        return self.er_matrix
    
    def get_ed_instance(self):
        return self.ed_matrix
    
    def get_ra_instance(self):
        return self.ra_matrix
    
    def get_da_instance(self):
        return self.da_matrix