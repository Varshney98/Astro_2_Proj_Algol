# This code averages individual events and saves files in the format "eventXXX_EXPOSURE_DATE-OBS.fits"
import os
import numpy as np
import logging
from astropy.io import fits
from tqdm import tqdm
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def findprefix(i):
    """Returns the prefix of the ith event files from raw data"""
    return f"event{int(i):03d}"

def time_avg(timestamps):
    """Computes the average timestamp from a list of ISO 8601 formatted timestamps"""
    datetime_objects = [datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f") for ts in timestamps]
    avg_timestamp = np.mean([dt.timestamp() for dt in datetime_objects])
    return datetime.fromtimestamp(avg_timestamp).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

def average_fits_files(fits_dir, i, output_path):
    """
    Averages multiple FITS files containing RA, Dec, and photon counts.

    Parameters:
    - fits_dir (str): Path to the directory containing FITS files
    - i (int): Event number to be averaged
    - output_path (str): Directory where output will be stored

    Returns:
    - output_filename (str): Name of the output FITS file
    """

    prefix = findprefix(i)
    fits_files = [os.path.join(fits_dir, f) for f in os.listdir(fits_dir) if f.startswith(prefix) and f.endswith(".fit")]

    if not fits_files:
        logging.warning(f"No FITS files found for event {i} in {fits_dir}.")
        return None

    # Read data and headers
    data_list, avg_date, avg_dateobs = zip(*[
        (fits.getdata(file), fits.getheader(file)['DATE'], fits.getheader(file)['DATE-OBS'])
        for file in fits_files
    ])

    logging.info(f"Number of frames averaged: {len(fits_files)}")

    # Compute averaged data
    averaged_data = np.mean(np.array(data_list), axis=0)

    # Load first header and update it
    hdr = fits.getheader(fits_files[0])
    avg_date_str = time_avg(avg_date)
    avg_dateobs_str = time_avg(avg_dateobs)

    hdr['DATE'] = avg_date_str
    hdr['DATE-OBS'] = avg_dateobs_str
    hdr['NFRAMES'] = len(fits_files)
    
    # Handle missing exposure time
    exposure = hdr.get('EXPOSURE', 'UNKNOWN')

    filename = f"{prefix}_{exposure}_{avg_dateobs_str}.fits"
    new_file = os.path.join(output_path, filename)

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    fits.writeto(new_file, averaged_data, hdr, overwrite=True)
    
    logging.info(f"Averaged file saved as: {new_file}")
    return new_file

# Example usage
fits_dir = "/home/ashutosh/tifr/assignments/astronomy_and_astrophysics_2/algol_project/data/PHD2_CameraFrames_2025-02-28-190953/"
output_path = "/home/ashutosh/tifr/assignments/astronomy_and_astrophysics_2/algol_project/data/Reduction1/"

for i in tqdm(range(1, 88), desc="Averaging frames."):
    average_fits_files(fits_dir, i, output_path)
