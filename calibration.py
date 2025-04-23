import os
import numpy as np
from astropy.io import fits
import scipy.ndimage as ndimage
import logging

# Setting up logging to show info messages
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Global dictionaries and variables to hold data
science_dict = {'algol': [], 'capella': []}
dark_list = []
flat_path = None
dark_data_dict = {}
dark_sub_sci = {}
flat_frame = None

def setup_output_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)

def scan_files(input_dir):
    global science_dict, dark_list, flat_path
    for fname in os.listdir(input_dir):
        if not fname.endswith(".fits"):
            continue
        full_path = os.path.join(input_dir, fname)
        parts = fname.split('_')
        if len(parts) < 3:
            continue
        kind = parts[0]
        try:
            exp = float(parts[1])
        except ValueError:
            continue

        if kind == 'flat':
            flat_path = full_path
        elif kind == 'dark':
            dark_list.append((full_path, exp))
        elif kind == 'algol':
            science_dict['algol'].append((full_path, exp))
        elif kind == 'capella':
            science_dict['capella'].append((full_path, exp))

    logging.info(f"Found {len(science_dict['algol'])} algol, {len(science_dict['capella'])} capella, {len(dark_list)} dark, and 1 flat frame.")

def highpass_filter(img, sigma_val=50):
    # Use Gaussian smoothing for high-pass filter
    return ndimage.gaussian_filter(img, sigma=sigma_val)

def handle_flat_frame(output_dir):
    global flat_frame
    if flat_path is None:
        logging.warning("Flat file not found.")
        return

    with fits.open(flat_path) as hdu:
        raw_flat = hdu[0].data.astype(np.float64)
        header = hdu[0].header

    smoothed = highpass_filter(raw_flat)
    flat_frame = raw_flat / smoothed  # Normalize

    flat_output_path = os.path.join(output_dir, "flat_processed.fits")
    fits.writeto(flat_output_path, flat_frame, header, overwrite=True)
    logging.info(f"Saved processed flat frame: {flat_output_path}")

def subtract_dark_frames(output_dir):
    global dark_data_dict, dark_sub_sci
    # Load all dark frames and group by exposure time
    for path, exp in dark_list:
        with fits.open(path) as hdu:
            dark_data_dict[exp] = hdu[0].data.astype(np.float64)

    # Go through each science file and subtract the matching dark
    for obj_type, sci_entries in science_dict.items():
        for path, exp in sci_entries:
            with fits.open(path) as hdu:
                sci_data = hdu[0].data.astype(np.float64)
                header = hdu[0].header

            if exp in dark_data_dict:
                subtracted = sci_data - dark_data_dict[exp]
            else:
                logging.warning(f"No matching dark for {exp}s exposure. Using original frame.")
                subtracted = sci_data

            dark_sub_sci[path] = (subtracted, header)

            newname = os.path.basename(path).replace(".fits", "_d.fits")
            save_path = os.path.join(output_dir, newname)
            fits.writeto(save_path, subtracted, header, overwrite=True)
            logging.info(f"Saved dark-subtracted: {save_path}")

def apply_flat(output_dir):
    if flat_frame is None:
        logging.warning("Flat frame not available. Skipping correction.")
        return

    for original_path, (dark_sub, header) in dark_sub_sci.items():
        corrected = dark_sub / flat_frame
        newname = os.path.basename(original_path).replace(".fits", "_df.fits")
        save_path = os.path.join(output_dir, newname)
        fits.writeto(save_path, corrected, header, overwrite=True)
        logging.info(f"Saved flat-corrected frame: {save_path}")

def run_all_steps(input_dir, output_dir):
    logging.info("Starting full reduction process...")
    setup_output_folder(output_dir)
    scan_files(input_dir)
    handle_flat_frame(output_dir)
    subtract_dark_frames(output_dir)
    apply_flat(output_dir)
    logging.info("All processing completed.")

# === Run it ===
input_dir = r"C:\algol_project\Reduction"
output_dir = r"C:\algol_project\calibrated_frames_prashant_trial"
run_all_steps(input_dir, output_dir)
