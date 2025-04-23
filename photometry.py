import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from photutils.detection import DAOStarFinder
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import KDTree
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from tqdm import tqdm


def calibrated_fits(folder_path):
    fits_files = [f for f in os.listdir(folder_path) if f.endswith("_df.fits") and f.startswith("algol")]
    fits_files.sort(key=lambda x: x.split("_")[2])  # sort by timestamp
    return fits_files


def zscale_of_image(img_data, title="Pick a star"):
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(img_data)
    
    plt.imshow(img_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(label="Pixel Value")
    plt.title(title)


def find_stars_in_image(img_data):
    daofind = DAOStarFinder(fwhm=5.0, threshold=1.5 * np.std(img_data))
    sources = daofind(img_data)
    
    if sources is None or len(sources) == 0:
        return np.array([])
    
    # (x,y) coordinates as array
    return np.array([sources["xcentroid"], sources["ycentroid"]]).T


def locate_nearest_star(img_data, x_guess, y_guess, box_size=20):
    ny, nx = img_data.shape
    x_min = max(0, int(x_guess - box_size))
    x_max = min(nx, int(x_guess + box_size))
    y_min = max(0, int(y_guess - box_size)) 
    y_max = min(ny, int(y_guess + box_size))
    
    cropped_img = img_data[y_min:y_max, x_min:x_max]
    if cropped_img.size == 0:
        return None
    
    cropped_img = np.nan_to_num(cropped_img)
    detected_stars = find_stars_in_image(cropped_img)
    
    if len(detected_stars) == 0:
        return None
    
    detected_stars[:, 0] += x_min
    detected_stars[:, 1] += y_min
    
    star_tree = KDTree(detected_stars)
    _, idx = star_tree.query([x_guess, y_guess])
    return tuple(detected_stars[idx])

def algol_position_tracking(input_dir, fits_files):
    algol_pos_list = []
    time_stamps = []
    
    for file in tqdm(fits_files, desc="Tracking Algol"):
        file_path = os.path.join(input_dir, file)
        with fits.open(file_path) as hdul:
            img_data = hdul[0].data
            header = hdul[0].header
            x_guess, y_guess = header["PHDLOCKX"], header["PHDLOCKY"]
            timestamp = header["DATE-OBS"]
        
        algol_pos = locate_nearest_star(img_data, x_guess, y_guess)
        algol_pos_list.append(algol_pos)
        time_stamps.append(timestamp)
    
    return algol_pos_list, time_stamps


def reference_stars_tracking(input_dir, fits_files):
    # showing first frame for star selection
    first_file = os.path.join(input_dir, fits_files[0])
    with fits.open(first_file) as hdul:
        first_img = hdul[0].data
    
    
    num_ref_stars = int(input("How many reference stars do you want to track? "))
    
    
    plt.figure(figsize=(8, 6))
    zscale_of_image(first_img, title="Click to select reference stars")
    ref_pos_initial = plt.ginput(num_ref_stars, timeout=0)
    plt.close()
    
    # initializing tracking arrays
    ref_pos_all = [[] for _ in range(num_ref_stars)]
    
    # refining initial positions
    for i, pos in enumerate(ref_pos_initial):
        refined_pos = locate_nearest_star(first_img, *pos)
        ref_pos_all[i].append(refined_pos)
    
    # track stars through remaining frames
    for file_idx, file in tqdm(enumerate(fits_files[1:], start=1), desc="Tracking reference stars"):
        file_path = os.path.join(input_dir, file)
        with fits.open(file_path) as hdul:
            img_data = hdul[0].data
        
        for i in range(num_ref_stars):
            if ref_pos_all[i][-1] is None:
                ref_pos_all[i].append(None)
                continue
            
            # find all stars in frame
            all_stars = find_stars_in_image(img_data)
            if len(all_stars) == 0:
                ref_pos_all[i].append(None)
                continue
            
            # to find closest star to previous position
            star_tree = KDTree(all_stars)
            _, idx = star_tree.query(ref_pos_all[i][-1])
            ref_pos_all[i].append(tuple(all_stars[idx]))
    
    return ref_pos_all


def save_tracking_pdf(input_dir, fits_files, algol_pos, ref_pos, timestamps, output_file="tracked_stars.pdf"):
    with PdfPages(output_file) as pdf:
        for i, file in tqdm(enumerate(fits_files), desc="Saving tracking PDF"):
            file_path = os.path.join(input_dir, file)
            with fits.open(file_path) as hdul:
                img_data = hdul[0].data
                exposure = hdul[0].header["EXPOSURE"]
            
            timestamp = timestamps[i]
            
            plt.figure(figsize=(8, 6))
            zscale_of_image(img_data, title=f"Frame {i+1}: {timestamp}\nExposure: {exposure}s")
            
            # plotting Algol position
            if algol_pos[i] is not None:
                plt.scatter(*algol_pos[i], color='blue', marker='x', s=100, label="Algol")
            
            # plotting reference star positions
            for j, star_pos in enumerate(ref_pos):
                if i < len(star_pos) and star_pos[i] is not None:
                    plt.scatter(*star_pos[i], color='red', marker='o', s=50, alpha=0.5, label=f"Ref Star {j+1}")
            
            plt.legend()
            pdf.savefig()
            plt.close()
    
    print(f"Saved tracking results to {output_file}")


def measure_star_flux(input_dir, fits_files, algol_pos, ref_pos, timestamps, aperture_radius=5, adu=48.0):
    inner_radius = 2 * aperture_radius
    outer_radius = 3 * aperture_radius
    
    
    algol_flux_data = {}  # {timestamp: (flux, variance)}
    ref_flux_data = [{} for _ in range(len(ref_pos))]  
    
    for i, file in tqdm(enumerate(fits_files), desc="Measuring star flux"):
        file_path = os.path.join(input_dir, file)
        with fits.open(file_path) as hdul:
            img_data = hdul[0].data
            exposure_time = hdul[0].header["EXPOSURE"]
        
        timestamp = timestamps[i]
        
        # measure Algol
        algol_ap = CircularAperture(algol_pos[i], r=aperture_radius)
        algol_annulus = CircularAnnulus(algol_pos[i], r_in=inner_radius, r_out=outer_radius)
        
        # photometry for Algol
        algol_phot = aperture_photometry(img_data, algol_ap)
        algol_bkg = aperture_photometry(img_data, algol_annulus)
        
        # background calculation for Algol
        bkg_mean = algol_bkg["aperture_sum"] / algol_annulus.area
        bkg_total = bkg_mean * algol_ap.area
        
        # final flux and error for Algol
        algol_raw_flux = (algol_phot["aperture_sum"][0] - bkg_total) * adu / exposure_time
        algol_var = (algol_phot["aperture_sum"][0] + bkg_total) * adu / (exposure_time**2)
        
        algol_flux_data[timestamp] = (algol_raw_flux, algol_var)
        
        
        for j, star_pos_list in enumerate(ref_pos):
            star_ap = CircularAperture(star_pos_list[i], r=aperture_radius)
            star_annulus = CircularAnnulus(star_pos_list[i], r_in=inner_radius, r_out=outer_radius)
            
            # photometry for ref star
            star_phot = aperture_photometry(img_data, star_ap)
            star_bkg = aperture_photometry(img_data, star_annulus)
            
            # background calculation for ref star
            star_bkg_mean = star_bkg["aperture_sum"] / star_annulus.area
            star_bkg_total = star_bkg_mean * star_ap.area
            
            # final flux and error for ref star
            star_raw_flux = (star_phot["aperture_sum"][0] - star_bkg_total) * adu / exposure_time
            star_var = (star_phot["aperture_sum"][0] + star_bkg_total) * adu / (exposure_time**2)
            
            ref_flux_data[j][timestamp] = (star_raw_flux, star_var)
    
    return algol_flux_data, ref_flux_data


def calculate_relative_flux(algol_flux, ref_flux_list, timestamps):
    rel_flux_data = {}
    
    for j in range(len(ref_flux_list)):
        ref_key = f"ref_{j+1}"
        rel_flux_data[ref_key] = {}
        
        for ts in timestamps:
            if ts in algol_flux and ts in ref_flux_list[j]:
                a_flux, a_var = algol_flux[ts]
                r_flux, r_var = ref_flux_list[j][ts]
                
                
                rel_flux = a_flux / r_flux
            
                rel_err = rel_flux * np.sqrt(a_var/a_flux**2 + r_var/r_flux**2)
                
                rel_flux_data[ref_key][ts] = (rel_flux, rel_err)
    
    return rel_flux_data

# saving light curves to PDF (fixed shape mismatch)
def save_lightcurve_pdf(rel_flux_data, timestamps, output_file="algol_lightcurves.pdf"):
    with PdfPages(output_file) as pdf:
        for ref_name, flux_dict in tqdm(rel_flux_data.items(), desc="Creating light curves"):
            # get valid timestamps
            valid_timestamps = [ts for ts in timestamps if ts in flux_dict]
            
            
            time_labels = [ts.split("T")[1][:-7] for ts in valid_timestamps]
            
            flux_values = []
            flux_errors = []
            
            for ts in valid_timestamps:
                flux, err = flux_dict[ts]
                if hasattr(flux, 'item'):
                    flux = flux.item()
                if hasattr(err, 'item'):
                    err = err.item()
                
                flux_values.append(flux)
                flux_errors.append(abs(err))
            
            # Converting to regular Python lists for compatibility
            flux_values = list(map(float, flux_values))
            flux_errors = list(map(float, flux_errors))
            
            # plotting the light curve
            plt.figure(figsize=(8, 6))
            plt.errorbar(time_labels, flux_values, yerr=flux_errors, color="k", 
                         linestyle="-", ecolor="red", fmt='o', capsize=3)
            
            plt.xlabel("Time(UTC)")
            plt.ylabel(f"Relative Flux (Algol / {ref_name})")
            plt.title(f"Algol Light Curve (Relative to {ref_name})")
            plt.grid(True)
            plt.ylim(0, 255)
            
            # make x-axis readable
            step = max(1, len(time_labels) // 10)  # show ~10 labels max
            plt.xticks(time_labels[::step], rotation=45)
            
            pdf.savefig()
            plt.close()
    
    print(f"Saved light curves to {output_file}")

# main function to run everything
def process_algol_data():
    # folder with calibrated data files
    data_folder = r"C:\algol_project\calibrated_frames_prashant_trial"
    
    print("Starting star tracking...")
    
    
    fits_files = calibrated_fits(data_folder)
    
    # tracking Algol and ref stars
    algol_positions, time_stamps = algol_position_tracking(data_folder, fits_files)
    ref_positions = reference_stars_tracking(data_folder, fits_files)
    
    # saving tracking results
    save_tracking_pdf(data_folder, fits_files, algol_positions, ref_positions, time_stamps)
    
    print("Star tracking complete!")
        
    print("Starting photometry...")
    
    
    aperture_radius = 5  # fixed aperture radius
    algol_flux, ref_flux = measure_star_flux(data_folder, fits_files, algol_positions, 
                                           ref_positions, time_stamps, aperture_radius)
    
    
    rel_flux = calculate_relative_flux(algol_flux, ref_flux, time_stamps)
    
    
    save_lightcurve_pdf(rel_flux, time_stamps)
    
    print("Photometry complete!")


# running the analysis
if __name__ == "__main__":
    process_algol_data()