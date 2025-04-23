import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks

# ------------------------
# 1. Load Data
# ------------------------
def load_subset_data(filename):
    with fits.open(filename) as hdulist:
        data = hdulist[1].data
        ra = data['RA']
        dec = data['DEC']
    return ra, dec

ra_data, dec_data = load_subset_data('group_with_redshift.fits')

# ------------------------
# 2. Generate Random Catalog
# ------------------------
n_rand = 50 * len(ra_data)  # Ensure sufficient randoms
ra_min, ra_max = np.min(ra_data), np.max(ra_data)
dec_min, dec_max = np.min(dec_data), np.max(dec_data)
ra_rand = np.random.uniform(ra_min, ra_max, n_rand)
dec_rand = np.random.uniform(dec_min, dec_max, n_rand)

# Convert to radians
ra_data_rad = np.radians(ra_data)
dec_data_rad = np.radians(dec_data)
ra_rand_rad = np.radians(ra_rand)
dec_rand_rad = np.radians(dec_rand)

# ------------------------
# 3. Compute Pair Counts
# ------------------------
# Define angular bins (adjust to your science case)
theta_bins = np.logspace(-2, 1 , 100)  # 0.01 to 10 degrees

# Calculate raw pair counts
DD = DDtheta_mocks(autocorr=1, nthreads=4, binfile=theta_bins,
                   RA1=ra_data_rad, DEC1=dec_data_rad,
                   RA2=ra_data_rad, DEC2=dec_data_rad)
DR = DDtheta_mocks(autocorr=0, nthreads=4, binfile=theta_bins,
                   RA1=ra_data_rad, DEC1=dec_data_rad,
                   RA2=ra_rand_rad, DEC2=dec_rand_rad)
RR = DDtheta_mocks(autocorr=1, nthreads=4, binfile=theta_bins,
                   RA1=ra_rand_rad, DEC1=dec_rand_rad,
                   RA2=ra_rand_rad, DEC2=dec_rand_rad)

# Extract counts
dd, dr, rr = DD['npairs'], DR['npairs'], RR['npairs']

# ------------------------
# 4. Normalize Pair Counts
# ------------------------
n_data = len(ra_data)
n_rand = len(ra_rand)

# Normalization factors
dd_norm_factor = n_data * (n_data - 1) / 2
dr_norm_factor = n_data * n_rand
rr_norm_factor = n_rand * (n_rand - 1) / 2

# Normalized pair counts
dd_normalized = dd / dd_norm_factor
dr_normalized = dr / dr_norm_factor
rr_normalized = rr / rr_norm_factor

# Compute w(theta)
with np.errstate(divide='ignore', invalid='ignore'):
    w_theta = (dd_normalized - 2 * dr_normalized + rr_normalized) / rr_normalized

# Mask invalid bins (RR=0 or NaN)
valid_mask = (rr > 0) & np.isfinite(w_theta)
theta_centers = (theta_bins[1:] + theta_bins[:-1]) / 2
theta_valid = theta_centers[valid_mask]
w_valid = w_theta[valid_mask]

# ------------------------
# 5. Plot
# ------------------------
plt.figure(figsize=(10, 6))
plt.plot(theta_valid, w_valid, 'ko-', markersize=5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\theta$ [deg]', fontsize=12)
plt.ylabel(r'$w(\theta)$', fontsize=12)
plt.title('Angular Correlation Function', fontsize=14)
plt.grid(which='both', alpha=0.5)
plt.savefig('angular_correlation_function.png', dpi=300)
plt.show()
