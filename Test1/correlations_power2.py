import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from scipy.optimize import curve_fit

# Load data
def load_subset_data(filename):
    with fits.open(filename) as hdulist:
        data = hdulist[1].data
        ra = data['RA']
        dec = data['DEC']
    return ra, dec

ra_data, dec_data = load_subset_data('group_with_redshift.fits')  # Update filename

# Generate random catalog
n_data = len(ra_data)
n_rand = 100 * n_data  # Increased random points
ra_min, ra_max = np.min(ra_data), np.max(ra_data)
dec_min, dec_max = np.min(dec_data), np.max(dec_data)
ra_rand = np.random.uniform(ra_min, ra_max, n_rand)
dec_rand = np.random.uniform(dec_min, dec_max, n_rand)

# Convert to radians
ra_data_rad = np.radians(ra_data)
dec_data_rad = np.radians(dec_data)
ra_rand_rad = np.radians(ra_rand)
dec_rand_rad = np.radians(dec_rand)

# Compute pair counts
theta_bins = np.logspace(-2, 1, 100)  # 0.1° to 10°, 14 bins
DD = DDtheta_mocks(autocorr=1, nthreads=4, binfile=theta_bins,
                   RA1=ra_data_rad, DEC1=dec_data_rad,
                   RA2=ra_data_rad, DEC2=dec_data_rad)
DR = DDtheta_mocks(autocorr=0, nthreads=4, binfile=theta_bins,
                   RA1=ra_data_rad, DEC1=dec_data_rad,
                   RA2=ra_rand_rad, DEC2=dec_rand_rad)
RR = DDtheta_mocks(autocorr=1, nthreads=4, binfile=theta_bins,
                   RA1=ra_rand_rad, DEC1=dec_rand_rad,
                   RA2=ra_rand_rad, DEC2=dec_rand_rad)

# Normalize pair counts
dd, dr, rr = DD['npairs'], DR['npairs'], RR['npairs']
dd_norm = n_data * (n_data - 1) / 2
dr_norm = n_data * n_rand
rr_norm = n_rand * (n_rand - 1) / 2
dd_normalized = dd / dd_norm
dr_normalized = dr / dr_norm
rr_normalized = rr / rr_norm

# Compute ACF
with np.errstate(divide='ignore', invalid='ignore'):
    w_theta = (dd_normalized - 2 * dr_normalized + rr_normalized) / rr_normalized
valid_mask = (rr > 0) & np.isfinite(w_theta)
theta_centers = (theta_bins[1:] + theta_bins[:-1]) / 2
theta_valid = theta_centers[valid_mask]
w_valid = w_theta[valid_mask]

# Fit power law
def power_law(theta, A, delta):
    return A * theta**(-delta)

popt, pcov = curve_fit(power_law, theta_valid, w_valid, p0=[1.0, 0.7])
A_fit, delta_fit = popt
A_err, delta_err = np.sqrt(np.diag(pcov))

print(f"Best-fit A: {A_fit:.3f} ± {A_err:.3f}")
print(f"Best-fit δ: {delta_fit:.3f} ± {delta_err:.3f}")

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(theta_valid, w_valid, s=50, color='k', label='Data')
theta_fine = np.logspace(np.log10(theta_valid.min()), np.log10(theta_valid.max()), 100)
plt.plot(theta_fine, power_law(theta_fine, A_fit, delta_fit), 'r--',
         label=f'Fit: $A={A_fit:.2f}$, $\delta={delta_fit:.2f}$')

plt.plot(theta_fine, 1.0 * theta_fine**(-0.7), 'b:', 
         label='Maddox+90: $A=1.0$, $\delta=0.7$')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$ \theta $ [deg]')
plt.ylabel('$w( \theta )$')
plt.legend()
plt.grid(which='both', alpha=0.5)
plt.savefig('acf_fit2.png', dpi=300)
plt.show()