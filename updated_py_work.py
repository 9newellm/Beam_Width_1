import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Step 1: Input Data
position_deg = np.array([
    0, 360, 720, 1080, 1100, 1125, 1155, 1190, 1230, 1275, 1321, 1369, 1440, 1442,
    1448, 1457, 1467, 1479, 1493, 1509, 1527, 1547, 1569, 1593, 1619, 1648, 1679,
    1716, 1752, 1789, 1829, 1871, 1915, 1961, 2009, 1800, 2160, 2164, 2172, 2182,
    2194, 2208, 2224, 2242, 2262, 2284, 2318, 2348, 2383, 2421, 2461, 2506, 2510,
    2515, 2525, 2545, 2580
])
power_uW = np.array([
    98, 96, 97, 94, 93, 90, 86, 84, 81, 80, 78, 76, 75, 74, 72, 69, 68, 67, 65, 63,
    62, 60, 58, 57, 55, 53, 52, 49, 48, 47, 45, 44, 42, 41, 39, 38, 36, 33, 32, 31,
    30, 29, 28, 27, 25, 24, 23, 21, 19, 18, 17, 16, 15, 14, 13.8, 12.7, 11
])

# Step 2: Convert motor steps from degrees to micrometers
deg_to_um = 50 / 360  # 50 µm per 360°
position_um = position_deg * deg_to_um

# Step 3: Normalize data
position_norm = (position_um - position_um.min()) / (position_um.max() - position_um.min())
power_norm = (power_uW - power_uW.min()) / (power_uW.max() - power_uW.min())

# Step 4: Uncertainty in normalized power (±2 µW absolute uncertainty)
power_err_uW = np.full_like(power_uW, 2.0)
power_err_norm = power_err_uW / (power_uW.max() - power_uW.min())

# Step 5: Define Gaussian function
def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Step 6: Perform curve fit with error propagation
p0 = [1.0, 0.5, 0.1]  # Initial guess
params, cov = curve_fit(
    gaussian, position_norm, power_norm,
    p0=p0, sigma=power_err_norm, absolute_sigma=True
)
A_fit, mu_fit, sigma_fit = params
A_err, mu_err, sigma_err = np.sqrt(np.diag(cov))

# Step 7: Calculate properties
norm_range_um = position_um.max() - position_um.min()

FWHM_norm = 2.355 * sigma_fit
FWHM_um = FWHM_norm * norm_range_um
FWHM_um_err = 2.355 * sigma_err * norm_range_um

waist_um = sigma_fit * norm_range_um
waist_um_err = sigma_err * norm_range_um

z_R = 1.0  # assumed Rayleigh range (unitless here)
wavelength = (sigma_fit ** 2) * np.pi / z_R
wavelength_err = 2 * sigma_fit * sigma_err * np.pi / z_R

# Step 8: Compute residuals and R²
fit_vals = gaussian(position_norm, *params)
residuals = power_norm - fit_vals
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((power_norm - power_norm.mean()) ** 2)
r_squared = 1 - ss_res / ss_tot

# Step 9: Plot data and fit
x_fit = np.linspace(mu_fit - 3 * sigma_fit, mu_fit + 3 * sigma_fit, 1000)
y_fit = gaussian(x_fit, *params)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1], sharex=True)

# Top plot: Fit and error bars
ax1.errorbar(position_norm, power_norm, yerr=power_err_norm, fmt='o', label='Data ± Error', color='blue')
ax1.plot(x_fit, y_fit, 'r-', label='Gaussian Fit')
ax1.axhline(0.5, color='gray', linestyle=':', label='Half Max')
ax1.axvline(mu_fit - sigma_fit, color='green', linestyle='--', label='±σ')
ax1.axvline(mu_fit + sigma_fit, color='green', linestyle='--')
ax1.axvline(mu_fit - 2 * sigma_fit, color='purple', linestyle='--', label='±2σ')
ax1.axvline(mu_fit + 2 * sigma_fit, color='purple', linestyle='--')

textstr = '\n'.join([
    f"FWHM = {FWHM_um:.1f} ± {FWHM_um_err:.1f} µm",
    f"w₀ = {waist_um:.1f} ± {waist_um_err:.1f} µm",
    f"λ = ({wavelength:.2e} ± {wavelength_err:.1e}) m",
    f"R² = {r_squared:.4f}"
])
ax1.annotate(textstr, xy=(0.05, 0.5), xycoords='axes fraction',
             fontsize=10, bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"))
ax1.set_title("Gaussian Fit to Beam Profile")
ax1.set_ylabel("Normalized Power")
ax1.legend()
ax1.grid(True)

# Bottom plot: Residuals
ax2.axhline(0, color='black')
ax2.plot(position_norm, residuals, 'ro')
ax2.set_xlabel("Normalized Position")
ax2.set_ylabel("Residuals")
ax2.grid(True)

plt.tight_layout()
plt.show()

# Step 10: Print summary
print("\n=== Beam Profile Fit Summary ===")
print(f"Amplitude (A):            {A_fit:.4f} ± {A_err:.4f}")
print(f"Center (μ):               {mu_fit:.4f} ± {mu_err:.4f} (normalized)")
print(f"Standard Deviation (σ):   {sigma_fit:.4f} ± {sigma_err:.4f} (normalized)")
print(f"FWHM:                     {FWHM_um:.1f} ± {FWHM_um_err:.1f} µm")
print(f"Beam Waist (w₀):          {waist_um:.1f} ± {waist_um_err:.1f} µm")
print(f"Estimated Wavelength (λ): {wavelength:.2e} ± {wavelength_err:.1e} m")
print(f"Goodness of Fit (R²):     {r_squared:.4f}")