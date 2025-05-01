import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data (position in degrees and power in microwatts)
position = np.array([0, 360, 720, 1080, 1100, 1125, 1155, 1190, 1230, 1275, 1321, 1369, 1440, 1442, 
                     1448, 1457, 1467, 1479, 1493, 1509, 1527, 1547, 1569, 1593, 1619, 1648, 1679, 1716, 
                     1752, 1789, 1829, 1871, 1915, 1961, 2009, 1800, 2160, 2164, 2172, 2182, 2194, 2208, 
                     2224, 2242, 2262, 2284, 2318, 2348, 2383, 2421, 2461, 2506, 2510, 2515, 2525, 2545, 2580])
power = np.array([98, 96, 97, 94, 93, 90, 86, 84, 81, 80, 78, 76, 75, 74, 72, 69, 68, 67, 65, 63, 62, 60, 
                  58, 57, 55, 53, 52, 49, 48, 47, 45, 44, 42, 41, 39, 38, 36, 33, 32, 31, 30, 29, 28, 27, 
                  25, 24, 23, 21, 19, 18, 17, 16, 15, 14, 13.8, 12.7, 11])

# Normalize position (scale to [0, 1])
position_normalized = (position - min(position)) / (max(position) - min(position))

# Normalize power (scale to [0, 1])
power_normalized = (power - min(power)) / (max(power) - min(power))

# Gaussian function for fitting
def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Fit the Gaussian to the normalized data
params, covariance = curve_fit(gaussian, position_normalized, power_normalized, p0=[max(power_normalized), np.mean(position_normalized), 0.3])

# Extract the fitted parameters
A_fit, mu_fit, sigma_fit = params

# Calculate the Full Width at Half Maximum (FWHM)
FWHM = 2.355 * sigma_fit

# Print the beam width (FWHM)
print(f"Beam Width (FWHM) = {FWHM:.4f} (normalized position units)")

# Optionally: Plotting the Gaussian fit and the normalized data
x_fit = np.linspace(mu_fit - 2*sigma_fit, mu_fit + 2*sigma_fit, 1000)
y_fit = gaussian(x_fit, *params)

plt.plot(position_normalized, power_normalized, 'bo', label='Normalized Data', markersize=5)
plt.plot(x_fit, y_fit, 'r-', label=f'Gaussian Fit: A={A_fit:.2f}, μ={mu_fit:.2f}, σ={sigma_fit:.2f}')
plt.axvline(x=mu_fit - sigma_fit, color='g', linestyle='--', label='-σ')
plt.axvline(x=mu_fit + sigma_fit, color='g', linestyle='--', label='+σ')
plt.axvline(x=mu_fit - 2*sigma_fit, color='m', linestyle='--', label='-2σ')
plt.axvline(x=mu_fit + 2*sigma_fit, color='m', linestyle='--', label='+2σ')

# Add annotation for FWHM on the plot
plt.annotate(f"FWHM = {FWHM:.4f}",
             xy=(mu_fit, max(y_fit) / 2), 
             xytext=(mu_fit + 0.1, max(y_fit) / 2),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=12, color='black')

plt.xlabel('Normalized Position')
plt.ylabel('Normalized Power')
plt.title('Gaussian Fit and Beam Width (FWHM)')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data (position in degrees and power in microwatts)
position = np.array([0, 360, 720, 1080, 1100, 1125, 1155, 1190, 1230, 1275, 1321, 1369, 1440, 1442, 
                     1448, 1457, 1467, 1479, 1493, 1509, 1527, 1547, 1569, 1593, 1619, 1648, 1679, 1716, 
                     1752, 1789, 1829, 1871, 1915, 1961, 2009, 1800, 2160, 2164, 2172, 2182, 2194, 2208, 
                     2224, 2242, 2262, 2284, 2318, 2348, 2383, 2421, 2461, 2506, 2510, 2515, 2525, 2545, 2580])


power = np.array([98, 96, 97, 94, 93, 90, 86, 84, 81, 80, 78, 76, 75, 74, 72, 69, 68, 67, 65, 63, 62, 60, 
                  58, 57, 55, 53, 52, 49, 48, 47, 45, 44, 42, 41, 39, 38, 36, 33, 32, 31, 30, 29, 28, 27, 
                  25, 24, 23, 21, 19, 18, 17, 16, 15, 14, 13.8, 12.7, 11])

# Normalize position (scale to [0, 1])
position_normalized = (position - min(position)) / (max(position) - min(position))

# Normalize power (scale to [0, 1])
power_normalized = (power - min(power)) / (max(power) - min(power))

# Gaussian function for fitting
def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Fit the Gaussian to the normalized data
params, covariance = curve_fit(gaussian, position_normalized, power_normalized, p0=[max(power_normalized), np.mean(position_normalized), 0.3])

# Extract the fitted parameters
A_fit, mu_fit, sigma_fit = params

# Calculate the Full Width at Half Maximum (FWHM)
FWHM = 2.355 * sigma_fit

# Print the beam width (FWHM)
print(f"Beam Width (FWHM) = {FWHM:.4f} (normalized position units)")

# Calculate the beam waist (w_0) from the Gaussian fit and the FWHM
w_0 = FWHM / 2.355

# Approximate the Rayleigh range (z_R) using the Gaussian beam relation
# w_0 = sqrt(λ * z_R / π)
# So, solving for λ: λ = w_0^2 * π / z_R
# Here, we approximate the Rayleigh range z_R as 1 for simplicity
z_R = 1  # Can be adjusted if known

# Calculate the wavelength λ
wavelength = (w_0**2 * np.pi) / z_R

# Print the estimated wavelength
print(f"Estimated Wavelength (λ) = {wavelength:.4e} (normalized units)")

# Optionally: Plotting the Gaussian fit and the normalized data
x_fit = np.linspace(mu_fit - 2*sigma_fit, mu_fit + 2*sigma_fit, 1000)
y_fit = gaussian(x_fit, *params)

plt.plot(position_normalized, power_normalized, 'bo', label='Normalized Data', markersize=5)
plt.plot(x_fit, y_fit, 'r-', label=f'Gaussian Fit: A={A_fit:.2f}, μ={mu_fit:.2f}, σ={sigma_fit:.2f}')
plt.axvline(x=mu_fit - sigma_fit, color='g', linestyle='--', label='-σ')
plt.axvline(x=mu_fit + sigma_fit, color='g', linestyle='--', label='+σ')
plt.axvline(x=mu_fit - 2*sigma_fit, color='m', linestyle='--', label='-2σ')
plt.axvline(x=mu_fit + 2*sigma_fit, color='m', linestyle='--', label='+2σ')
plt.xlabel('Normalized Position')
plt.ylabel('Normalized Power')
plt.title('Gaussian Fit and Beam Width (FWHM)')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
