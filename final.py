from scipy.stats import chisquare, chi2
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, curve_fit, minimize_scalar
from scipy import signal
from scipy.special import spherical_jn
from mpl_toolkits.mplot3d import Axes3D

#settings
ROTATE_IMAGE = False

PLOT_UNIQUE_INTENSITIES = False
PLOT_IMAGE_INITIAL = True
PLOT_NORMALISED_HIST = True

CLIP_DATA = True

NORMALISE_AUTOCORRELATION = False
CLIP_DATA_3DPLOT = False

#importing data from excel file
path_link = r"C:\Users\GAdmin\Desktop\srp\beam.xlsx"
Data_x = pd.read_excel(path_link)
Data_x = Data_x[['Distance from lens mm', 'beam radius mm', 'stddev mm']]

source_list = [
    r"C:\Users\GAdmin\Desktop\srp\Hemanya and Sujen\speckle pictures june 13\with iris\001.csv",
    # r"C:\Users\GAdmin\Desktop\srp\Hemanya and Sujen\speckle pictures june 13\with iris\002.csv",
    # r"C:\Users\GAdmin\Desktop\srp\Hemanya and Sujen\speckle pictures june 13\with iris\003.csv",
    # r"C:\Users\GAdmin\Desktop\srp\Hemanya and Sujen\speckle pictures june 13\with iris\004.csv",
    # r"C:\Users\GAdmin\Desktop\srp\Hemanya and Sujen\speckle pictures june 13\with iris\005.csv"
]

pic_list = []

def get_pics(source_list):
    for source in source_list:
        data = pd.read_csv(source, header=None, delimiter=r"\s+").to_numpy()
        pic_list.append(data)

    return pic_list

pic_list = get_pics(source_list)

x_values = Data_x['Distance from lens mm'] * 10**-3 #m
y_values = Data_x['beam radius mm'] * 10**-6        #m
stddev = Data_x['stddev mm'] * 10**-6               #m

#curve fitting
wavelength = 532 * 10**-9 #m

def waist_func(z, w0, z0, Zrm):
    return w0 * np.sqrt(1 + (((z - z0) ** 2) / (Zrm**2)))

popt, pcov = curve_fit(waist_func, x_values, y_values, p0 = [0.0002, 0.6, 0.02])

print("Optimal parameters", popt)

w0, z0, Zrm = popt

unc_w0 = pcov[0,0]**0.5
unc_z0 = pcov[1,1]**0.5
unc_Zrm = pcov[2,2]**0.5

dx = np.linspace(0.03992, 1.10976, 10000)
y_fit = waist_func(dx, w0, z0, Zrm)

#plotting equation with fitted variables (X axis) 
plt.xlabel("distance from lens (m)")
plt.ylabel("width of beam (m)")
plt.title("width of beam against distance from lens")
plt.plot(dx, y_fit)
plt.errorbar(x_values, y_values, fmt = 'o',  yerr=stddev, capsize=3, ecolor = "black")
plt.show()

#Calculating parameters
def parameters(w0, Zrm):
    beam_divergence_angle = w0/Zrm
    unc_bda = (unc_w0 + unc_Zrm) * beam_divergence_angle
    m_squared_value = (np.pi*w0**2)/(wavelength*Zrm)
    unc_m2 = (2 * unc_w0 + unc_Zrm) * m_squared_value

    return beam_divergence_angle, m_squared_value, unc_bda, unc_m2

beam_divergence_angle, m_squared_value, unc_bda, unc_m2 = parameters(w0, Zrm)

print("waist: ", w0, "m +-", unc_w0)
print("rayleigh range: ", Zrm, "m +-", unc_Zrm / Zrm * 100)
print("beam divergence angle: ", beam_divergence_angle, "m +-", unc_bda)
print("M^2 value: ", m_squared_value, "m +-", unc_m2)

y_pred = waist_func(x_values, w0, z0, Zrm)
y_meas = y_values
errors = y_meas - y_pred
chi_sq = np.sum(((y_meas - y_pred) ** 2) / y_pred)
reduced_chisq = chi_sq / (len(y_meas)-1)
reduced_chisq

#camera stuff
sensor_size = pic_list[0].shape
print(sensor_size)

#rotational transform of matrix to compensate for bending of lens
if ROTATE_IMAGE:
    from skimage.transform import rotate
    rotation_angle = -30
    im_matrix_4_rotated = rotate(im_matrix_4, rotation_angle)
    plt.imshow(im_matrix_4)
    plt.show()
    plt.imshow(im_matrix_4_rotated)

#plot images, hist, hist-normalised for all pictures
def plot_images_and_histograms(pic_list):
    for i, im_matrix in enumerate(pic_list):

        #image
        if PLOT_IMAGE_INITIAL:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            im = plt.imshow(im_matrix, cmap='gray', aspect='auto')
            plt.title(f"image - {i+1}")
            plt.colorbar(im)

        #hist
        if PLOT_UNIQUE_INTENSITIES:
            unique_intensities, counts = np.unique(im_matrix.flatten(), return_counts=True)
            plt.subplot(1, 2, 2)
            plt.scatter(unique_intensities, counts, s=2)
            plt.title(f"hist - Image {i+1}")
            plt.xlabel("value")
            plt.ylabel("frequency")
            plt.grid(True)
            plt.show()

        #hist-normalised
        if PLOT_NORMALISED_HIST:
            counts, bins = np.histogram(im_matrix, bins=100)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            plt.figure(figsize=(10, 5))
            plt.bar(bin_centers, counts/counts.max(), width=np.diff(bins), align='center', edgecolor='black')
            plt.title(f"normalised hist - Image {i+1}")
            plt.xlabel("value")
            plt.ylabel("frequency")
            plt.grid(True)
            plt.scatter(bin_centers, counts/counts.max(), c='orange', marker='x', label='Counts')
            plt.legend()
            plt.tight_layout()
            plt.show()

plot_images_and_histograms(pic_list)

#pdf intensity
def intensity_probability_curve(x_o, a, b, c):
    return (1/(a)) * np.exp(-(x_o - b) / (a) + c)

def curve_fitting_pics(pic_list):
    highest_intensity_recorded = 80 * 10**-6
    area_of_pixel = 8.1 * 10**-11
    
    for i, im_matrix in enumerate(pic_list):
        #normalise intensity to highest value recorded
        highest_pixel_density = np.max(im_matrix)
        normalisation_factor = (highest_intensity_recorded / highest_pixel_density) * 1 / area_of_pixel

        #finding unique intensity values with counts
        unique_intensities, counts = np.unique(im_matrix.flatten(), return_counts=True)

        #binning
        counts_hist, bins = np.histogram(im_matrix, bins=100)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_centers_normalised = bin_centers * normalisation_factor

        #fitting
        fitted_parameters, cov = curve_fit(intensity_probability_curve, bin_centers, counts_hist / counts_hist.max(), p0=[174, 159, 4.11])
        a, b, c = fitted_parameters

        large_range_of_X_values = np.arange(0, np.max(bin_centers), 0.1)
        generated_y_values = intensity_probability_curve(large_range_of_X_values, a, b, c)
        

        if not CLIP_DATA:
            #plot with fit, initial data, normalised
            plt.subplot(1, 2, 2)
            plt.bar(bin_centers_normalised, counts_hist / counts_hist.max(), width=np.diff(bins) * normalisation_factor, align='center', edgecolor='black')
            plt.plot(large_range_of_X_values * normalisation_factor, generated_y_values, label='Fitted Curve')
            plt.scatter(bin_centers_normalised, counts_hist / counts_hist.max(), c='orange', marker='x', label='Counts')
            plt.xlabel('Intensity per square meter')
            plt.ylabel('Counts / Intensity Probability')
            plt.title(f'Curve Fitting on Histogram Data - Image {i+1}')
            plt.text(0.5, 0.5, f"Fitted average intensity: {np.round(a * normalisation_factor, 2)}\nHorizontal offset value: {np.round(b * normalisation_factor, 2)}\nVertical offset value: {np.round(c, 2)}", transform=plt.gca().transAxes)
            plt.legend()
            plt.grid(True)
            plt.show()

        else:
            #clip points less than max
            position_of_largest_value = np.argmax(counts_hist)
            counts_new = np.delete(counts_hist, range(0, position_of_largest_value))
            bin_centers_new = np.delete(bin_centers, range(0, position_of_largest_value))

            #second curve fitting without initial data points
            fitted_parameters_new, cov_new = curve_fit(intensity_probability_curve, bin_centers_new, counts_new / counts_hist.max(), p0=[170, -800, 4.0])
            A, B, C = fitted_parameters_new

            #y-values for the new fitted curve
            large_range_of_X_values_new = np.arange(bin_centers_new[0], np.max(bin_centers_new), 0.1)
            generated_y_values_new = intensity_probability_curve(large_range_of_X_values_new, A, B, C)

            #new normalized histogram with new curve fit
            plt.bar(bin_centers_new * normalisation_factor, counts_new / counts_hist.max(), width=np.diff(bins[position_of_largest_value:]) * normalisation_factor, align='center', edgecolor='black')
            plt.plot(large_range_of_X_values_new * normalisation_factor, generated_y_values_new, label='Fitted Curve', c='red')
            plt.scatter(bin_centers_new * normalisation_factor, counts_new / counts_hist.max(), c='orange', marker='x', label='Counts')
            plt.xlabel('Intensity per square meter')
            plt.ylabel('Counts / Intensity Probability')
            plt.title(f'Curve Fitting on Histogram Data after Removing Initial Data - Image {i+1}')
            plt.text(0.5, 0.5, f"Fitted average intensity: {np.round(A * normalisation_factor, 2)}\nHorizontal offset value: {np.round(B * normalisation_factor, 2)}\nVertical offset value: {np.round(C, 2)}", transform=plt.gca().transAxes)
            plt.legend()
            plt.grid(True)
            plt.show()

curve_fitting_pics(pic_list)

#vertical coeff offset - account for stray light
def bessel_function(distance_dist, I_ave, B, A, C):
    return (I_ave**2) * (1 + ((2 * B * spherical_jn(0, A * distance_dist)) / A)**2) + C

def correlation(pic_list):
    for i, im_matrix in enumerate(pic_list):

        #compute autocorrelation
        correlation = signal.correlate(im_matrix, im_matrix)
        correlation_dist_x = signal.correlation_lags(len(im_matrix), len(im_matrix))
        correlation_dist_y = signal.correlation_lags(624, 624)
        L, B = np.shape(correlation)
        
        #finding midpoint
        correlation_plotted_X = correlation[::1, int(((B + 1) / 2) - 1)]
        correlation_plotted_y = correlation[int(((L + 1) / 2) + 1), ::1]

        if NORMALISE_AUTOCORRELATION:
            correlation_plotted_X /= np.max(correlation_plotted_X)
            correlation_plotted_y /= np.max(correlation_plotted_y)
        
        # Plot autocorrelation in the x-axis
        plt.figure(figsize=(10, 6))
        plt.scatter(correlation_dist_x, correlation_plotted_X, label='autocorrelation - x', s=2)
        plt.grid(True); plt.legend(); plt.ylabel("Autocorrelation"); plt.xlabel("Distance "); plt.title(f"Graph of autocorrelation against distance  - image {i + 1} (x axis)")
        
        #autocorrelation in the y-axis
        plt.figure(figsize=(10, 6))
        plt.scatter(correlation_dist_y, correlation_plotted_y, label='autocorrelation - y', s=1)
        plt.grid(True); plt.legend() ;plt.ylabel("Autocorrelation") ;plt.xlabel("Distance ")  ;plt.title(f"Graph of autocorrelation against Distance  - image {i + 1} (y axis)") ;plt.show()

        #modifying distance and autocorrelation arrays - finds the max value of autocorrelation function and extracts the surrounding values in the vicinity
        max_value_autocorrelation  = np.max(correlation_plotted_X)
        max_value_autocorrelation_pos = np.where(correlation_plotted_X == max_value_autocorrelation) 
        max_value_autocorrelation_pos = int(max_value_autocorrelation_pos[0])
        correlation_plotted_X_fit = correlation_plotted_X[max_value_autocorrelation_pos-7:max_value_autocorrelation_pos+8]
        correlation_dist_x_fit = correlation_dist_x[max_value_autocorrelation_pos-7:max_value_autocorrelation_pos+8]

        #fit bessel func and plot
        p0 = [2.2449*10**2, 3.57615*10**2, -1.771, 8.623*10**9]
        Fitted_parameters_autocorrelation, _ = curve_fit(bessel_function, correlation_dist_x_fit, correlation_plotted_X_fit, p0=p0)
        I_ave, B, A, C = Fitted_parameters_autocorrelation
        print(I_ave)

        large_number_X_values_autocorrelation = np.arange(np.min(correlation_dist_x_fit), np.max(correlation_dist_x_fit), 0.0001)
        fitted_autocorrelation_values = bessel_function(large_number_X_values_autocorrelation, I_ave, B, A, C)

        plt.figure(figsize=(10, 6))
        plt.plot(large_number_X_values_autocorrelation, fitted_autocorrelation_values, c='blue', label='Fitted Bessel Function')
        plt.scatter(correlation_dist_x_fit, correlation_plotted_X_fit, c='orange', label='Autocorrelation')
        plt.grid(True); plt.legend(); plt.ylabel("Autocorrelation"); plt.xlabel("Distance "); plt.title(f"Graph of autocorrelation against distance  - image {i + 1} (x axis)"); plt.show()

        #extracting relevant parameters, adding y coeff offset to the bessel function (find fwhm)
        fitted_autocorrelation_values_analyse = fitted_autocorrelation_values - C
        max_value = np.max(fitted_autocorrelation_values_analyse)
        half_max_value = max_value/2
        tolerance = 1000000
        half_max_values = np.where(np.abs(fitted_autocorrelation_values_analyse - half_max_value) < tolerance)[0]
        half_max_value_neg = half_max_values[0]
        half_max_value_pos = half_max_values[len(half_max_values) - 1]
        fwhm = large_number_X_values_autocorrelation[half_max_value_pos] - large_number_X_values_autocorrelation[half_max_value_neg]

        #minimum point and finding autocorr length
        minimum_point_neg = minimize_scalar(lambda x: bessel_function(x, I_ave, B, A, C), bounds=(-3, 0))
        minimum_point_pos = minimize_scalar(lambda x: bessel_function(x, I_ave, B, A, C), bounds=(0, 3))
        autocorrelation_length = np.round(minimum_point_pos.x - minimum_point_neg.x, 3)

        #final function with the measurements
        plt.figure(figsize=(10, 6))
        plt.plot(large_number_X_values_autocorrelation, fitted_autocorrelation_values_analyse, label='fitted bessel function')
        plt.scatter(minimum_point_pos.x, bessel_function(minimum_point_pos.x, I_ave, B, A, C) - C, label='left min point')
        plt.scatter(minimum_point_neg.x, bessel_function(minimum_point_neg.x, I_ave, B, A, C) - C, label='right min point')
        print("autocorrelation length: ", autocorrelation_length); print("FWHM: ", fwhm)
        plt.grid(True); plt.legend(); plt.ylabel("Autocorrelation"); plt.xlabel("Distance "); plt.title(f"Graph of autocorrelation against distance  - image {i + 1} (x axis)"); plt.show()

        #correlation matrix
        plt.figure(figsize=(10, 6))
        plt.imshow(correlation)
        plt.title(f"Correlation Matrix - Image {i + 1}")
        plt.colorbar() ;plt.show()

        #3d plot of overall autocorrelation
        if CLIP_DATA_3DPLOT:
            clip = 600
            correlation_dist_x = correlation_dist_x[len(correlation_dist_x) // 2 - clip: len(correlation_dist_x) // 2 + clip]
            correlation_dist_y = correlation_dist_y[len(correlation_dist_y) // 2 - clip: len(correlation_dist_y) // 2 + clip]
            correlation_plotted_X = correlation_plotted_X[len(correlation_plotted_X) // 2 - clip: len(correlation_plotted_X) // 2 + clip]
            correlation_plotted_y = correlation_plotted_y[len(correlation_plotted_y) // 2 - clip: len(correlation_plotted_y) // 2 + clip]

        X, Y = np.meshgrid(correlation_dist_x, correlation_dist_y)
        autocorr2d = np.outer(correlation_plotted_X, correlation_plotted_y)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_surface(X, Y, autocorr2d.T, cmap='viridis', edgecolor='none')
        ax.set_title('autocorrelation')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Autocorrelation')
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, label='Autocorrelation')
        plt.show()

correlation(pic_list)