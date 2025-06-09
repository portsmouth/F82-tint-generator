

import numpy as np
import colour
import math, os
from colour.models import RGB_COLOURSPACES
import matplotlib.pyplot as plt
import pandas as pd


IOR_FILE_DIR = "ior_data/"

############################################################################################
# IOR data reading utils, from https://github.com/natyh/material-params
############################################################################################

standard_lambda_sampling = np.arange(360, 781)

def read_rii_format_ior_file(data_idx, iorfilelist):

    # Assuming that data_idx is the index of the start of the numerical data (triples
    # of float numbers).
    wavelength_num = (len(iorfilelist) - data_idx) // 3

    # Read 'wavelength_num' wavelengths and pairs of numbers (convert wavelengths from um to nm)
    wavelength_vals = np.empty(wavelength_num, dtype=np.float64)
    c_ior_data = np.empty(wavelength_num, dtype=np.complex64)
    for i in range(0, wavelength_num):
        wavelength_vals[i] = 1000.0 * float(iorfilelist[i*3+data_idx])
        c_ior_data[i] = complex(float(iorfilelist[i*3+(data_idx+1)]), float(iorfilelist[i*3+(data_idx+2)]))

    return wavelength_vals, c_ior_data

def read_ior_file(ior_filename, resample=True):
    """
    Read common spectral IOR file formats and return two arrays: one of wavelengths and one of complex IOR values.

    Parameters
    ----------
    ior_filename : string
        Path of spectral ior file.
    resample : bool, optional (True by default)
        Whether to resample the spectral IOR to a regular 1nm spacing from 360 nm to 780 nm.
    """
    # First put all whitespace-separated strings in a list and then iterate over the list
    with open(ior_filename) as f:
        iorfilelist = []
        for line in f:
            iorfilelist.extend(line.replace(',',' ').split()) # some files use commas, so replace commas with spaces

    # There are two flavors of RII format - in one, the numerical data is preceded by "DATA nk", and in the other, it
    # is preceded by "data: |"
    data_idx = iorfilelist.index("data:")+2 if "data:" in iorfilelist else (iorfilelist.index("DATA")+2 if "DATA" in iorfilelist else -1)
    if (data_idx == -1):
        wavelength_vals, c_ior_data = read_sopra_format_ior_file(iorfilelist)
    else:
        wavelength_vals, c_ior_data = read_rii_format_ior_file(data_idx, iorfilelist)

    # Optionally use linear interpolation to resample the spectral complex ior into a 1nm sampling between 360 and 780nm.
    if resample:
        c_ior_data = np.interp(standard_lambda_sampling, wavelength_vals, c_ior_data).astype('complex64')
        wavelength_vals = np.copy(standard_lambda_sampling)

    return wavelength_vals, c_ior_data

############################################################################################
# Fresnel utils, from https://github.com/natyh/material-params
############################################################################################

def fresnel_reflectance(complex_ior, theta_i, ior_external=1.0):
    """
    Compute Fresnel reflectance (for a single wavelength) from complex IOR and angle of incidence.

    Parameters
    ----------
    complex_ior : complex number
        Complex index of refraction.
    theta_i: numeric
        Angle of incidence in degrees.
    ior_external:
        IOR of ambient medium
    """

    # Math from Sebastien Lagarde's blog post on Fresnel.
    complex_ior = complex_ior / ior_external
    sinTheta = np.sin(np.deg2rad(theta_i))
    sinTheta2 = sinTheta * sinTheta
    cosTheta2 = 1.0 - sinTheta2
    cosTheta = np.sqrt(cosTheta2)
    eta2 = complex_ior.real * complex_ior.real
    etak2 = complex_ior.imag * complex_ior.imag
    temp0 = eta2 - etak2 - sinTheta2
    a2plusb2 = np.sqrt(temp0 * temp0 + 4.0 * eta2 * etak2)
    temp1 = a2plusb2 + cosTheta2
    a = np.sqrt(0.5 * (a2plusb2 + temp0))
    temp2 = 2.0 * a * cosTheta
    Rs = (temp1 - temp2) / (temp1 + temp2)
    temp3 = cosTheta2 * a2plusb2 + sinTheta2 * sinTheta2
    temp4 = temp2 * sinTheta2
    Rp = Rs * (temp3 - temp4) / (temp3 + temp4)
    return np.float64(0.5 * (Rp + Rs))


def spectral_ior_to_spd_fresnel(wavelength_vals, c_ior_data, theta_i, ior_external = 1.0):

    assert len(wavelength_vals) == len(c_ior_data), "Vectors are not the same length."
    f_data = np.empty(len(c_ior_data), dtype=np.float64)
    for i in range(len(c_ior_data)):
        f_data[i] = fresnel_reflectance(c_ior_data[i], theta_i, ior_external)
    spectral_dict = dict(zip(wavelength_vals, f_data))
    return colour.SpectralDistribution(spectral_dict, name="Sample")


def schlick_approx_fresnel(F0, theta_i):

    cosTheta = np.cos(np.deg2rad(theta_i))
    cosTheta = max(cosTheta, 0.0)
    omc = 1.0 - cosTheta
    omc2 = omc * omc
    omc5 = omc2 * omc2 * omc
    return F0 + (1.0 - F0) * omc5

############################################################################################

cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
theta_82 = np.rad2deg(math.acos(1.0/7.0))

def compute_metal_colors(ior_file, colorspace):

    # Get IOR and k versus frequency
    wavelength_vals, c_ior_data = read_ior_file(ior_file)

    # Gives Fresnel reflectance at given angle, as a function of wavelength
    F0_spectral  = spectral_ior_to_spd_fresnel(wavelength_vals, c_ior_data, 0.0)
    F82_spectral = spectral_ior_to_spd_fresnel(wavelength_vals, c_ior_data, theta_82)

    colorspace = colorspace.strip()
    if colorspace=="ACEScg":   illuminant = colour.SDS_ILLUMINANTS["D60"]
    elif colorspace=="sRGB":   illuminant = colour.SDS_ILLUMINANTS["D65"]
    else:
        print("need to specify illuminant for colorspace ", colorspace)
        quit()

    # Integrate Fresnel over CMFs to get XYZ color (for given angle)
    XYZ_F0 = colour.sd_to_XYZ(F0_spectral, cmfs=cmfs, illuminant=illuminant) / 100
    XYZ_F82 = colour.sd_to_XYZ(F82_spectral, cmfs=cmfs, illuminant=illuminant) / 100

    # Convert to RGB
    RGB_F0 = colour.XYZ_to_RGB(XYZ_F0, RGB_COLOURSPACES[colorspace])
    RGB_F82 = colour.XYZ_to_RGB(XYZ_F82, RGB_COLOURSPACES[colorspace])

    # Thus compute the F82-tint (specular_color) and F0 (base_color)
    FSchlick_82_RGB = np.array([schlick_approx_fresnel(RGB_F0[0], theta_82),
                                schlick_approx_fresnel(RGB_F0[1], theta_82),
                                schlick_approx_fresnel(RGB_F0[2], theta_82)])
    base_color = RGB_F0
    specular_color = RGB_F82 / FSchlick_82_RGB
    return (base_color, specular_color)


metals = {}

for ior_file in os.listdir(IOR_FILE_DIR):

    metal_name = os.path.splitext(ior_file)[0]
    ior_file_path = os.path.join(IOR_FILE_DIR, ior_file)

    base_color_sRGB,   specular_color_sRGB   = compute_metal_colors(ior_file_path, colorspace="sRGB")
    base_color_ACEScg, specular_color_ACEScg = compute_metal_colors(ior_file_path, colorspace="ACEScg")

    metal_props = {}
    metal_props["base_color"]     = {'sRGB':base_color_sRGB,     'ACEScg':base_color_ACEScg}
    metal_props["specular_color"] = {'sRGB':specular_color_sRGB, 'ACEScg':specular_color_ACEScg}

    metals[metal_name] = metal_props

    print('%s: \n\t\tF0 (sRGB) %s,\n\t\tF0 (ACEScg) %s,\n\t\tF82-tint (sRGB) %s,\n\t\tF82-tint (ACEScg) %s\n' %
          (metal_name,
           str(base_color_sRGB).strip('[]'),
           str(base_color_ACEScg).strip('[]'),
           str(specular_color_sRGB).strip('[]'),
           str(specular_color_ACEScg).strip('[]')))

data = {}
data['Metal']                 = []
data['F0 (sRGB)']             = []
data['F0 (ACEScg)']           = []
data['F82-tint (sRGB)']       = []
data['F82-tint (ACEScg)']     = []
data['F0 (sRGB color)']       = []
data['F82-tint (sRGB color)'] = []

np.set_printoptions(precision=3, suppress=True)

for metal, metal_props in metals.items():
    data['Metal'].append(metal)
    data['F0 (sRGB)'].append(str(metal_props["base_color"]['sRGB']).strip('[]'))
    data['F0 (ACEScg)'].append(str(metal_props["base_color"]['ACEScg']).strip('[]'))
    data['F82-tint (sRGB)'].append(str(metal_props["specular_color"]['sRGB']).strip('[]'))
    data['F82-tint (ACEScg)'].append(str(metal_props["specular_color"]['ACEScg']).strip('[]'))
    data['F0 (sRGB color)'].append('    ')
    data['F82-tint (sRGB color)'].append('    ')

df = pd.DataFrame()

df['Metal']                 = data['Metal']
df['F0 (sRGB)  ']           = data['F0 (sRGB)']
df['F0 (ACEScg)']           = data['F0 (ACEScg)']
df['F82-tint (sRGB)  ']     = data['F82-tint (sRGB)']
df['F82-tint (ACEScg)']     = data['F82-tint (ACEScg)']
df['F0 (sRGB color)']       = data['F0 (sRGB color)']
df['F82-tint (sRGB color)'] = data['F82-tint (sRGB color)']

fig = plt.figure()
ax=fig.gca()
ax.axis('off')
r,c = df.shape

cellColours=[['none']*c]*(1 + r)
cellColours[0] = ['lightgrey']*c
cellColours[1] = ['none']*c
cellColours[2] = ['none']*c
table = ax.table(cellText=np.vstack([df.columns, df.values]), fontsize=8,
                 cellColours=cellColours, bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.scale(1.5, 1.5)

for col in range(0, c):
        row = 0
        cell = table[row, col]
        cell.visible_edges = 'closed'
        cell = table[row, col]
        cell.set(color='lightgrey')
        cell.set(edgecolor='black')
        cell.set(fill = True)
        cell.set_text_props(weight='bold')


for row in range(1, r+1):

    metal = data['Metal'][row-1]
    metal_props = metals[metal]

    F0 = metal_props["base_color"]['sRGB']
    Fg = metal_props["specular_color"]['sRGB']
    F0 = np.clip(F0, 0.0, 1.0)
    Fg = np.clip(Fg, 0.0, 1.0)

    f0_color_cell = table[row, 5]
    f0_color_cell.set(color=F0)

    fg_color_cell = table[row, 6]
    fg_color_cell.set(color=Fg)

# need to draw here so the text positions are calculated
fig.canvas.draw()


plt.show()

