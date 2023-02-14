# fourier_synthesis.py

import numpy as np
import matplotlib.pyplot as plt
import cv2

def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)

def calculate_2dift(input):
    ift = np.fft.ifftshift(input)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    return ift.real

def calculate_distance_from_centre(coords, centre):
    # Distance from centre is √(x^2 + y^2)
    return np.sqrt(
        (coords[0] - centre) ** 2 + (coords[1] - centre) ** 2
    )

def find_symmetric_coordinates(coords, centre):
    return (centre + (centre - coords[0]),
            centre + (centre - coords[1]))

def display_plots(individual_grating, reconstruction, idx):
    plt.subplot(121)
    plt.imshow(individual_grating)
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(reconstruction)
    plt.axis("off")
    plt.suptitle(f"Terms: {idx}")
    plt.pause(0.01)
    
#Image_res is simply just setting the resolution of the image by powers of 2
# 8 = 256*256, 9 = 512*512, 10 = 1024 * 1024, etc 
# Higher number will mean longer computation time
image_res = 8
dim = (2**image_res, 2**image_res)

#convert image and resize to square (This is actaully squashing the image into a square)
image = cv2.imread("Image_file_location")
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

#Code for the next few lines is used for the computations
image_size = image[:, :, :3].mean(axis=2)  # Convert to grayscale

# Array dimensions (array is square) and centre pixel
# Use smallest of the dimensions and ensure it's odd
array_size = min(image_size.shape) - 1 + min(image_size.shape) % 2

# Crop image so it's a square image
image = image[:array_size, :array_size]
centre = int((array_size - 1) / 2)

#Back to colored image

# Convert the image to color
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.axis("off")
plt.imshow(img)
plt.show()
plt.clf()

# Split the image into color channels
r_img, g_img, b_img = cv2.split(img)


# Get all coordinate pairs in the left half of the array,
# including the column at the centre of the array (which
# includes the centre pixel)
coords_left_half = (
    (x, y) for x in range(array_size) for y in range(centre+1)
)

# Sort points based on distance from centre
coords_left_half = sorted(
    coords_left_half,
    key=lambda x: calculate_distance_from_centre(x, centre)
)


ft_red = calculate_2dft(r_img)
ft_green = calculate_2dft(g_img)
ft_blue = calculate_2dft(b_img)

# Show grayscale image and its Fourier transform
rows, cols = 2, 3
plt.subplot(rows, cols, 5)
plt.imshow(np.uint8(img))
plt.axis("off")
plt.subplot(rows, cols, 1)
plt.imshow(np.log(abs(ft_red)))
plt.set_cmap("Reds")
plt.axis("off")
plt.subplot(rows, cols, 2)
plt.imshow(np.log(abs(ft_green)))
plt.set_cmap("Greens")
plt.axis("off")
plt.subplot(rows, cols, 3)
plt.imshow(np.log(abs(ft_blue)))
plt.set_cmap("Blues")
plt.axis("off")
plt.show()

image = b_img

# Reconstruct image
fig = plt.figure()
fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 2]})

# Step 1
# Set up empty arrays for each color channels

rec_image = np.zeros(image.shape)
rec_image_green = np.zeros(image.shape)
rec_image_blue = np.zeros(image.shape)

# individual gratings for each color channel
individual_grating = np.zeros(
    image.shape, dtype="complex"
)

individual_grating_green = np.zeros(
    image.shape, dtype="complex"
)

individual_grating_blue = np.zeros(
    image.shape, dtype="complex"
)
idx = 0

# All steps are displayed until display_all_until value
display_all_until = 100
# After this, skip which steps to display using the
# display_step value
display_step = 10
# Work out index of next step to display
next_display = display_all_until + display_step

# Step 2
for coords in coords_left_half:
    # Central column: only include if points in top half of
    # the central column
    if not (coords[1] == centre and coords[0] > centre):
        idx += 1
        symm_coords = find_symmetric_coordinates(
            coords, centre
        )
        # Step 3
        # Copy values from Fourier transform into
        # individual_grating for the pair of points in
        # current iteration
        individual_grating[coords] = ft_red[coords]
        individual_grating[symm_coords] = ft_red[symm_coords]

        individual_grating_green[coords] = ft_green[coords]
        individual_grating_green[symm_coords] = ft_green[symm_coords]

        individual_grating_blue[coords] = ft_blue[coords]
        individual_grating_blue[symm_coords] = ft_blue[symm_coords]
        # Step 4
        # Calculate inverse Fourier transform to give the
        # reconstructed grating. Add this reconstructed
        # grating to the reconstructed ima(individual_grating)ge

        rec_grating = calculate_2dift(individual_grating)

        rec_grating_green = calculate_2dift(individual_grating_green)

        rec_grating_blue = calculate_2dift(individual_grating_blue)

        rec_image += rec_grating
        rec_image_green += rec_grating_green
        rec_image_blue += rec_grating_blue

        # Clear individual_grating array, ready for
        # next iteration
        individual_grating[coords] = 0
        individual_grating[symm_coords] = 0

        individual_grating_green[coords] = 0
        individual_grating_green[symm_coords] = 0

        individual_grating_blue[coords] = 0
        individual_grating_blue[symm_coords] = 0

        # Don't display every step
        if idx < display_all_until or idx == next_display:
            if idx > display_all_until:
                next_display += display_step
                # Accelerate animation the further the
                # iteration runs by increasing
                # display_step
                display_step += 10
            rows, cols = 3, 3
            plt.subplot(rows, cols, 1)
            plt.imshow(rec_grating)
            plt.set_cmap("Reds")
            plt.axis("off")
            plt.subplot(rows, cols, 4)
            plt.imshow(rec_grating_green)
            plt.set_cmap("Greens")
            plt.axis("off") 
            plt.subplot(rows, cols, 7)
            plt.imshow(rec_grating_blue)
            plt.set_cmap("Blues")
            plt.axis("off")
            #Combine all channels
            img_combined = np.uint8(np.dstack((rec_image, rec_image_green, rec_image_blue)))
            ax[1].imshow(img_combined)
            ax[1].set_title("Term " + str(idx))
            ax[1].set_axis_off()
            plt.axis("off")
            plt.pause(0.01)
            #If you want to save each image then get rid of # bellow. Change DPI to increase resolution
            #plt.savefig("img" + str(idx), dpi = 300)

plt.show()
