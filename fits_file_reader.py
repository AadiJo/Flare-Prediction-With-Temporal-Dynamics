from astropy.io import fits
import matplotlib.pyplot as plt

file_path = 'filepath.fits' 

try:
    with fits.open(file_path) as hdul:
        # Print information about the FITS file
        hdul.info()
        
        # Alternative approach to find the first HDU with data
        image_data = None
        header = None
        for i, hdu in enumerate(hdul):
            if hdu.data is not None:
                print(f"Found image data in extension {i}")
                image_data = hdu.data
                header = hdu.header
                break

        if image_data is None:
            raise ValueError("No image data found in this FITS file")

        print("\nImage Data Shape:", image_data.shape)
        print("Example Header Keyword (NAXIS1):", header.get('NAXIS1', 'Not found'))

        # Display the image
        plt.figure(figsize=(10, 8))
        plt.imshow(image_data, cmap='viridis')
        plt.colorbar(label='Pixel Value')
        plt.title('FITS Image')
        plt.show()

except FileNotFoundError:
    print(f"Error: FITS file not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")