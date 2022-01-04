import numpy as np
import cv2
import matplotlib.pyplot as plt


def show(src, img_back, magnitude_spectrum, fshift_mask_mag):
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(src, cmap='gray')
    ax1.title.set_text('Input Image')
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(magnitude_spectrum, cmap='gray')
    ax2.title.set_text('FFT of image')
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(fshift_mask_mag, cmap='gray')
    ax3.title.set_text('FFT + Mask')
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(img_back, cmap='gray')
    ax4.title.set_text('After inverse FFT')
    plt.show()


class HighPass:
    def high_pass_flitter(imagePath):
        src = cv2.imread(imagePath, 0)
        dft = cv2.dft(np.float32(src), flags=cv2.DFT_COMPLEX_OUTPUT)
        # SHIFT FROM THE TOP TO THE CENTER
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * \
            np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

        # high-pass filter is a filter that attenuates low frequencies while "passing" high frequencies.
        # create a new array mask with the same size of the orignal image
        rows, cols = src.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        # fill the mask with 1
        mask = np.ones((rows, cols, 2), np.uint8)
        r = 120  # r is radius
        # go to the center and fill the radius with 0
        center = [crow, ccol]  # find the center
        x, y = np.ogrid[:rows, :cols]

        # circle center
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 0
        # apply mask and inverse DFT: Multiply fourier transformed image
        # with the mask values.
        fshift = dft_shift * mask

        # Get the magnitude spectrum (only for plotting purposes)
        fshift_mask_mag = 20 * \
            np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

        # Inverse shift to shift origin back to top left.
        f_ishift = np.fft.ifftshift(fshift)

        # Inverse DFT to convert back to image domain from the frequency domain.
        # Will be complex numbers
        img_back = cv2.idft(f_ishift)

        # Magnitude spectrum of the image domain
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        show(src, img_back, magnitude_spectrum, fshift_mask_mag)
