import numpy as np
import cv2
import matplotlib.pyplot as plt


class Histogram:
    def histogram_equalization(imagePath):
        src = cv2.imread(imagePath, 0)
        dft = cv2.dft(np.float32(src), flags=cv2.DFT_COMPLEX_OUTPUT)
        # SHIFT FROM THE TOP TO THE CENTER
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * \
            np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

        # read image and convert it to gray scal image
        src = cv2.imread(imagePath)

        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # store the output of the equalized image
        dst = cv2.equalizeHist(src)
        cv2.imwrite('output-imgs/equalized-img.png', dst)

        # show the histogram of the source image
        srcHistogram = cv2.calcHist([src], [0], None, [256], [0, 256])

        # plt.plot(srcHistogram)
        # plt.show()

        # show the histogram of the equalized image
        dstHistogram = cv2.calcHist([dst], [0], None, [256], [0, 256])

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(src, cmap='gray')
        ax1.title.set_text('Input Image')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(dst, cmap='gray')
        ax2.title.set_text('Equalized Image')
        ax3 = fig.add_subplot(2, 2, 3)
        plt.plot(srcHistogram)
        ax3.title.set_text('histogram of the source image')
        ax4 = fig.add_subplot(2, 2, 4)
        plt.plot(dstHistogram)
        ax4.title.set_text('histogram of the equalized image')
        plt.show()

