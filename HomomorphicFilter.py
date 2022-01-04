import numpy as np

# Homomorphic filter class
class HomomorphicFilter:
  
    def __init__(self, a=0.5, b=1.5):
        self.a = float(a)
        self.b = float(b)

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        crow = I_shape[0]/2
        ccol = I_shape[1]/2
        # make a grid
        U, V = np.meshgrid(range(I_shape[0]), range(
            I_shape[1]), sparse=False, indexing='ij')
        mask_area = (((U-crow)**2+(V-ccol)**2)).astype(float)
        H = 1/(1+(mask_area/filter_params[0]**2)**filter_params[1])
        return (1 - H)

    # Methods

    def __apply_filter(self, I, H):  # I --> FFT
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b*H)*I
        return I_filtered

    def filter(self, I, filter_params, filter='butterworth', H=None):
       

        #  Validating image
        if len(I.shape) != 2:
            raise Exception('Improper image')

        # Take the image to log domain and then to frequency domain
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        # Filters
        H = self.__butterworth_filter(
            I_shape=I_fft.shape, filter_params=filter_params)

        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I=I_fft, H=H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt))-1
        return np.uint8(I)

