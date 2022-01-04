from tkinter.font import BOLD
from matplotlib.colors import cnames
import cv2
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from numpy.lib.type_check import imag
from Histogram import*
from BandPass import*
from HighPass import*
from LowPass import*
from HomomorphicFilter import*


# End of class HomomorphicFilter
if __name__ == "__main__":
    import cv2

    # Main code

photoChosen = 0
# Colors
backGround = "#041C32"

# GUI using tkinter libirary
# start of the GUI main loop
root = Tk()
root.title('Imgae Proccessing')
root.resizable(width=False, height=False)

# canvas
canvas = Canvas(root, width=650, height=450, bg=backGround)
canvas.grid(pady=3, padx=3)

# Functions


def open():  # function to open an image from a dirctory
    global myImage, srcId, dft, dft_shift, magnitude_spectrum, flags, photoChosen
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select A File", filetypes=(
        ("png files", "*.png"), ("jpeg files", "*.jpeg"), ("jpg files", "*.jpg")))
#     myLabel = Label(root, text=root.filename).pack()
    photoChosen = 1


def selectPhotoWarning():
    messagebox.showwarning(
        "Warning", "Please select a photo befor applying any filter")


def runBandPassFilter():
    if(photoChosen == 0):
        selectPhotoWarning()
    else:
        BandPass.band_pass_filter(root.filename)


def runHighPassFilter():
    if(photoChosen == 0):
        selectPhotoWarning()
    else:
        HighPass.high_pass_flitter(root.filename)


def runLowPassFilter():
    if(photoChosen == 0):
        selectPhotoWarning()
    else:
        LowPass.low_pass_filter(root.filename)


def runHomomoFilter():
    if(photoChosen == 0):
        selectPhotoWarning()
    else:
        img = cv2.imread(root.filename)[:, :, 0]
        homo_filter = HomomorphicFilter(a=1, b=1.5)
        img_filtered = homo_filter.filter(I=img, filter_params=[30, 2])
        cv2.imshow('HomomorphicFilter', img_filtered)
        cv2.waitKey()


def runHistoGramFilter():
    if(photoChosen == 0):
        selectPhotoWarning()
    else:
        Histogram.histogram_equalization(root.filename)


def lightTheme():
    global backGround
    backGround = "white"
    canvas.config(bg=backGround)
    photoSelect.config(bg=backGround, fg="#F0A500")
    filterSelect.config(bg=backGround, fg="#F0A500")


def darkTheme():
    global backGround
    backGround = "#041C32"
    canvas.config(bg=backGround)
    photoSelect.config(bg=backGround)
    filterSelect.config(bg=backGround)


# instructions
photoSelect = Label(
    root, text="Please Select a photo to apply the filters to it. ", bg=backGround, font=("Helvetica", 18), fg="#ECB365")

filterSelect = Label(
    root, text="Apply the filter you want from the button below. ", bg=backGround, font=("Helvetica", 18), fg="#ECB365")


# Buttons
openButton = Button(root, text="Open File", command=open,
                    padx=15, pady=5, bg="#EEB76B", fg=backGround, font=("Helvetica", 12, BOLD))

bandPassButton = Button(root, text="Band Pass Filter",
                        command=runBandPassFilter, padx=24, pady=15, bg="#EEB76B", fg=backGround, font=("Helvetica", 12, BOLD))

highPassButton = Button(root, text="High Pass Filter",
                        command=runHighPassFilter, padx=25, pady=15, bg="#EEB76B", fg=backGround, font=("Helvetica", 12, BOLD))

lowPassButton = Button(root, text="Low Pass Filter",
                       command=runLowPassFilter, padx=26, pady=15, bg="#EEB76B", fg=backGround, font=("Helvetica", 12, BOLD))

homomoButton = Button(root, text="Homomorphic Filter",
                      command=runHomomoFilter, padx=10, pady=15, bg="#EEB76B", fg=backGround, font=("Helvetica", 12, BOLD))

histoGramButton = Button(root, text="Histogram Filter",
                         command=runHistoGramFilter, padx=21, pady=15, bg="#EEB76B", fg=backGround, font=("Helvetica", 12, BOLD))

lightThemeButton = Button(root, text="Light", command=lightTheme,
                          bg="#EEB76B", fg=backGround, font=("Helvetica", 12, BOLD), padx=3, pady=3)

darkThemeButton = Button(root, text="Dark", command=darkTheme,
                         bg="#EEB76B", fg=backGround, font=("Helvetica", 12, BOLD), padx=4, pady=3)

# Windows

photoSelectWindow = canvas.create_window(325, 65, window=photoSelect)

filterSelectWindow = canvas.create_window(325, 200, window=filterSelect)


openWindow = canvas.create_window(325, 100, window=openButton)

bandPassWindow = canvas.create_window(525, 265, window=bandPassButton)

highPassWindow = canvas.create_window(325, 265, window=highPassButton)

lowPassWindow = canvas.create_window(125, 265, window=lowPassButton)

homomoWindow = canvas.create_window(225, 365, window=homomoButton)

histoGramWindow = canvas.create_window(425, 365, window=histoGramButton)

lightThemeWindow = canvas.create_window(30, 20, window=lightThemeButton)

darkThemeWindow = canvas.create_window(30, 55, window=darkThemeButton)

root.mainloop()
# End of the GUI main loop
