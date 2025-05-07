"""A module for manipulating the image and converting it to a waveform representation."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from PIL import Image, ImageFilter, ImageOps
from scipy.ndimage import zoom


class UnknownPixels:
    """A class for converting an image to a waveform representation."""

    def __init__(self, imgpath):
        """Intialise the object and prepare the image for rendering.

        Args:
            imgpath (str): The path to the image file.
        """
        self.imgpath = imgpath

        # Open the image file
        self.img = Image.open(self.imgpath)

        # Convert to grayscale
        self.img = self.img.convert("L")

        # Make the image square without distortion
        self._pad_to_square()

        # Attribute to store the resulting lines
        self.waveform = None

    @property
    def shape(self):
        """Get the shape of the image.

        Returns:
            tuple: The shape of the image.
        """
        return self.img.size

    @property
    def arr(self):
        """Get the image as a normalized numpy array.

        Returns:
            np.ndarray: The image as a normalized numpy array.
        """
        # Convert the image to a numpy array
        arr = np.array(self.img, dtype=np.float64)

        # Normalize the image to [0, 1]
        vmax = arr.max()
        arr = np.clip(arr, 0, vmax) / vmax

        return arr

    def smooth_image(self, rad):
        """Smooth the image using a Gaussian filter.

        Args:
            rad (int): The radius of the Gaussian filter.

        Returns:
            PIL.Image: The smoothed image.
        """
        # Smooth the image
        self.img = self.img.filter(ImageFilter.GaussianBlur(radius=rad))

    def _pad_to_square(self):
        """Pad the image to make it square without distortion.

        Returns:
            PIL.Image: The padded image.
        """
        # Nothing to do if we're already square
        width, height = self.shape
        if width == height:
            return self.img

        # Calculate padding
        if width > height:
            padding = (0, (width - height) // 2, 0, (width - height + 1) // 2)
        else:
            padding = ((height - width) // 2, 0, (height - width + 1) // 2, 0)

        return ImageOps.expand(self.img, border=padding, fill=(0, 0, 0))

    def show(self, arr, target_lines):
        """Plot the image as a waveform representation.

        Args:
            arr (np.ndarray): The image as a numpy array.
            target_lines (int): The number of lines to render along the y-axis.

        Returns:
            np.ndarray: The waveform representation of the image.
        """
        # Set up the figure
        fig, ax = plt.subplots(figsize=(1, 1), dpi=1080)
        ax.set_axis_off()

        # Plot the image
        ax.imshow(arr, cmap="gray", aspect="auto")

        # Show the plot
        plt.show()

        # Close the figure
        plt.close(fig)

    def plot_unknown_pleasures(
        self,
        contrast=10,
        target_lines=50,
        figsize=(8, 8),
        title="",
        outpath=None,
    ):
        """Plot the image in the style of Unknown Pleasures.

        Creates a representation of an image similar in style to Joy
        Division's seminal 1979 album Unknown Pleasures.

        Borrows some code from this matplotlib examples:
        https://matplotlib.org/stable/gallery/animation/unchained.html

        Args:
            contrast (float):
                The contrast, i.e. the maximum value in the image.
            target_lines (int):
                The target number of individual lines to use.
            figsize (tuple):
                The size of the figure to create.
            title (str):
                The title to add to the image. If None no title is added.
            outpath (str):
                The path to save the image to. If None no image is saved.
        """
        # Extract data
        data = self.arr

        # Normalise to the maximum and take the log10
        data /= np.max(data)
        data = np.log10(data)

        # Set any -np.inf values to zero (once renormalised)
        data[data == -np.inf] = -np.log10(contrast)

        # Define normalising function
        norm = Normalize(vmin=-np.log10(contrast), vmax=0.0)

        # Normalise data
        data = norm(data) * 5

        # Set any data <0.0 to zero
        data[data < 0.0] = 0.0

        # Unknown Pleasures works best with about 50 lines so reshape the data
        # to have approximately 50 lines.

        # Calcualate the eventual number of lines.
        nlines = int(data.shape[0] / (data.shape[0] // target_lines))

        # Resample the data to have the same x resolution but nlines y
        # resolution.
        data = zoom(
            data,
            (nlines / data.shape[0], 1),
            order=3,
        )

        # Create new Figure with black background
        fig = plt.figure(figsize=figsize, facecolor="black")

        # Add a subplot with no frame
        ax = fig.add_subplot(111, frameon=False)

        # Create the x axis
        X = np.linspace(-1, 1, data.shape[-1])

        # Generate line plots
        lines = []
        for i in range(data.shape[0]):
            # Small reduction of the X extents to get a cheap perspective
            # effect.
            xscale = 1 - i / 100.0
            # Same for linewidth (thicker strokes on bottom)
            lw = 1.5 - i / 100.0

            (line,) = ax.plot(xscale * X, i + data[i], color="w", lw=lw)
            lines.append(line)

        # Set y limit (or first line is cropped because of thickness)
        ax.set_ylim(-20, nlines + 20)

        # No ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add title
        if len(title) > 0:
            ax.text(
                0.5,
                0.8,
                title,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                color="w",
                family="sans-serif",
                fontweight="light",
                fontsize=16,
            )

        # Save the figure
        plt.savefig(
            outpath,
            dpi=300,
            bbox_inches="tight",
            facecolor="black",
            edgecolor="none",
        )

        # Show the plot
        plt.show()
