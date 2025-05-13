"""A module for manipulating the image and converting it to a waveform.

This module contains the UnknownPixels class, which is used to convert an
image to a waveform representation. The class provides methods for
smoothing the image, padding it to a square shape, and plotting the
waveform representation. It also provides a method for plotting the image
in the style of Unknown Pleasures, a famous album by Joy Division.

This is mostly a helper class wrapping PIL and mattplotlib to simplify the
process of converting an image to a waveform representation.

Example usage:

    from unknownpixels.image import UnknownPixels

    # Create an instance of the UnknownPixels class
    img = UnknownPixels("path/to/image.png")

    # Smooth the image
    img.smooth_image(5)

    # Plot the image in the style of Unknown Pleasures
    img.plot_unknown_pleasures(
        contrast=10,
        vmax=None,
        vmin=None,
        nlines=50,
        figsize=(8, 8),
        title="My Title",
        outpath="output.png",
        log=False,
    )
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from PIL import Image, ImageFilter, ImageOps
from scipy.ndimage import zoom


class UnknownPixels:
    """A class for converting an image to a waveform representation.

    This class provides methods for smoothing the image, padding it to a
    square shape, and plotting the waveform representation. It also provides
    a method for plotting the image in the style of Unknown Pleasures, a
    famous album by Joy Division.

    Note that the image will always be converted to grayscale and padded to a
    square shape without distortion. The result of this can be seen using the
    `show` method.

    Attributes:
        imgpath (str): The path to the image file.
        img (PIL.Image): The image object.
        shape (tuple): The shape of the image.
        arr (np.ndarray): The image as a numpy array.

    """

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
        self.img = self._pad_to_square()

    @property
    def shape(self):
        """Get the shape of the image.

        Returns:
            tuple: The shape of the image.
        """
        return self.img.size

    @property
    def arr(self):
        """Get the image as a numpy array.

        Returns:
            np.ndarray: The image as a normalized numpy array.
        """
        # Convert the image to a numpy array
        arr = np.array(self.img, dtype=np.float64)
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

        return ImageOps.expand(self.img, border=padding, fill=0)

    def show(self):
        """Plot the image as a waveform representation.

        Returns:
            np.ndarray: The waveform representation of the image.
        """
        # Set up the figure
        fig, ax = plt.subplots(figsize=(1, 1), dpi=1080)
        ax.set_axis_off()

        # Plot the image
        ax.imshow(self.arr, cmap="gray", aspect="auto")

        # Show the plot
        plt.show()

        # Close the figure
        plt.close(fig)

    def plot_unknown_pleasures(
        self,
        contrast,
        vmax,
        vmin,
        nlines,
        aspect,
        title,
        outpath,
        log,
        perspective,
        linewidth,
    ):
        """Plot the image in the style of Unknown Pleasures.

        Creates a representation of an image similar in style to Joy
        Division's seminal 1979 album Unknown Pleasures.

        Adapts some code from this matplotlib examples:
        https://matplotlib.org/stable/gallery/animation/unchained.html

        Args:
            contrast (float):
                The contrast to apply to the image. This is the hieght of the
                peaks in the waveform (i.e. the difference in nlines units
                between vmin and vmax). A contrast of 5 will place the maximum
                peak 5 lines above the flat minimum value.
            vmax (float):
                The maximum value to use for the image. If None the maximum
                value in the image is used.
            vmin (float):
                The minimum value to use for the image. If None the minimum
                value in the image is used.
            nlines (int):
                The number of individual lines to use along the y-axis.
            aspect (float):
                The aspect ratio of the image. This is the ratio of the width
                to the height of the image.
            title (str):
                The title to add to the image. If None no title is added.
            outpath (str):
                The path to save the image to. If None no image is saved.
            log (bool):
                Whether to log scale the input image. Default is False.
            perspective (bool):
                Whether to add a false perspective effect to the image.
                Default is False.
            linewidth (float):
                The width of the lines in the plot.
        """
        # Extract data
        data = self.arr

        # define vmin and vmax
        vmax = np.max(data) if vmax is None else vmax
        vmin = np.min(data) if vmin is None else vmin

        # Are we doing log scaling?
        if log:
            # We are, do the log scaling
            data = np.log10(data)

            # Log scale vmin and vmax
            vmin = np.log10(vmin)
            vmax = np.log10(vmax)

            # Set any -np.inf values to zero (once renormalised)
            data[data == -np.inf] = vmin

            # Define normalising function
            norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

        else:
            # No log scaling, just normalise
            norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

        # Normalise data
        data = norm(data) * contrast

        # Resample the data to have the same x resolution but nlines y
        # resolution.
        data = zoom(
            data,
            (nlines / data.shape[0], 1),
            order=3,
        )

        # Flip vertically to keep top of image at top of plot
        data = np.flipud(data)

        # Create new Figure with black background
        fig = plt.figure(figsize=(6, 6 * aspect), facecolor="black")

        # Add a subplot with no frame
        ax = fig.add_subplot(111, frameon=False)

        # Create the x axis
        X = np.linspace(-1, 1, data.shape[-1])

        # Generate line plots
        lines = []
        for i in range(data.shape[0]):
            # Are we doing the false perspective effect?
            if perspective:
                # Small reduction of the X extents to get a cheap perspective
                # effect
                xscale = 1 - i / (nlines * 2.0)

                # Same for linewidth (thicker strokes on bottom)
                lw = linewidth - i / (nlines * 2.0)

            else:
                # No perspective, just use the full x range and a constant
                # linewidth
                xscale = 1
                lw = linewidth

            # Plot a black "face" to the line
            ax.fill_between(
                xscale * X,
                i * np.ones_like(X),
                i + data[i],
                color="k",
                zorder=-i,
            )

            # Plot the line itself
            (line,) = ax.plot(
                xscale * X,
                i + data[i],
                color="w",
                lw=lw,
                zorder=-i,
            )
            lines.append(line)

        # # Set y limit (or first line is cropped because of thickness)
        # ax.set_ylim(-2, nlines + 2)

        # No ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add title
        if len(title) > 0:
            ax.set_title(
                title,
                color="w",
                family="sans-serif",
                fontweight="light",
                fontsize=12,
            )

        # Save the figure
        if outpath is not None:
            fig.savefig(
                outpath,
                format=outpath.split(".")[-1],
                dpi=300,
                bbox_inches="tight",
                facecolor="black",
                edgecolor="none",
            )

        # Show the plot
        plt.show()
