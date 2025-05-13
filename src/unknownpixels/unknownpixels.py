"""The main module for the Unknown Pixels project.

This module defines an entry point for injesting a image and converting it to
a waveform representation (i.e. in the style of Joy Division's "Unknown
Pleasures" album cover) and saving it to a file.

The input file can be any format PIL supports:
    - PNG
    - JPEG
    - BMP
    - GIF
    - TIFF
    - WEBP

Command line arguments:

    -i, --input: Path to the input image file.
    -o, --output: Path to the output file. If not specified, the output will
        be saved to the same directory as the input file with a .png
        extension.
    -n, --nlines: Number of lines to render along the y-axis. Default is 50.
    -f, --figsize: Size of the figure to create. Default is (8, 8).
    -t, --title: Title to add to the image. Default is no title.
    -p, --preview: Show a preview of the input image after some processing.
    -l, --log: Whether to log scale the input image. Default is False.
    -v, --vmax: Maximum value to use for the image. Default is None.
    -V, --vmin: Minimum value to use for the image. Default is None.
    -c, --contrast: The contrast defining the height of the peaks in the
        waveform. A contrast of 5 will place the maximum peak 5 lines above
        the flat minimum value. Default is 10.
    -r, --smooth: Radius of the Gaussian smoothing kernel. Default is None.
    -h, --help: Show this help message and exit.

Example usage:

    unknown-pixels -i input.png -o output.png -n 50 -f 10 10 -T "My Title"

"""

import argparse
import os

import matplotlib.pyplot as plt
import PIL.Image as Image

from unknownpixels._version import __version__
from unknownpixels.image import UnknownPixels


def render():
    """Render the image to a waveform representation.

    This function is the main entry point for Unknown Pixels. It
    takes an input image file, converts it to a waveform representation, and
    shows the resulting plot. (With optional smoothing, log scaling, and
    saving to a file.)
    """
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Render an image to a waveform representation."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Path to the input image file.",
        required=True,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to the output file.",
        default=None,
    )
    parser.add_argument(
        "--nlines",
        "-n",
        type=int,
        help="Number of lines to render along the y-axis.",
        default=50,
    )
    parser.add_argument(
        "--aspect",
        "-a",
        type=float,
        help="The aspect ratio of the image (1.0 is square, "
        "< 1.0 is wide, > 1.0 is tall).",
        default=1.0,
    )
    parser.add_argument(
        "--title",
        "-t",
        type=str,
        help="Title to add to the image, default is no title.",
        default="",
    )
    parser.add_argument(
        "--preview",
        "-p",
        action="store_true",
        help="Show a preview of the input image after some processing.",
    )
    parser.add_argument(
        "--log",
        "-L",
        action="store_true",
        help="Whether to log scale the input image.",
    )
    parser.add_argument(
        "--vmax",
        "-v",
        type=float,
        help="Maximum value to use for the image.",
        default=None,
    )
    parser.add_argument(
        "--vmin",
        "-V",
        type=float,
        help="Minimum value to use for the image.",
        default=None,
    )
    parser.add_argument(
        "--contrast",
        "-c",
        type=float,
        help=(
            "The contrast defining the height of the peaks in the waveform. "
            "A contrast of 5 will place the maximum peak 5 lines above the "
            "flat minimum value."
        ),
        default=10,
    )
    parser.add_argument(
        "--smooth",
        "-r",
        type=float,
        help="Radius of the Gaussian smoothing kernel.",
        default=None,
    )
    parser.add_argument(
        "--perspective",
        "-P",
        action="store_true",
        help="Whether to add a false perspective effect to the image.",
    )
    parser.add_argument(
        "--linewidth",
        "-l",
        type=float,
        help="The width of the lines in the plot.",
        default=1.5,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show the version number and exit.",
    )

    # Parse the command-line arguments and unpack some for convenience
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    nlines = args.nlines
    preview = args.preview
    aspect = args.aspect
    title = args.title
    log = args.log
    vmax = args.vmax
    vmin = args.vmin
    contrast = args.contrast
    smooth_radius = args.smooth
    perspective = args.perspective
    linewidth = args.linewidth

    # Check if the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' does not exist.")

    # Get the valid input image formats from PIL
    input_extensions = Image.registered_extensions()

    # Check if the input file is a valid image format
    valid_input_image_formats = list(
        {ext for ext, fmt in input_extensions.items() if fmt in Image.OPEN}
    )
    if not any(
        input_file.lower().endswith(ext) for ext in valid_input_image_formats
    ):
        raise ValueError(
            f"Input file '{input_file}' is not a valid image format. "
            f"Supported formats are: {', '.join(valid_input_image_formats)}."
        )

    # Check if the output file is a valid image format
    valid_output_image_formats = list(
        plt.gcf().canvas.get_supported_filetypes().keys()
    )
    if output_file and not any(
        output_file.lower().endswith(ext) for ext in valid_output_image_formats
    ):
        raise ValueError(
            f"Output file '{output_file}' is not a valid image format. "
            f"Supported formats are: {', '.join(valid_output_image_formats)}."
        )

    # Close the figure we opened to get the supported filetypes
    plt.close()

    # Check if the target lines is a positive integer
    if not isinstance(nlines, int) or nlines <= 0:
        raise ValueError(f"Target lines '{nlines}' is not a positive integer.")

    # Create an instance of the UnknownPixels class
    up = UnknownPixels(input_file)

    # Show a preview of the input image after some processing if requested
    if preview:
        up.show()

    # If the smooth radius is set smooth the image
    if smooth_radius is not None:
        up.smooth_image(smooth_radius)

    # Render the image to a waveform representation
    up.plot_unknown_pleasures(
        contrast=contrast,
        vmax=vmax,
        vmin=vmin,
        nlines=nlines,
        aspect=aspect,
        title=title,
        outpath=output_file,
        log=log,
        perspective=perspective,
        linewidth=linewidth,
    )
