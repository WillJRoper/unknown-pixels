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


"""

import argparse
import os

from unknownpixels.image import UnknownPixels


def render():
    """Render the image to a waveform representation.

    This function is the main entry point for the Unknown Pixels project. It
    takes an input image file, converts it to a waveform representation, and
    saves it to a file.

    Returns:
        None
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
        "--target-lines",
        "-t",
        type=int,
        help="Number of lines to render along the y-axis.",
        default=50,
    )
    parser.add_argument(
        "--contrast",
        "-c",
        type=float,
        help="Contrast of the image.",
        default=10,
    )
    parser.add_argument(
        "--figsize",
        "-f",
        type=float,
        nargs=2,
        help="Size of the figure to create.",
        default=(8, 8),
    )
    parser.add_argument(
        "--title",
        "-T",
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

    # Parse the command-line arguments and unpack some for convenience
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    preview = args.preview
    target_lines = args.target_lines
    contrast = args.contrast
    figsize = args.figsize
    title = args.title

    # Check if the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' does not exist.")

    # Check if the input file is a valid image format
    valid_image_formats = [
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".gif",
        ".tiff",
        ".webp",
    ]
    if not any(
        input_file.lower().endswith(ext) for ext in valid_image_formats
    ):
        raise ValueError(
            f"Input file '{input_file}' is not a valid image format. "
            f"Supported formats are: {', '.join(valid_image_formats)}."
        )

    # Check if the output file is a valid image format
    if output_file and not any(
        output_file.lower().endswith(ext) for ext in valid_image_formats
    ):
        raise ValueError(
            f"Output file '{output_file}' is not a valid image format. "
            f"Supported formats are: {', '.join(valid_image_formats)}."
        )

    # Check if the target lines is a positive integer
    if not isinstance(target_lines, int) or target_lines <= 0:
        raise ValueError(
            f"Target lines '{target_lines}' is not a positive integer."
        )

    # Create an instance of the UnknownPixels class
    up = UnknownPixels(input_file)

    # Show a preview of the input image after some processing if requested
    if preview:
        up.show()

    # Render the image to a waveform representation
    up.plot_unknown_pleasures(
        contrast=contrast,
        target_lines=target_lines,
        figsize=figsize,
        title=title,
        output_file=output_file,
    )
