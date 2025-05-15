"""A module for converting a video or gif to a waveform representation.

This module uses moviepy to split videos into individual frames and then uses
the UnknownPixels class to convert each frame to a waveform representation.
Once done the frames are stitched back together to create a video or gif.
"""

import os
import sys
from contextlib import contextmanager
from functools import partial

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.video.io.VideoFileClip import VideoFileClip

from unknownpixels.image import image_to_waveform


@contextmanager
def suppress_output():
    """Suppress stdout and stderr output.

    This context manager temporarily redirects stdout and stderr to os.devnull,
    effectively silencing any output during the execution of the code block
    within the context. This is useful for suppressing output from libraries
    like moviepy that may not be relevant to the user.

    This is mainly used to silence the overbaring outputs of moviepy.
    """
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def video_to_waveform(
    input_path,
    output_path,
    smooth_radius=None,
    contrast=1.0,
    vmax=None,
    vmin=None,
    nlines=100,
    aspect=None,
    title=None,
    log=False,
    perspective=False,
    linewidth=1.0,
    fps=None,
):
    """Convert a video or gif to a waveform representation.

    This function takes a video or gif file as input, splits it into
    individual frames, and then converts each frame to a waveform
    representation using the UnknownPixels class. The resulting frames are
    stitched back together to create a video or gif.

    Args:
        input_path (str): Path to the input video or gif file.
        output_path (str): Path to save the output video or gif file.
        smooth_radius (float, optional): Radius of the Gaussian smoothing
            kernel.
        contrast (float, optional): The contrast defining the height of the
            peaks in the waveform.
        vmax (float, optional): Maximum value to use for the image.
        vmin (float, optional): Minimum value to use for the image.
        nlines (int, optional): Number of lines in the waveform representation.
        aspect (float, optional): Aspect ratio of the output image.
        title (str, optional): Title of the output image.
        log (bool, optional): Whether to use a logarithmic scale for
            the waveform.
        perspective (bool, optional): Whether to add a false perspective
            effect to the image.
        linewidth (float, optional): The width of the lines in the plot.
        fps (float, optional): Frames per second for the output video. If None,
            the input video's fps will be used.
    """
    # Load the clip
    with suppress_output():
        clip = VideoFileClip(input_path, audio=False)

    # Extract the fps from the clip
    fps = clip.fps if fps is None else fps

    # Prepare the processing function with all parameters
    func = partial(
        image_to_waveform,
        smooth_radius=smooth_radius,
        contrast=contrast,
        vmax=vmax,
        vmin=vmin,
        nlines=nlines,
        aspect=aspect,
        title=title,
        log=log,
        perspective=perspective,
        linewidth=linewidth,
        preview=False,
        show=False,
        return_array=True,
    )

    # Process each frame manually
    processed_frames = []
    for frame in clip.iter_frames(fps=clip.fps, dtype="uint8", logger="bar"):
        wf_frame = func(frame)
        processed_frames.append(wf_frame)

    # Clean up reader resources
    clip.reader.close()
    clip.close()

    # Create a new clip from processed frames
    new_clip = ImageSequenceClip(processed_frames, fps=fps)

    # Write out based on extension
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".gif":
        with suppress_output():
            new_clip.write_gif(output_path, fps=fps)
    else:
        with suppress_output():
            new_clip.write_videofile(
                output_path,
                codec="libx264",
                fps=fps,
                audio=False,
            )
