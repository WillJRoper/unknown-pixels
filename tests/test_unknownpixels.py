"""A test suite for the unknownpixels module."""

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from unknownpixels.image import UnknownPixels


@pytest.fixture(autouse=True)
def suppress_plot_show(monkeypatch):
    """Suppress plt.show() during tests to avoid displaying plots."""
    monkeypatch.setattr(plt, "show", lambda: None)


@pytest.fixture
def temp_image_path():
    """Creates a temporary grayscale image file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img = Image.new("L", (120, 80), color=128)  # Non-square image
        img.save(tmp.name)
        return tmp.name


class TestUnknownPixels:
    def test_init_and_grayscale_conversion(self, temp_image_path):
        up = UnknownPixels(temp_image_path)
        assert isinstance(up.img, Image.Image)
        assert up.img.mode == "L"

    def test_shape_property_is_square(self, temp_image_path):
        up = UnknownPixels(temp_image_path)
        width, height = up.shape
        assert width == height

    def test_arr_returns_float64_numpy_array(self, temp_image_path):
        up = UnknownPixels(temp_image_path)
        arr = up.arr
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert arr.shape == (up.img.height, up.img.width)

    def test_pad_to_square_idempotence(self, temp_image_path):
        up = UnknownPixels(temp_image_path)
        size_before = up.shape
        padded_img = up._pad_to_square()
        assert padded_img.size == size_before  # Already padded in __init__

    def test_smooth_image_alters_pixel_values(self, temp_image_path):
        up = UnknownPixels(temp_image_path)
        arr_before = np.array(up.img)
        up.smooth_image(rad=5)
        arr_after = np.array(up.img)
        assert not np.array_equal(arr_before, arr_after)

    def test_plot_unknown_pleasures_outputs_file(self, temp_image_path):
        up = UnknownPixels(temp_image_path)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            outpath = tmp.name

        up.plot_unknown_pleasures(
            contrast=10,
            vmax=None,
            vmin=None,
            nlines=50,
            figsize=(5, 5),
            title="Test",
            outpath=outpath,
            log=False,
        )

        assert os.path.isfile(outpath)
        os.remove(outpath)

    def test_plot_unknown_pleasures_no_output_file(self, temp_image_path):
        up = UnknownPixels(temp_image_path)
        # Should not raise even without outpath
        up.plot_unknown_pleasures(
            contrast=5,
            vmax=None,
            vmin=None,
            nlines=40,
            figsize=(3, 3),
            title="No Save",
            outpath=None,
            log=False,
        )

    def test_plot_unknown_pleasures_log_scaling(self, temp_image_path):
        up = UnknownPixels(temp_image_path)
        # Modify image so it contains strictly positive values
        up.img = Image.fromarray(np.full((100, 100), 10, dtype=np.uint8))

        # Log mode: should handle without error and show a plot
        up.plot_unknown_pleasures(
            contrast=10,
            vmax=None,
            vmin=1,
            nlines=40,
            figsize=(4, 4),
            title="Log Test",
            outpath=None,
            log=True,
        )

    def test_plot_title_empty_is_valid(self, temp_image_path):
        up = UnknownPixels(temp_image_path)
        # Should not fail if title is empty
        up.plot_unknown_pleasures(
            contrast=5,
            vmax=None,
            vmin=None,
            nlines=30,
            figsize=(3, 3),
            title="",
            outpath=None,
            log=False,
        )
