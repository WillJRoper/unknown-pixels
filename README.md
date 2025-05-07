# unknown-pixels

<p align="center">
  <img src="https://github.com/user-attachments/assets/a57ab1dc-eb40-4716-8c13-ed03db5d965c" alt="Input" width="30%" />
  <img src="https://github.com/user-attachments/assets/0fb9b396-2e40-4e47-8647-7e88c63b4dd5" alt="UP" width="30%" />
  <img src="https://github.com/user-attachments/assets/2c223958-316d-4985-bbf3-abf1fa2929b7" alt="UP_Perspective" width="30%" />
</p>


**unknown-pixels** is a simple Python command-line tool that transforms images into waveform art reminiscent of Joy Division's _Unknown Pleasures_ album cover.

Unknown-pixels first converts the input image to grayscale, then (if necessary) it will pad the image along the smallest axis to make the image square. It then slices the image into `nlines` horizontal slices and renders each slice as a stylised "waveform", creating a unique visual representation of the original image.

## Installation

Ensure you have Python 3.8 or higher installed. Then, install `unknownpixels` using pip:

```bash
pip install unknownpixels
```

Or clone the repo and run

```bash
pip install .
```

in the root of the repository to get the latest "development" version.

## Usage

After installation, use the `unknown-pixels` command:

```bash
unknown-pixels --input path/to/image.jpg
```

This will process the input image and automatically show the waveform representation of the image.

### Options

- `-i`, `--input`: Path to the input image file. This image can be in any PIL-compatible format.
- `-o`, `--output`: [Optional] Path to the output file. If not specified, the output will be saved to the same directory as the input file with a .png extension.
- `-n`, `--nlines`: [Optional] Number of lines to render along the y-axis. Default is 50.
- `-f`, `--figsize`: [Optional] Size of the figure to create. Default is (8, 8).
- `-t`, `--title`: [Optional] Title to add to the image. Default is no title.
- `-p`, `--preview`: [Optional] Show a preview of the input image after some processing.
- `-l`, `--log`: [Optional] Whether to log scale the input image. Default is False.
- `-v`, `--vmax`: [Optional] Maximum value to use for the image. Default is None.
- `-V`, `--vmin`: [Optional] Minimum value to use for the image. Default is None.
- `-c`, `--contrast`: [Optional] The contrast defining the height of the peaks in the waveform. A contrast of 5 will place the maximum peak 5 lines above the flat minimum value. Default is 10.
- `-r`, `--smooth`: [Optional] Radius of the Gaussian smoothing kernel. Default is None.
- `-P`, `--perspective`: [Optional] Add a false perspective effect. Default False.
- `--version`: Print the current version.
- `--help`: Show a help message and exit.

Example:

```bash
unknown-pixels -i path/to/image.jpg -n 50 -t "Joy Division" -c 10
```

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
