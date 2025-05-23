# unknown-pixels

**unknown-pixels** is a simple Python command-line tool that transforms images or videos into waveform art reminiscent of Joy Division's _Unknown Pleasures_ album cover.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d88d2fb4-88d6-4678-9e30-46ef993c6b32" alt="Input" width="60%" />
  <img src="https://github.com/user-attachments/assets/7f89f1f4-e9e5-4d50-94f8-6fef5470b787" alt="UP" width="60%" />
</p>


Unknown-pixels first converts the input image to grayscale, then (if necessary) it will pad the image along the smallest axis to make the image square. It then slices the image into `nlines` horizontal slices and renders each slice as a stylised "waveform", creating a unique visual representation of the original image.

## Installation

Ensure you have Python 3.8 or higher installed. Then, install `unknown-pixels` using pip:

```bash
pip install unknown-pixels
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

Or for a video:

```bash
unknown-pixels --input path/to/video.gif
```

This will process the input image and automatically show the waveform representation of the image.

### Options

- `-i`, `--input`: Path to the input image/video file. This image can be in any PIL-compatible format or any ffmpeg compatible video format.
- `-o`, `--output`: [Optional] Path to the output file.
- `-n`, `--nlines`: [Optional] Number of lines to render along the y-axis. Default is 50.
- `-a`, `--aspect`: [Optional] The aspect ratio of the final image, 1.0 is square, < 1.0 is wide and > 1.0 is tall. Default uses the input image's aspect ratio.
- `-t`, `--title`: [Optional] Title to add to the image. Default is no title.
- `-p`, `--preview`: [Optional] Show a preview of the input image after some processing.
- `-L`, `--log`: [Optional] Whether to log scale the input image. Default is False.
- `-v`, `--vmax`: [Optional] Maximum value to use for the image. Default is None.
- `-V`, `--vmin`: [Optional] Minimum value to use for the image. Default is None.
- `-c`, `--contrast`: [Optional] The contrast defining the height of the peaks in the waveform. A contrast of 5 will place the maximum peak 5 lines above the flat minimum value. Default is 10.
- `-r`, `--smooth`: [Optional] Radius of the Gaussian smoothing kernel. Default is None.
- `-P`, `--perspective`: [Optional] Add a false perspective effect. Default False.
- `-l`, `--linewidth`: [Optional] The width of the lines. Default is 1.0.
- `-f`, `--fps`: [Optional] The target frames per second for an output video. Only applicable for a video input. If omitted, the input video FPS will be used.
- `--version`: Print the current version.
- `--help`: Show a help message and exit.

Example:

```bash
unknown-pixels -i path/to/image.jpg -n 50 -t "Joy Division" -c 10
```

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Gallery 

### Stephan's Quintet 

![stephans_quintet_up](https://github.com/user-attachments/assets/8551b1d5-5c25-471c-9c45-6e5a1df3fa32)

### M87 

<p align="center">
  <img src="https://github.com/user-attachments/assets/372dba0d-8157-48ec-bf53-cd0508bd1402" alt="Input" width="100%" />
</p>


Courtesy of Robyn Munoz.

### Wolf Rayet Star

![wolf_rayet_up](https://github.com/user-attachments/assets/8ef112c7-491b-445c-9e9e-009edd6a2f05)

### Pillars of Creation

![pillars_up](https://github.com/user-attachments/assets/2bab40e4-156f-4409-82bd-44471b4b139d)

### Neptune and Triton 

![neptune_up](https://github.com/user-attachments/assets/62ea1c02-f6c3-42f2-8272-7baf3b3a3850)

### Dark matter Halo from FLAMINGO

![flamingo_cluster_uo](https://github.com/user-attachments/assets/e513fcd2-09b6-49e3-bde6-c853c40ad5df)

### Millennium Cosmic Web

![cosmic_web_up](https://github.com/user-attachments/assets/23793ee8-e59b-40b6-8b0e-bcccf94fea94)

### Homer Hedge Meme

![homer_UP](https://github.com/user-attachments/assets/b3c72ad4-c4f1-4c19-aadf-8eba70e4df41)

### Blinking meme 

![blinking_up](https://github.com/user-attachments/assets/4f561884-4adb-44d3-9436-e370a10412af)






