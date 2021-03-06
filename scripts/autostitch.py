"""Randomly tile together input images.

Automatically crops input images by thresholding them in greyscale and taking
the smallest square size that can be inscribed in the circle given by the
cropped image. (This automatically deals with the circular mask from the
microscope.)

``<outdim>'' gives, in the format 'NxM', the dimension of the output image
in tiles.

Usage:
  autostitch.py <dir> <outdim> [--crop-threshold=<thresh>] [--downscale=<factor>] [--crossfade=<n>] [--px-per-mm=<n>] [--crop-to-size=<dims>] [--n-rotations=<nrots>] [--generate-n=<n>]

Options:
  -h --help                 Show this screen
  --crop-threshold=<thresh> Threshold for cropping, percentage.  [default: 0.1]
  --n-rotations=<nrots>     Number of rotations of each tile to use. [default: 1]
  --downscale=<factor>      Downscale cropped tiles by <factor> before use [default: 1].
  --crossfade=<n>           Crossfade n pixels at each boundary
  --px-per-mm=<n>           Compute and print real-world sizes with given conversion factor [default: 1].
  --crop-to-size=<dims>     Crop the final image to this size in pixels, format NxM
  --generate-n=<n>          Generate n different final images. [default: 1]
"""

import numpy as np
RNG = np.random.default_rng()

#import skimage as ski
#from skimage.transform import downscale_local_mean
from PIL import Image, ImageOps

import glob
import itertools

from tqdm import tqdm

from docopt import docopt

args = docopt(__doc__, version = "stitch 0.1")

globbed = glob.glob(args['<dir>'] + "/*.JPG")

crop_thresh = float(args['--crop-threshold'])
tile_margin = int(args['--crossfade']) #15
scale_factor = int(args['--downscale']) #20
px_per_mm = int(args['--px-per-mm']) #450
crop_to_size = args['--crop-to-size']
n_rotations = int(args['--n-rotations'])
final_size = [int(e) for e in args['<outdim>'].split('x')]
generate_n = int(args['--generate-n'])

if crop_to_size is not None:
    crop_to_size = tuple(int(e) for e in crop_to_size.split('x'))

if np.product(final_size) > len(globbed)*n_rotations:
    raise ValueError("Not enough tiles (# tiles %i < total needed tiles %i). Add more tiles and/or increase --n-rotations" % ((len(globbed)*n_rotations), np.product(final_size)))

tiles = []

rotations = np.linspace(0, 360., n_rotations)

threshold_lut = np.empty(shape = 256)
threshold_lut[:] = 255
threshold_lut[:int(crop_thresh * len(threshold_lut))] = 0

# Load tiles
for f in tqdm(globbed, desc = "Loading, crop+scaling"):
    image = Image.open(f).convert('L')
    image_thresh = image.point(threshold_lut)
    image = image.crop(image_thresh.getbbox())
    for rotation in tqdm(rotations, desc = "rotation"):
        im2 = image.rotate(rotation)
        # Crop square
        diameter = min(im2.size)
        square_side_len = np.sqrt(0.5 * diameter * diameter)
        inset = int(1.2 * 0.5*(diameter - square_side_len))
        im2 = im2.crop((inset, inset, diameter - inset, diameter - inset))
        assert im2.size[0] == im2.size[1]
        # Resize
        im2 = im2.resize((int(im2.size[0] / scale_factor),) * 2)
        tiles.append(np.asarray(im2))
        del im2

# Crop to smallest
tile_shape = min(t.shape[0] for t in tiles)
tiles = [t[:tile_shape, :tile_shape] / 255. for t in tiles]
tile_shape = (tile_shape,)*2
px_per_mm /= scale_factor
print("Tile shape: %ix%ipx @ %.2fpx/mm" % (tile_shape[0], tile_shape[1], px_per_mm))

# Half margins:
gradient = np.linspace(0, 1, tile_margin)

for i in range(len(tiles)):
    tiles[i][:tile_margin, :] *= gradient[:, None]
    tiles[i][:, :tile_margin] *= gradient
    tiles[i][-tile_margin:, :] *= gradient[::-1][:, None]
    tiles[i][:, -tile_margin:] *= gradient[::-1]


out_tile_shape = tuple(s - tile_margin for s in tile_shape)

out = np.empty(
    shape = tuple(out_tile_shape[i] * final_size[i] + 2*tile_margin for i in range(2)),
    dtype = tiles[0].dtype
)

for image_i in tqdm(range(generate_n)):
    out.fill(0)

    # Choose randomly without replacement.
    tile_choices = RNG.choice(len(tiles), size = tuple(final_size), replace = False)

    for i, j in itertools.product(*[range(i) for i in final_size]):
        out[i * out_tile_shape[0]:(i + 1) * out_tile_shape[0] + tile_margin,
            j * out_tile_shape[1]:(j + 1) * out_tile_shape[1] + tile_margin] += tiles[tile_choices[i, j]]

    out_crop = out[tile_margin:-2*tile_margin, tile_margin:-2*tile_margin]

    print("Final size: %ix%ipx ~= %.2fx%.2fmm" % (out_crop.shape[0], out_crop.shape[1], out_crop.shape[0] / px_per_mm, out_crop.shape[1] / px_per_mm))

    assert np.max(out_crop) <= 1.
    assert np.min(out_crop) >= 0.

    print("Constrast and equalization...")
    out_im = Image.fromarray(out_crop * 255.).convert('L')
    out_im = ImageOps.autocontrast(out_im)
    out_im = ImageOps.equalize(out_im)

    print("Cropping...")
    if crop_to_size is not None:
        out_im = out_im.crop(box = (0, 0, crop_to_size[0], crop_to_size[1]))

    print("Saving...")

    out_im.save("landscape-%i.png" % image_i, format = 'png')
