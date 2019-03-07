from GenerateLandscape import *

# Call with folder as first argument
# generate_landscapes.py size out/

import os, sys
import skimage.io

os.chdir(sys.argv[2])

size = int(sys.argv[1])
assert size % 10 == 0

# -- Generate Original Image

print("Generating original...")

orig = checkerboard(size, size // 10)

skimage.io.imsave("original-landscape.png", orig)

# -- Generate Diffused Ones

# Times 0 through 2000
diffuse_times = np.arange(0, 10) * 10
for i, diffuse_time in enumerate(diffuse_times):
    print("Generating landscape %0.2i/%0.2i (diffuse time: %0.2i)" % (i, len(diffuse_times), diffuse_time))

    new = diffuse(orig, diffuse_time)
    new = image_from_prob_mat(new)

    basename = "landscape-diffuse-%i"

    np.save((basename + ".npy") % diffuse_time, new)
    skimage.io.imsave((basename + ".png") % diffuse_time, new)


print("Done.")
