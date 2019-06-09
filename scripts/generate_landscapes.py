from navsim.generate_landscapes import *

# Call with folder as first argument
# generate_landscapes.py orig_landscape.npy "[diff-time-1] [diff-time-2]" out/

import os, sys
import skimage.io

orig_name = sys.argv[1]

print("Reading original...")
orig = np.load(orig_name)

diffuse_times = list(map(int, sys.argv[2].split()))

os.chdir(sys.argv[3])

# -- Generate Original Image

skimage.io.imsave("original-landscape.png", orig.astype(np.float))

# -- Generate Diffused Ones

# Times 0 through 2000

for i, diffuse_time in enumerate(diffuse_times):
    print("Generating landscape %0.2i/%0.2i (diffuse time: %0.2i)" % (i, len(diffuse_times), diffuse_time))

    new = diffuse(orig, diffuse_time)
    new = image_from_prob_mat(new)

    basename = "landscape-diffuse-%i"

    np.save((basename + ".npy") % diffuse_time, new)
    skimage.io.imsave((basename + ".png") % diffuse_time, new)


print("Done.")
