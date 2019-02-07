import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import StrMethodFormatter

import skimage
import itertools
import math

class StopNavigationException(Exception):
    pass

class OutOfLandscapeBoundsException(StopNavigationException):
    pass

class NavBySceneFamiliarity(object):
    def __init__(self,
                 landscape,
                 sensor_radius,
                 step_size,
                 n_test_angles = 60,
                 pixel_scale_factor = 1):
        self.landscape = landscape
        self.familiar_scenes = None
        self.n_test_angles = n_test_angles
        self.angles = np.linspace(0, 2 * np.pi, self.n_test_angles)
        self.sensor_radius = sensor_radius
        self.sensor_size = sensor_radius * 2
        self._mask = self._make_sensor_mask()
        self.step_size = step_size
        self.angle_familiarity = np.empty(shape = n_test_angles)
        self.pixel_scale_factor = pixel_scale_factor


    def train_from_path(self, points):
        self.familiar_scenes = np.empty(shape = (len(points), self.sensor_size, self.sensor_size), dtype = self.landscape.dtype)

        angles = points[1:] - points[0:-1]
        angles = np.arctan2(angles[:,1], angles[:, 0])

        i = 0
        for pt, ang in zip(points, angles):
            self.familiar_scenes[i] = self.get_sensor_mat(pt, ang)
            i += 1
        self.familiar_scenes[i] = self.get_sensor_mat(points[-1], ang)

        self.scene_familiarity = np.zeros(shape = len(points))
        self.training_path = points

    def _make_sensor_mask(self):
        circ_mask = np.ones(shape = (self.sensor_size, self.sensor_size), dtype = np.bool)
        for i, j in itertools.product(range(self.sensor_size), repeat=2):
            if np.ceil(np.sqrt((i - (self.sensor_size - 1) / 2) **2 + (j - (self.sensor_size - 1) / 2)**2)) <= self.sensor_radius:
                circ_mask[i, j] = 0

        return circ_mask

    def get_sensor_mat(self, position, angle):
        position = np.round(position).astype(np.int)
        r = self.sensor_radius * self.pixel_scale_factor

        if np.any(np.logical_or(position < r, position > np.subtract(self.landscape.shape, r))):
            raise OutOfLandscapeBoundsException()

        sensor_mat = np.copy(self.landscape[position[0] - r:position[0] + r,
                                            position[1] - r:position[1] + r])

        out = skimage.transform.downscale_local_mean(sensor_mat, (self.pixel_scale_factor, self.pixel_scale_factor))
        out = skimage.transform.rotate(out, 180. * angle / np.pi)

        np.rint(out, out = out)

        out[self._mask] = 0.

        del sensor_mat
        return out


    def step_forward(self, position, start_angle):
        temp = np.empty(shape = (self.sensor_size, self.sensor_size))
        temp_fam = np.empty_like(self.scene_familiarity)
        temp_fam[:] = np.nan

        self.angle_familiarity[:] = np.nan
        self.scene_familiarity[:] = np.inf

        for a_idex, angle in enumerate(self.angles):

            smat_rot = self.get_sensor_mat(position, angle)

            temp_fam[:] = np.nan

            assert len(self.familiar_scenes) == len(self.scene_familiarity)
            for f_index in range(len(self.familiar_scenes)):
                np.subtract(smat_rot, self.familiar_scenes[f_index], out = temp)
                np.abs(temp, out = temp)
                temp_fam[f_index] = np.sum(temp)

                if temp_fam[f_index] < self.scene_familiarity[f_index]:
                    self.scene_familiarity[f_index] = temp_fam[f_index]

            del smat_rot

            self.angle_familiarity[a_idex] = np.min(temp_fam)

        angle = self.angles[np.argmin(self.angle_familiarity)]

        new_pos = (position[0] + self.step_size * np.cos(angle),
                   position[1] + self.step_size * np.sin(angle))

        return (new_pos, angle)


    def animate(self, posarg, angarg):

        fig, (main_ax, scene_ax, angle_ax) = plt.subplots(nrows = 3, ncols = 1,
                                                          figsize = (8, 8),
                                                          gridspec_kw={'height_ratios': [7, 1, 1]})

        main_ax.set_title("Navigation")

        scene_ax.set_title("Familiarity")
        scene_ax.set_xlabel("Training Scene")
        angle_ax.set_xlabel("Angle")

        main_ax.matshow(self.landscape, cmap = 'binary')
        main_ax.plot(self.training_path[:,0], self.training_path[:,1], color = "lime", linewidth = 3)

        scene_ln_x = np.arange(len(self.scene_familiarity))
        scene_ln, = scene_ax.plot(scene_ln_x, self.scene_familiarity, color = 'k', animated = True)
        scene_ax.set_ylim((0, self.sensor_size ** 2))
        scene_ax.yaxis.set_major_locator(plt.NullLocator())
        angle_ln_x = 180. * self.angles / np.pi
        angle_ln, = angle_ax.plot(angle_ln_x, self.angle_familiarity, color = 'k', animated = True)
        angle_ax.set_ylim((0, self.sensor_size ** 2))
        angle_ax.xaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}°"))
        angle_ax.yaxis.set_major_locator(plt.NullLocator())

        xpos, ypos = [posarg[0]], [posarg[1]]
        path_ln, = main_ax.plot(xpos, ypos, color = 'magenta', linewidth = 6, animated = True)

        posthing = [posarg, angarg]
        #anim_ref = [1]

        def init():
            return (path_ln, scene_ln, angle_ln)

        def upd(frame):
            # Step forward
            a, b = self.step_forward(*posthing)
            posthing[0] = a; posthing[1] = b
            xpos.append(posthing[0][0]); ypos.append(posthing[0][1])
            path_ln.set_data(xpos, ypos)

            scene_ln.set_data(scene_ln_x, self.scene_familiarity)
            angle_ln.set_data(angle_ln_x, self.angle_familiarity)

            return (path_ln, scene_ln, angle_ln)
            #anim_ref[0].event_source.stop()

        anim = FuncAnimation(fig, upd,
                             frames = 200, interval = 100, #itertools.repeat(1),
                             init_func = init, blit = True, repeat = False)
        #anim_ref[0] = anim

        fig.tight_layout()

        return fig, anim
