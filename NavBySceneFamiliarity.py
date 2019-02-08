import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import StrMethodFormatter
import matplotlib.patches
import matplotlib.gridspec

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
                 #sensor_radius,
                 sensor_dimensions,
                 step_size,
                 n_test_angles = 60,
                 sensor_pixel_dimensions = [1, 1],
                 n_sensor_levels = 5):
        self.landscape = landscape
        self.familiar_scenes = None
        self.n_test_angles = n_test_angles
        self.angles = np.linspace(0, 2 * np.pi, self.n_test_angles)
        self.sensor_dimensions = np.asarray(sensor_dimensions)
        assert np.all(self.sensor_dimensions % 2 == 0)
        self.n_sensor_pixels = np.product(self.sensor_dimensions)
        assert n_sensor_levels >= 2
        self.n_sensor_levels = n_sensor_levels - 1
        self.step_size = step_size
        self.angle_familiarity = np.empty(shape = n_test_angles)
        self.sensor_pixel_dimensions = np.asarray(sensor_pixel_dimensions)


    def train_from_path(self, points):
        self.familiar_scenes = np.empty(shape = (len(points), self.sensor_dimensions[0], self.sensor_dimensions[1]), dtype = self.landscape.dtype)

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
        #r = self.sensor_radius * self.pixel_scale_factor
        dists = self.sensor_dimensions * self.sensor_pixel_dimensions // 2
        r = np.max(dists)

        if np.any(np.logical_or(position < r, position > np.subtract(self.landscape.shape, r))):
            raise OutOfLandscapeBoundsException()

        sensor_mat = np.copy(self.landscape[position[0] - r:position[0] + r,
                                            position[1] - r:position[1] + r])

        out = skimage.transform.rotate(sensor_mat, 180. * angle / np.pi)
        out = skimage.transform.downscale_local_mean(out, tuple(self.sensor_pixel_dimensions))

        r0 = out.shape[0] // 2
        r1 = out.shape[1] // 2
        out = out[r0 - self.sensor_dimensions[0] // 2 : r0 + self.sensor_dimensions[0] // 2, r1 - self.sensor_dimensions[1] // 2 : r1 + self.sensor_dimensions[1] // 2]

        #np.rint(out, out = out)
        out *= self.n_sensor_levels
        np.rint(out, out = out)
        out /= self.n_sensor_levels

        #out[self._mask] = 0.

        del sensor_mat
        return out

        # position = np.round(position).astype(np.int)
        # r = self.sensor_radius * self.pixel_scale_factor
        #
        # if np.any(np.logical_or(position < r, position > np.subtract(self.landscape.shape, r))):
        #     raise OutOfLandscapeBoundsException()
        #
        # sensor_mat = np.copy(self.landscape[position[0] - r:position[0] + r,
        #                                     position[1] - r:position[1] + r])
        #
        # out = skimage.transform.downscale_local_mean(sensor_mat, (self.pixel_scale_factor, self.pixel_scale_factor))
        # out = skimage.transform.rotate(out, 180. * angle / np.pi)
        #
        # np.rint(out, out = out)
        #
        # #out[self._mask] = 0.
        # out = out[self.sensor_radius-1]
        #
        # del sensor_mat
        # return out


    def step_forward(self, position, start_angle):
        temp = np.empty(shape = self.sensor_dimensions)
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

        navcolor = 'darkorchid'
        traincolor = 'dodgerblue'

        fig = plt.figure(figsize = (10, 8))
        gs = matplotlib.gridspec.GridSpec(3, 2,
                                          width_ratios = [2, 1])

        main_ax = plt.subplot(gs[:, 0])
        sensor_ax = plt.subplot(gs[0, 1])
        scene_ax = plt.subplot(gs[1, 1])
        angle_ax = plt.subplot(gs[2, 1])



        #fig, (main_ax, scene_ax, angle_ax) = plt.subplots(nrows = 3, ncols = 2,
        #                                                  figsize = (8, 8),
        #                                                  gridspec_kw={'height_ratios': [7, 1, 1]})

        main_ax.set_title("Navigation")

        scene_ax.set_title("Familiarity")
        scene_ax.set_xlabel("Training Scene")
        angle_ax.set_xlabel("Angle")

        sensor_ax.set_title("Sensor Matrix")
        init_sens_mat = np.zeros(shape = self.sensor_dimensions)
        init_sens_mat[0] = 1.
        sensor_im = sensor_ax.matshow(init_sens_mat, cmap = 'binary', vmax = 1.4, animated = True)
        sensor_ax.xaxis.set_major_locator(plt.NullLocator())
        sensor_ax.yaxis.set_major_locator(plt.NullLocator())

        main_ax.matshow(self.landscape, cmap = 'binary', vmax = 1.4)
        main_ax.plot(self.training_path[:,0], self.training_path[:,1], color = traincolor, linewidth = 3)

        scene_ln_x = np.arange(len(self.scene_familiarity))
        scene_min = scene_ax.axvline(0, color = navcolor, animated = True, zorder = 0, alpha = 0.7, linewidth = 1)
        scene_ln, = scene_ax.plot(scene_ln_x, self.scene_familiarity, color = 'k', animated = True)
        scene_ax.set_ylim((0, self.n_sensor_pixels))
        scene_ax.yaxis.set_major_locator(plt.NullLocator())

        angle_min = angle_ax.axvline(0, color = navcolor, animated = True, zorder = 0, alpha = 0.7, linewidth = 1)
        angle_ln_x = 180. * self.angles / np.pi
        angle_ln, = angle_ax.plot(angle_ln_x, self.angle_familiarity, color = 'k', animated = True)
        angle_ax.set_ylim((0, self.n_sensor_pixels))
        angle_ax.xaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}°"))
        angle_ax.yaxis.set_major_locator(plt.NullLocator())

        xpos, ypos = [posarg[0]], [posarg[1]]
        path_ln, = main_ax.plot(xpos, ypos, color = navcolor, marker = 'o', markersize = 5, linewidth = 2, animated = True)


        sens_rect_dims = (self.sensor_dimensions * self.sensor_pixel_dimensions)
        sensor_rect = matplotlib.patches.Rectangle(xy = (0., 0.),
                                                   width = sens_rect_dims[0],
                                                   height = sens_rect_dims[1],
                                                   angle = 0.,
                                                   alpha = 0.35,
                                                   animated = True,
                                                   color = navcolor)

        sens_rect_dims = (sens_rect_dims // 2)

        main_ax.add_patch(sensor_rect)

        posthing = [posarg, angarg]
        #anim_ref = [1]

        def init():
            return (path_ln, scene_ln, angle_ln, sensor_rect, sensor_im)

        def upd(frame):
            # Step forward
            pos, ang = self.step_forward(*posthing)
            posthing[0] = pos; posthing[1] = ang
            xpos.append(posthing[0][0]); ypos.append(posthing[0][1])
            path_ln.set_data(xpos, ypos)

            sensor_rect.set_xy((pos[0] - sens_rect_dims[0] * np.cos(ang) + sens_rect_dims[1] * np.sin(ang),
                                pos[1] - sens_rect_dims[0] * np.cos(ang) - sens_rect_dims[1] * np.cos(ang)))
            sensor_rect.angle = np.rad2deg(ang)
            #sensor_rect.set_xy( + sens_rect_dims * [np.cos(ang), np.sin(ang)]))

            scene_ln.set_ydata(self.scene_familiarity)
            s_amin = np.argmin(self.scene_familiarity)
            scene_min.set_xdata([s_amin, s_amin])

            angle_ln.set_ydata(self.angle_familiarity)
            a_amin = 360. * np.argmin(self.angle_familiarity) / len(self.angle_familiarity)
            angle_min.set_xdata([a_amin, a_amin])

            sensor_im.set_array(self.get_sensor_mat(pos, ang))

            return (path_ln, scene_ln, scene_min, angle_ln, angle_min, sensor_rect, sensor_im)
            #anim_ref[0].event_source.stop()

        anim = FuncAnimation(fig, upd,
                             frames = 200, interval = 100, #itertools.repeat(1),
                             blit = True, repeat = False)
        #anim_ref[0] = anim

        fig.tight_layout()

        return fig, anim
