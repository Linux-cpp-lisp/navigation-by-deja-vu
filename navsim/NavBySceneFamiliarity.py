import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import StrMethodFormatter
import matplotlib.patches
import matplotlib.gridspec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1 import inset_locator
import matplotlib.font_manager as fm
import matplotlib.colors

import skimage
import itertools
import math
import warnings

from navsim.util import sads

class StopNavigationException(Exception):
    def get_reason(self):
        raise NotImplementedError()
    def get_code(self):
        raise NotImplementedError()

class ReachedEndOfTrainingPathException(StopNavigationException):
    def get_reason(self):
        return "agent reached end of training path"
    def get_code(self):
        return 1

class NavigatingFailedException(StopNavigationException):
    pass

class TooFarFromTrainingPathException(NavigatingFailedException):
    def get_reason(self):
        return "agent went too far from training path"
    def get_code(self):
        return -1

class OutOfLandscapeBoundsException(NavigatingFailedException):
    def get_reason(self):
        return "agent went too close to boundary of landscape"
    def get_code(self):
        return -2

#SCALING_VMAX = 1.4
LANDSCAPE_CMAP = 'Greens'
navcolor = 'darkorchid'
traincolor = 'dodgerblue'

class NavBySceneFamiliarity(object):
    """
    Args:
        sensor_real_area (tuple: (float, string)): First number is an area,
            second is the unit to display while plotting (i.e. a length unit).
    """

    def __init__(self,
                 landscape,
                 sensor_dimensions,
                 step_size,
                 n_test_angles = 60,
                 sensor_pixel_dimensions = [1, 1],
                 max_distance_to_training_path = np.inf,
                 n_sensor_levels = 5,
                 threshold_factor = 2.,
                 saccade_degrees = 180.,
                 sensor_real_area = None):
        self.landscape = landscape

        self.position = (0., 0.)
        self.angle = 0.

        self.familiar_scenes = None
        self.n_test_angles = n_test_angles
        self.threshold_factor = threshold_factor

        self.sensor_real_area = sensor_real_area

        self.saccade_degrees = saccade_degrees
        sd2 = saccade_degrees / 2
        self.angle_offsets = np.linspace(-(np.pi * sd2/180.), np.pi * sd2/180., self.n_test_angles)

        self.sensor_dimensions = np.asarray(sensor_dimensions)
        self.sensor_pixel_dimensions = np.asarray(sensor_pixel_dimensions)
        assert np.all((self.sensor_dimensions * self.sensor_pixel_dimensions) % 2 == 0)

        self.n_sensor_pixels = np.product(self.sensor_dimensions)
        assert n_sensor_levels >= 2
        self.n_sensor_levels = n_sensor_levels
        self.step_size = step_size
        self.angle_familiarity = np.empty(shape = n_test_angles)
        self.step_familiarity = np.inf
        self.max_distance_to_training_path = max_distance_to_training_path

        self.training_path = None

        self.reset_error()


    def train_from_path(self, points):
        if self.training_path is not None:
            raise ValueError("Tried to train NavBySceneFamiliarity more than once.")

        self.familiar_scenes = np.empty(shape = (len(points), self.sensor_dimensions[1], self.sensor_dimensions[0]), dtype = self.landscape.dtype)

        angles = points[1:] - points[0:-1]
        angles = np.arctan2(angles[:,1], angles[:, 0])

        i = 0
        for pt, ang in zip(points, angles):
            self.familiar_scenes[i] = self.get_sensor_mat(pt, ang)
            i += 1
        self.familiar_scenes[i] = self.get_sensor_mat(points[-1], ang)

        self.scene_familiarity = np.zeros(shape = len(points))
        self.training_path = points

        self.reset_error()


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

        if np.any(np.logical_or(position < r, position[::-1] > np.subtract(self.landscape.shape, r))):
            raise OutOfLandscapeBoundsException()

        sensor_mat = np.copy(self.landscape[position[1] - r:position[1] + r,
                                            position[0] - r:position[0] + r])

        out = skimage.transform.rotate(sensor_mat, 270. + 180. * angle / np.pi)
        out = skimage.transform.downscale_local_mean(out, (self.sensor_pixel_dimensions[1], self.sensor_pixel_dimensions[0]))

        r0 = out.shape[0] // 2
        r1 = out.shape[1] // 2
        out = out[r0 - int(np.floor(self.sensor_dimensions[1] / 2)) : r0 + int(np.ceil(self.sensor_dimensions[1] / 2)),
                  r1 - int(np.floor(self.sensor_dimensions[0] / 2)) : r1 + int(np.ceil(self.sensor_dimensions[0] / 2))]

        out *= (self.n_sensor_levels - 1)
        np.rint(out, out = out)
        out /= (self.n_sensor_levels - 1)

        #out[self._mask] = 0.

        del sensor_mat
        return out


    def reset_error(self):
        self.stopped_with_exception = None

        self.navigated_for_frames = 0
        self._navigation_error = 0.0
        self._n_navigation_error = 0

        if self.training_path is not None:
            # Error temps
            self._error_tempdiff1 = np.empty(self.training_path.shape)
            self._error_tempdiff2 = np.empty(len(self.training_path))
            self._error_tempdiff3 = np.empty(len(self.training_path), dtype = np.bool)
            self._coverage_array = np.zeros(len(self.training_path))

    @property
    def navigation_error(self):
        return np.sqrt(self._navigation_error / self._n_navigation_error)

    @property
    def percent_recapitulated(self):
        return np.sum(self._coverage_array) / len(self._coverage_array)

    def update_error(self):
        self.navigated_for_frames += 1

        np.subtract(self.training_path, self.position, out = self._error_tempdiff1)
        self._error_tempdiff1 *= self._error_tempdiff1
        np.sum(self._error_tempdiff1, axis = 1, out = self._error_tempdiff2)
        np.sqrt(self._error_tempdiff2, out = self._error_tempdiff2)
        #np.linalg.norm(self._error_tempdiff1, axis = 1, out = self._error_tempdiff2)

        diff = np.min(self._error_tempdiff2)

        if diff > self.max_distance_to_training_path:
            raise TooFarFromTrainingPathException()

        # - RMSD ERROR
        self._navigation_error += diff * diff
        self._n_navigation_error += 1

        # - % COVERAGE ERROR
        cvge_thresh = self.threshold_factor * self.step_size
        if diff <= cvge_thresh:
            # There will be at least one:
            np.less_equal(self._error_tempdiff2, cvge_thresh, out = self._error_tempdiff3)
            # Mark all within step size as covered
            self._coverage_array[self._error_tempdiff3] = True


    def step_forward(self, fake = False):
        position = self.position

        #temp = np.empty(shape = (self.sensor_dimensions[1], self.sensor_dimensions[0]))
        temp_fam = np.empty_like(self.scene_familiarity)
        temp_fam[:] = np.nan

        self.angle_familiarity[:] = np.nan
        self.scene_familiarity[:] = np.inf

        for a_idex, angle_offset in enumerate(self.angle_offsets):

            angle = (self.angle + angle_offset) % (2*np.pi)

            smat_rot = self.get_sensor_mat(position, angle)

            temp_fam[:] = np.nan

            assert len(self.familiar_scenes) == len(self.scene_familiarity)
            for f_index in range(len(self.familiar_scenes)):
                # np.subtract(smat_rot, self.familiar_scenes[f_index], out = temp)
                # np.abs(temp, out = temp)
                # temp_fam[f_index] = np.sum(temp)
                temp_fam[f_index] = sads(smat_rot, self.familiar_scenes[f_index])

                if temp_fam[f_index] < self.scene_familiarity[f_index]:
                    self.scene_familiarity[f_index] = temp_fam[f_index]

            del smat_rot

            self.angle_familiarity[a_idex] = np.min(temp_fam)

        best_idex = np.argmin(self.angle_familiarity)
        self.step_familiarity = self.angle_familiarity[best_idex]
        angle = (self.angle + self.angle_offsets[best_idex]) % (2 * np.pi)

        new_pos = (position[0] + self.step_size * np.cos(angle),
                   position[1] + self.step_size * np.sin(angle))

        self.position = new_pos
        self.angle = angle

        if not fake:
            self.update_error()

            if np.linalg.norm(self.training_path[-1] - self.position) <= self.threshold_factor * self.step_size:
                raise ReachedEndOfTrainingPathException()

    @property
    def landscape_real_pixel_size(self):
        s = np.sqrt(self.sensor_real_area[0] / np.product(self.sensor_dimensions * self.sensor_pixel_dimensions))
        return s

    # ----- Plotting Code
    def quiver_plot(self, **kwargs):
        data = self.quiver_plot_data(**kwargs)
        return self.plot_quiver(**data)


    def quiver_plot_data(self, n_box = 10, n_test_angles = None, max_distance = np.inf):
        if n_test_angles is None:
            n_test_angles = self.n_test_angles

        max_dist = max_distance
        # Remember
        old_pos = self.position
        old_ang = self.angle
        old_ang_offsets = self.angle_offsets
        old_ang_familiarity = self.angle_familiarity

        # Compute
        dists = self.sensor_dimensions * self.sensor_pixel_dimensions
        r = np.max(dists) / 2
        land_size = self.landscape.shape - 2 * r

        box_size = land_size / n_box

        nX, nY = n_box, n_box

        familiarities = np.empty(shape = (nY, nX))
        U = np.empty(shape = (nY, nX))
        V = np.empty(shape = (nY, nX))

        # Set 360 angle offsets
        self.angle_offsets = np.linspace(-np.pi, np.pi, n_test_angles)
        self.angle_familiarity = np.empty(len(self.angle_offsets))

        for j in range(nY):
            for i in range(nX):
                self.position = np.array([r + box_size[1] * 0.5 + box_size[1] * i,
                                          r + box_size[0] * 0.5 + box_size[0] * j])
                self.angle = 0

                path_dist = np.min(np.linalg.norm(self.training_path - self.position, axis = 1))
                if path_dist <= max_dist:
                    self.step_forward(fake = True)
                    familiarities[j, i] = self.step_familiarity
                    U[j, i] = np.cos(self.angle)
                    V[j, i] = np.sin(self.angle)
                else:
                    familiarities[j, i] = np.nan
                    U[j, i] = np.nan
                    V[j, i] = np.nan

        # Restore positions
        self.position = old_pos
        self.angle = old_ang
        self.angle_offsets = old_ang_offsets
        self.angle_familiarity = old_ang_familiarity

        return {
            'U' : U,
            'V' : V,
            'familiarities' : familiarities,
            'box_size' : box_size,
            'r' : r,
            'nX' : nX,
            'nY' : nY,
            'land_size' : land_size,
            'training_path' : self.training_path,
            'landscape' : self.landscape
        }


    @staticmethod
    def plot_quiver(U, V, familiarities, r, box_size, training_path, landscape,
                    nX, nY, land_size):

        # Plot
        fig = plt.figure()
        ax = plt.subplot()

        X, Y = r + box_size[1] / 2. + np.mgrid[0:nX] * box_size[1], r + box_size[0] / 2. + np.mgrid[0:nY] * box_size[0]

        ax.imshow(landscape, cmap = LANDSCAPE_CMAP, origin = 'lower', alpha = 0.4, zorder = 0)
        im = ax.imshow(familiarities, vmin = 0, origin = 'lower', extent = (r, r + land_size[0], r, r + land_size[1]), alpha = 0.3, cmap = "winter", zorder = 1)
        ax.plot(training_path[:,0], training_path[:,1], color = traincolor, linewidth = 3, zorder = 2)
        #im = ax.quiver(X, Y, U, V, familiarities, cmap = "winter", zorder = 2)
        ax.quiver(X, Y, U, V, pivot = "middle", zorder = 3, headwidth = 5, headlength = 4, headaxislength = 3.5)
        fig.colorbar(im)

        fig.tight_layout()

        return fig


    def _plot_landscape(self, main_ax, training_path = True):
        # -- Plot basic elements
        main_ax.imshow(self.landscape, cmap = LANDSCAPE_CMAP, vmax = 1.0, origin = 'lower', alpha = 0.6)
        if training_path:
            main_ax.plot(self.training_path[:,0], self.training_path[:,1], color = traincolor, linewidth = 3)

        scalebar_len_px = np.max(self.sensor_dimensions * self.sensor_pixel_dimensions)
        scalebar_fp = fm.FontProperties(size=11)
        scalebar = AnchoredSizeBar(main_ax.transData,
                                   scalebar_len_px,
                                   "%.0f %s" % (self.landscape_real_pixel_size * scalebar_len_px, self.sensor_real_area[1]),
                                   'lower right',
                                   pad = 0.2,
                                   color = 'k',
                                   frameon = True,
                                   size_vertical = 2,
                                   fontproperties = scalebar_fp)
        scalebar.patch.set_alpha(0.6)
        main_ax.add_artist(scalebar)


    def compass_plot(self,
                     figsize = (10, 8),
                     dpi = 180,
                     show_every = 50,
                     show_navpath = False,
                     frames = 20,
                     inner_radius = 0.25,
                     total_radius = 25.0,
                     arrow_radius = 1.0):
        fig = plt.figure(figsize = figsize, dpi = dpi)
        main_ax = plt.subplot()
        main_ax.xaxis.set_ticks([])
        main_ax.yaxis.set_ticks([])

        self._plot_landscape(main_ax, training_path = True)

        xpos, ypos = [self.position[0]], [self.position[1]]

        stoped_for = None

        position = self.position
        angle = self.angle

        inner_radius = inner_radius * total_radius
        graph_radius = total_radius - inner_radius
        full_radius = inner_radius + graph_radius
        agentz  = 12

        try:
            for i in range(frames):
                self.step_forward()
                xpos.append(self.position[0]); ypos.append(self.position[1])

                if i % show_every == 0:
                    x, y = position

                    agent_circle = matplotlib.patches.Circle(
                        xy = (x, y),
                        radius = 2,
                        edgecolor = None,
                        facecolor = 'k',
                        zorder = agentz
                    )
                    main_ax.add_patch(agent_circle)

                    scaledfam = self.n_sensor_pixels - self.angle_familiarity
                    scaledfam -= np.min(scaledfam)
                    scaledfam /= np.max(scaledfam)
                    scaledfam *= graph_radius
                    xs = x + np.cos(angle + self.angle_offsets) * (inner_radius + scaledfam)
                    ys = y + np.sin(angle + self.angle_offsets) * (inner_radius + scaledfam)
                    main_ax.plot(xs, ys, color = 'k', zorder = agentz, linewidth = 0.75)

                    wedge = matplotlib.patches.Wedge(
                        center = (x, y),
                        #width = graph_radius,
                        r = full_radius,
                        theta1 = (180 * (angle % (2 * np.pi)) / np.pi) - (self.saccade_degrees * 0.5),
                        theta2 = (180 * (angle % (2 * np.pi)) / np.pi) + (self.saccade_degrees * 0.5),
                        edgecolor = 'k',
                        facecolor = (0.6, 0., 1., 0.2),
                        linewidth = 0.75,
                        zorder = agentz - 2
                    )
                    main_ax.add_patch(wedge)

                    best_angle = angle + self.angle_offsets[np.argmin(self.angle_familiarity)]
                    main_ax.arrow(
                        x = x,
                        y = y,
                        dx = arrow_radius * self.step_size * np.cos(best_angle),
                        dy = arrow_radius * self.step_size * np.sin(best_angle),
                        zorder = agentz + 1,
                        width = 0.15,
                        facecolor = 'k',
                        edgecolor = 'k',
                        head_width = 2.5,
                    )

                position = self.position
                angle = self.angle

        except navsim.StopNavigationException as e:
            stoped_for = e

        if show_navpath:
            path_ln, = main_ax.plot(xpos, ypos, color = navcolor, linewidth = 1.5, linestyle = '--', zorder = 8)

        return fig, stoped_for


    def animate(self, frames, interval = 100):

        # -- Make figure & axes
        fig = plt.figure(figsize = (10, 8))
        gs = matplotlib.gridspec.GridSpec(4, 2,
                                          width_ratios = [2, 1],
                                          height_ratios = [6, 6, 6, 1])

        main_ax = plt.subplot(gs[:3, 0])
        sensor_ax = plt.subplot(gs[0, 1])
        scene_ax = plt.subplot(gs[1, 1])
        angle_ax = plt.subplot(gs[2, 1])
        status_ax = plt.subplot(gs[3, :])

        # -- Set titles, ticks, etc.

        scene_ax.set_title("Familiarity")
        scene_ax.set_xlabel("Training Scene")
        angle_ax.set_xlabel("Angle")

        status_ax.axes.get_xaxis().set_visible(False)
        status_ax.axes.get_yaxis().set_visible(False)
        for spline in ["top", "bottom", "left", "right"]:
            status_ax.spines[spline].set_visible(False)

        status_string = "%s. RMSD error: %0.2f; Coverage: %i%%"
        info_txt = status_ax.text(0.0, 0.0, "Step size: %0.1f; num. test angles: %i; sensor matrix: %i levels, %ix%i @ %ix%i px/px" % (self.step_size, self.n_test_angles, self.n_sensor_levels, self.sensor_dimensions[0], self.sensor_dimensions[1], self.sensor_pixel_dimensions[0], self.sensor_pixel_dimensions[1]), ha = 'left', va='center', fontsize = 10, zorder = 2, transform = status_ax.transAxes, backgroundcolor = "w")
        status_txt = status_ax.text(0.0, 0.8, status_string % ("Navigating", 0.0, 0.0), ha = 'left', va='center', fontsize = 10, animated = True, zorder = 2, transform = status_ax.transAxes, backgroundcolor = "w")

        sensor_ax.set_title("Sensor Matrix")
        init_sens_mat = np.zeros(shape = (self.sensor_dimensions[1], self.sensor_dimensions[0]))
        init_sens_mat[0, 0] = 1.
        sensor_im = sensor_ax.imshow(init_sens_mat, cmap = LANDSCAPE_CMAP, vmax = 1.0, alpha = 0.7,
                                     origin = 'lower', aspect = 'auto', animated = True)
        sensor_ax.xaxis.set_major_locator(plt.NullLocator())
        sensor_ax.yaxis.set_major_locator(plt.NullLocator())

        self._plot_landscape(main_ax, training_path = True)

        scene_inset_width = 20

        scene_ln_x = np.arange(len(self.scene_familiarity))
        scene_min = scene_ax.axvline(0, color = navcolor, animated = True, zorder = 1, alpha = 0.6, linewidth = 1)
        scene_min_box =  matplotlib.patches.Rectangle(xy = (0., 0.),
                                                   width = 2 * scene_inset_width + 1,
                                                   height = 0.5 * self.n_sensor_pixels,
                                                   alpha = 0.05,
                                                   animated = True,
                                                   color = navcolor)
        scene_ax.add_patch(scene_min_box)
        scene_ln, = scene_ax.plot(scene_ln_x, self.scene_familiarity, color = 'k', animated = True, linewidth = 0.75)
        scene_ax.set_ylim((0, 0.5 * self.n_sensor_pixels))
        scene_ax.yaxis.set_major_locator(plt.NullLocator())

        scene_inset_ax = inset_locator.inset_axes(scene_ax,
                                    width="35%",
                                    height="30%",
                                    loc = 1)
        scene_inset_ax.set_zorder(5)
        scene_inset_ax.set_facecolor([min(e + 0.76, 1.0) for e in matplotlib.colors.to_rgb(navcolor)])
        scene_inset_ax.axvline(scene_inset_width, color = navcolor, zorder = 1, alpha = 0.6, linewidth = 1.)
        scene_inset_dat = np.zeros(scene_inset_width * 2 + 1)
        scene_inset_ln, = scene_inset_ax.plot(np.arange(len(scene_inset_dat)), scene_inset_dat, color = 'k', animated = True, linewidth = 0.75, zorder = 10)
        scene_inset_ax.set_ylim((0, 0.3 * self.n_sensor_pixels))
        scene_inset_ax.xaxis.set_major_locator(plt.NullLocator())
        scene_inset_ax.yaxis.set_major_locator(plt.NullLocator())

        angle_min = angle_ax.axvline(0, color = navcolor, animated = True, zorder = 1, alpha = 0.6, linewidth = 1)
        angle_ln_x = 180. * self.angle_offsets / np.pi
        angle_ln, = angle_ax.plot(angle_ln_x, self.angle_familiarity, color = 'k', animated = True)
        angle_ax.set_ylim((0, 0.5 * self.n_sensor_pixels))
        angle_ax.xaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}°"))
        angle_ax.yaxis.set_major_locator(plt.NullLocator())

        xpos, ypos = [self.position[0]], [self.position[1]]
        path_ln, = main_ax.plot(xpos, ypos, color = navcolor, marker = 'o', markersize = 5, linewidth = 2, animated = True)

        sens_rect_dims = (self.sensor_dimensions * self.sensor_pixel_dimensions)[::-1]
        sensor_rect = matplotlib.patches.Rectangle(xy = (0., 0.),
                                                   width = sens_rect_dims[0],
                                                   height = sens_rect_dims[1],
                                                   angle = 0.,
                                                   alpha = 0.35,
                                                   animated = True,
                                                   color = navcolor)

        sens_rect_dims = (sens_rect_dims // 2)

        main_ax.add_patch(sensor_rect)

        #anim_ref = [1]

        self._anim_stop_cond = False

        def upd(frame):
            artist_list = (path_ln, scene_ln, scene_inset_ax, scene_inset_ln, scene_min, scene_min_box, angle_ln, angle_min, sensor_rect, sensor_im, status_txt)

            if not self._anim_stop_cond:
                # Step forward
                try:
                    self.step_forward()
                    new_sens_mat = self.get_sensor_mat(self.position, self.angle)
                except StopNavigationException as e:
                    self._anim_stop_cond = True
                    self.stopped_with_exception = e
                    status_txt.set_text(status_string % ("Stopped: %s" % e.get_reason(), self.navigation_error, 100 * self.percent_recapitulated))
                    status_txt.set_color("red")

                    #anim_ref[0].event_source.stop()
                    return artist_list

                xpos.append(self.position[0]); ypos.append(self.position[1])
                path_ln.set_data(xpos, ypos)

                status_txt.set_text(status_string % ("Navigating - position (%4.1f, %4.1f) heading %.0f°" % (self.position[0], self.position[1], 180. * self.angle / np.pi), self.navigation_error, 100*  self.percent_recapitulated))

                sensor_rect.set_xy((self.position[0] - sens_rect_dims[0] * np.cos(self.angle) + sens_rect_dims[1] * np.sin(self.angle),
                                    self.position[1] - sens_rect_dims[0] * np.sin(self.angle) - sens_rect_dims[1] * np.cos(self.angle)))
                sensor_rect.angle = np.rad2deg(self.angle)

                scene_ln.set_ydata(self.scene_familiarity)
                s_amin = np.argmin(self.scene_familiarity)
                scene_min.set_xdata([s_amin, s_amin])
                scene_min_box.set_xy((s_amin - scene_inset_width, 0.))

                scene_inset_dat[:] = np.nan
                scene_inset_dat[max(scene_inset_width - s_amin, 0):\
                                scene_inset_width + min(len(self.scene_familiarity) - s_amin, scene_inset_width + 1)] = \
                                self.scene_familiarity[max(s_amin - scene_inset_width, 0):s_amin + scene_inset_width + 1]
                scene_inset_ln.set_ydata(scene_inset_dat)

                angle_ln.set_ydata(self.angle_familiarity)
                a_amin = 180. * self.angle_offsets[np.argmin(self.angle_familiarity)] / np.pi
                angle_min.set_xdata([a_amin, a_amin])

                sensor_im.set_array(new_sens_mat)

            return artist_list

        anim = FuncAnimation(fig, upd,
                             frames = frames, interval = interval, #itertools.repeat(1),
                             blit = True, repeat = False)
        #anim_ref[0] = anim

        # Suppress the warning about unsupported axes for tight layout
        # See https://stackoverflow.com/questions/22227165/catch-matplotlib-warning#34622563
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
            fig.tight_layout()

        return fig, anim
