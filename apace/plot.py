from collections.abc import Iterable
from itertools import groupby
from math import inf
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.gridspec as grid_spec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from matplotlib.path import Path
from matplotlib.widgets import Slider

from .classes import (
    Base,
    Dipole,
    Drift,
    Element,
    Lattice,
    Octupole,
    Quadrupole,
    Sextupole,
)

FONT_SIZE = 8


class Color:
    RED = "#EF4444"
    YELLOW = "#FBBF24"
    GREEN = "#10B981"
    BLUE = "#3B82F6"
    ORANGE = "F97316"
    PURPLE = "#8B5CF6"
    CYAN = "#06B6D4"
    WHITE = "white"
    BLACK = "black"
    LIGHT_GRAY = "#E5E7EB"


ELEMENT_COLOR: Dict[type, str] = {
    Dipole: Color.YELLOW,
    Quadrupole: Color.RED,
    Sextupole: Color.GREEN,
    Octupole: Color.BLUE,
}


OPTICAL_FUNCTIONS = {
    "beta_x": (r"$\beta_x$ / m", Color.RED),
    "beta_y": (r"$\beta_y$ / m", Color.BLUE),
    "eta_x": (r"$\eta_x$ / m", Color.GREEN),
    "eta_x_dds": (r"$\eta_x'$ / m", Color.ORANGE),
    "psi_x": (r"$\psi_x$", Color.YELLOW),
    "psi_y": (r"$\psi_y$", Color.ORANGE),
    "alpha_x": (r"$\alpha_x$", Color.PURPLE),
    "alpha_y": (r"$\alpha_y$", Color.CYAN),
}


def draw_elements(
    ax: mpl.axes.Axes,
    lattice: Lattice,
    *,
    labels: bool = True,
    location: str = "top",
):
    """Draw the elements of a lattice onto a matplotlib axes."""
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    rect_height = 0.05 * (y_max - y_min)
    if location == "top":
        y0 = y_max = y_max + rect_height
    else:
        y0 = y_min - rect_height
        y_min -= 3 * rect_height
        plt.hlines(y0, x_min, x_max, color="black", linewidth=1)
    ax.set_ylim(y_min, y_max)

    sign = -1
    start = end = 0
    for element, group in groupby(lattice.sequence):
        start = end
        end += element.length * sum(1 for _ in group)
        if end <= x_min:
            continue
        elif start >= x_max:
            break

        try:
            color = ELEMENT_COLOR[type(element)]
        except KeyError:
            continue

        y0_local = y0
        if isinstance(element, Dipole) and element.angle < 0:
            y0_local -= rect_height / 2
        elif isinstance(element, Quadrupole) and element.k1 < 0:
            y0_local -= rect_height / 2
        elif isinstance(element, Sextupole) and element.k2 < 0:
            y0_local -= rect_height / 2
        elif isinstance(element, Octupole) and element.k3 < 0:
            y0_local -= rect_height / 2

        ax.add_patch(
            plt.Rectangle(
                (max(start, x_min), y0_local),
                min(end, x_max) - max(start, x_min),
                rect_height / 2,
                facecolor=color,
                clip_on=False,
                zorder=10,
            )
        )
        if labels and type(element) in {Dipole, Quadrupole}:
            sign = -sign
            ax.annotate(
                element.name,
                xy=((start + end) / 2, y0 + sign * rect_height),
                fontsize=FONT_SIZE,
                ha="center",
                va="center",
                annotation_clip=False,
                zorder=11,
            )


def draw_sub_lattices(
    ax: mpl.axes.Axes,
    lattice: Lattice,
    *,
    labels: bool = True,
    location: str = "bottom",
):
    x_min, x_max = ax.get_xlim()
    length_gen = [0.0, *(obj.length for obj in lattice.children)]
    position_list = np.add.accumulate(length_gen)
    i_min = np.searchsorted(position_list, x_min)
    i_max = np.searchsorted(position_list, x_max)
    ticks = position_list[i_min : i_max + 1]
    ax.set_xticks(ticks)
    # if len(ticks) < 5:
    #     ax.xaxis.set_minor_locator(AutoMinorLocator())
    #     ax.xaxis.set_minor_formatter(ScalarFormatter())
    ax.grid(axis="x", color=Color.LIGHT_GRAY, linestyle="--", linewidth=1)

    if labels:
        y_min, y_max = ax.get_ylim()
        height = 0.08 * (y_max - y_min)
        if location == "top":
            y0, y_max = y_max, y_max + height
        else:
            y0, y_min = y_min - height / 3, y_min - height

        ax.set_ylim(y_min, y_max)
        start = end = 0
        for obj in lattice.children:
            end += obj.length
            if not isinstance(obj, Lattice) or start >= x_max or end <= x_min:
                continue

            x0 = end - obj.length / 2
            ax.annotate(
                obj.name,
                xy=(x0, y0),
                fontsize=FONT_SIZE + 2,
                fontstyle="oblique",
                alpha=0.5,
                va="center",
                ha="center",
                clip_on=True,
                zorder=102,
            )
            start = end


def plot_twiss(
    ax,
    twiss,
    *,
    twiss_functions=("beta_x", "beta_y", "eta_x"),
    scales={"eta_x": 10, "eta_x_dds": 10},
    line_style="solid",
    line_width=1.3,
    alpha=1.0,
    show_ylabels=False,
):
    if scales is None:
        scales = {}

    text_areas = []
    for i, function in enumerate(twiss_functions):
        value = getattr(twiss, function)
        label, color = OPTICAL_FUNCTIONS[function]
        scale = scales.get(function)
        if scale is not None:
            label = f"{scale} {label}"
            value = scale * value

        ax.plot(
            twiss.s,
            value,
            color=color,
            linewidth=line_width,
            linestyle=line_style,
            alpha=alpha,
            zorder=10 - i,
            label=label,
        )
        text_areas.insert(0, TextArea(label, textprops=dict(color=color, rotation=90)))

    ax.set_xlabel("Orbit Position $s$ / m")
    if show_ylabels:
        ax.add_artist(
            AnchoredOffsetbox(
                child=VPacker(children=text_areas, align="bottom", pad=0, sep=20),
                loc="center left",
                bbox_to_anchor=(-0.125, 0, 1.125, 1),
                bbox_transform=ax.transAxes,
                frameon=False,
            )
        )


def _twiss_plot_section(
    ax,
    twiss,
    *,
    x_min=-inf,
    x_max=inf,
    y_min=None,
    y_max=None,
    annotate_elements=True,
    annotate_lattices=True,
    line_style="solid",
    line_width=1.3,
    ref_twiss=None,
    scales={"eta_x": 10},
    overwrite=False,
):
    if overwrite:
        ax.clear()
    if ref_twiss:
        plot_twiss(
            ax,
            ref_twiss,
            line_style="dashed",
            line_width=2.5,
            alpha=0.5,
        )

    plot_twiss(ax, twiss, line_style=line_style, line_width=line_width, scales=scales)
    x_min = max(x_min, 0)
    x_max = min(x_max, twiss.lattice.length)
    if y_min is None:
        y_min = -0.5
    if y_max is None:
        y_max = ax.get_ylim()[1]

    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    draw_elements(ax, twiss.lattice, labels=annotate_elements)
    draw_sub_lattices(ax, twiss.lattice, labels=annotate_lattices)


# TODO:
#   * make sub_class of figure
#   * add attribute which defines which twiss parameters are plotted
#   * add twiss_functions argument similar to plot_twiss
class TwissPlot:
    """Convenience class to plot twiss parameters

    :param Twiss twiss: The name of the object.
    :param tuple: List of sections to plot. Can be either (min, max), "name" or object.
    :type tuple: List[Union[Tuple[float, float], str, Base]
    :param y_max float: Maximum y-limit
    :param y_min float: Minimum y-limit
    :param main bool: Wheter to plot whole ring or only given sections
    :param scales Dict[str, int]: Optional scaling factors for optical functions
    :param Twiss ref_twiss: Reference twiss values. Will be plotted as dashed lines.
    :param pairs: List of (element, attribute)-pairs to create interactice sliders for.
    :type pairs: List[Tuple[Element, str]]
    """

    def __init__(
        self,
        twiss,
        twiss_functions=("beta_x", "beta_y", "eta_x"),
        *,
        sections=None,
        y_min=None,
        y_max=None,
        main=True,
        scales={"eta_x": 10},
        ref_twiss=None,
        pairs: Optional[List[Tuple[Element, str]]] = None,
    ):
        self.fig = plt.figure()
        self.twiss = twiss
        self.lattice = twiss.lattice
        self.twiss_functions = twiss_functions
        self.scales = scales
        height_ratios = [4, 14] if (main and sections) else [1]
        main_grid = grid_spec.GridSpec(
            len(height_ratios), 1, self.fig, height_ratios=height_ratios
        )
        self.axs_sections = []  # TODO: needed for update function

        if pairs:
            _, axs = plt.subplots(nrows=len(pairs))
            if not isinstance(axs, Iterable):
                axs = (axs,)
            self.sliders = []
            for ax, (element, attribute) in zip(axs, pairs):
                initial_value = getattr(element, attribute)
                label = f"{element.name} {attribute}"
                slider = Slider(ax, label, -5, 5, initial_value)
                slider.on_changed(
                    lambda value, element=element, attribute=attribute: (
                        setattr(element, attribute, value),
                        self.update(),
                    )
                )
                self.sliders.append(slider)  # prevent garbage collection

        if main:
            self.ax_main = self.fig.add_subplot(main_grid[0])
            _twiss_plot_section(
                self.ax_main,
                self.twiss,
                ref_twiss=ref_twiss,
                y_min=y_min,
                y_max=y_max,
                annotate_elements=False,
                scales=scales,
            )

        if sections:
            n_sections = len(sections)
            rows, cols = find_optimal_grid(n_sections)
            sub_grid = grid_spec.GridSpecFromSubplotSpec(rows, cols, main_grid[1])
            self.axs_sections = [
                self.fig.add_subplot(sub_grid[i]) for i in range(len(sections))
            ]
            for i, section in enumerate(sections):
                if isinstance(section, (str, Base)):
                    obj = self.lattice[section] if isinstance(section, str) else section
                    index = self.lattice.indices[obj][0]
                    x_min = sum(obj.length for obj in self.lattice.sequence[:index])
                    x_max = x_min + obj.length
                else:
                    x_min, x_max = section

                _twiss_plot_section(
                    self.axs_sections[i],
                    self.twiss,
                    ref_twiss=ref_twiss,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    annotate_elements=True,
                    scales=scales,
                )

        handles, labels = self.fig.axes[0].get_legend_handles_labels()
        self.fig.legend(handles, labels, loc="upper left", ncol=10, frameon=False)
        title = self.lattice.name
        if self.lattice.info != "":
            title += f"({self.lattice.info})"

        self.fig.suptitle(title, ha="right", x=0.98)
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.9)

    def update(self):
        twiss = self.twiss
        for ax in [self.ax_main] + self.axs_sections:
            for line, function in zip(ax.lines, self.twiss_functions):
                data = getattr(twiss, function)
                scale = self.scales.get(function)
                if scale is not None:
                    data *= scale
                line.set_data(twiss.s, data)
        self.fig.canvas.draw_idle()


def find_optimal_grid(n):
    rows, cols = 1, 1
    while rows * cols < n:
        cols += 1
        rows, cols = cols, rows

    return (rows, cols) if cols >= rows else (cols, rows)


def floor_plan(
    ax: mpl.axes.Axes,
    lattice: Lattice,
    *,
    start_angle: float = 0,
    labels: bool = True,
    direction: str = "clockwise",
):
    ax.set_aspect("equal")
    codes = Path.MOVETO, Path.LINETO
    current_angle = start_angle
    start = np.zeros(2)
    end = np.zeros(2)
    x_min = y_min = 0
    x_max = y_max = 0
    sign = 1
    for element, group in groupby(lattice.sequence):
        start = end.copy()
        length = element.length * sum(1 for _ in group)
        if isinstance(element, Drift):
            color = Color.BLACK
            line_width = 1
        else:
            color = ELEMENT_COLOR[type(element)]
            line_width = 6

        # TODO: refactor current angle
        angle = 0
        if isinstance(element, Dipole):
            angle = element.k0 * length
            radius = length / angle
            vec = radius * np.array([np.sin(angle), 1 - np.cos(angle)])
            sin = np.sin(current_angle)
            cos = np.cos(current_angle)
            rot = np.array([[cos, -sin], [sin, cos]])
            end += rot @ vec

            angle_center = current_angle + np.pi / 2
            center = start + radius * np.array(
                [np.cos(angle_center), np.sin(angle_center)]
            )
            diameter = 2 * radius
            arc_angle = -90
            theta1 = current_angle * 180 / np.pi
            theta2 = (current_angle + angle) * 180 / np.pi
            if angle < 0:
                theta1, theta2 = theta2, theta1

            line = patches.Arc(
                center,
                width=diameter,
                height=diameter,
                angle=arc_angle,
                theta1=theta1,
                theta2=theta2,
                color=color,
                linewidth=line_width,
            )
            current_angle += angle
        else:
            end += length * np.array([np.cos(current_angle), np.sin(current_angle)])
            line = patches.PathPatch(
                Path((start, end), codes), color=color, linewidth=line_width
            )

        x_min = min(x_min, end[0])
        y_min = min(y_min, end[1])
        x_max = max(x_max, end[0])
        y_max = max(y_max, end[1])

        ax.add_patch(line)  # TODO: currently splitted elements get drawn twice

        if labels and isinstance(element, (Dipole, Quadrupole)):
            angle_center = (current_angle - angle / 2) + np.pi / 2
            sign = -sign
            center = (start + end) / 2 + sign * 0.5 * np.array(
                [np.cos(angle_center), np.sin(angle_center)]
            )
            ax.annotate(
                element.name,
                xy=center,
                fontsize=6,
                ha="center",
                va="center",
                # rotation=(current_angle * 180 / np.pi -90) % 180,
                annotation_clip=False,
                zorder=11,
            )

    margin = 0.01 * max((x_max - x_min), (y_max - y_min))
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    return ax
