from typing import List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

from interfacemethod.result import StrainPointResult
from interfacemethod.structure import analyse_structure, get_voronoi_volume


def plot_solid_liquid_ratio(
    strain_results: List[StrainPointResult], project_parameter, debug_plot=True
):
    cna_str = project_parameter["crystalstructure"].upper()
    ratio_lst = []
    for result in strain_results:
        strain = result.strain
        struct = result.structure.copy()
        struct.wrap()
        cna = analyse_structure(
            structure=struct,
            mode="str",
            diamond=project_parameter["crystalstructure"].lower() == "diamond",
        )
        if not project_parameter["crystalstructure"].lower() == "diamond":
            bcc_count = sum(cna == "BCC")
            fcc_count = sum(cna == "FCC")
            hcp_count = sum(cna == "HCP")
            cond = (
                (cna_str == "BCC" and bcc_count > fcc_count and bcc_count > hcp_count)
                or (
                    cna_str == "FCC" and fcc_count > bcc_count and fcc_count > hcp_count
                )
                or (
                    cna_str == "HCP" and hcp_count > bcc_count and hcp_count > fcc_count
                )
            )
        else:
            cna_str = "Cubic diamond"
            cond = sum(cna == cna_str) > 0.05 * len(struct)
        if cond:
            bandwidth = (struct.get_volume() / len(struct)) ** (1.0 / 3.0)
            kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
                struct.positions[:, 2][cna == cna_str].reshape(-1, 1)
            )
            z_range = np.linspace(
                struct.positions[:, 2].min(), struct.positions[:, 2].max(), 1000
            )
            sample = kde.score_samples(z_range.reshape(-1, 1))
            gaussian_funct = np.exp(sample) / np.exp(sample).max()
            z_range_above_limit = z_range[np.where(gaussian_funct > 0.1)]
            z_range_below_limit = z_range[np.where(gaussian_funct < 0.1)]
            if len(z_range_above_limit) != 0:
                ratio_above = (
                    np.max(z_range_above_limit) - np.min(z_range_above_limit)
                ) / (np.max(z_range) - np.min(z_range))
            else:
                ratio_above = 1.0
            if len(z_range_below_limit) != 0:
                ratio_below = 1 - (
                    np.max(z_range_below_limit) - np.min(z_range_below_limit)
                ) / (np.max(z_range) - np.min(z_range))
            else:
                ratio_below = 0.0
            if ratio_below == 0.0:
                ratio = ratio_above
            elif ratio_above == 1.0:
                ratio = ratio_below
            else:
                ratio = np.min([ratio_below, ratio_above])
            ratio_lst.append(ratio)
        else:
            z_range = None
            gaussian_funct = None
            z_range_above_limit = None
            ratio = None
            ratio_lst.append(0.0)
        if debug_plot:
            plt.title("strain: " + str(strain))
            plt.xlabel("position z")
            plt.ylabel("position x")
            plt.plot(struct.positions[:, 2], struct.positions[:, 0], "o", label="all")
            if not project_parameter["crystalstructure"].lower() == "diamond":
                plt.plot(
                    struct.positions[:, 2][cna == "BCC"],
                    struct.positions[:, 0][cna == "BCC"],
                    "x",
                    label="BCC",
                )
                plt.plot(
                    struct.positions[:, 2][cna == "FCC"],
                    struct.positions[:, 0][cna == "FCC"],
                    "x",
                    label="FCC",
                )
                plt.plot(
                    struct.positions[:, 2][cna == "HCP"],
                    struct.positions[:, 0][cna == "HCP"],
                    "x",
                    label="HCP",
                )
            else:
                plt.plot(
                    struct.positions[:, 2][cna == "Cubic diamond"],
                    struct.positions[:, 0][cna == "Cubic diamond"],
                    "x",
                    label="Cubic diamond",
                )
                plt.plot(
                    struct.positions[:, 2][cna == "Cubic diamond (1st neighbor)"],
                    struct.positions[:, 0][cna == "Cubic diamond (1st neighbor)"],
                    "x",
                    label="Cubic diamond (1st neighbor)",
                )
                plt.plot(
                    struct.positions[:, 2][cna == "Cubic diamond (2nd neighbor)"],
                    struct.positions[:, 0][cna == "Cubic diamond (2nd neighbor)"],
                    "x",
                    label="Cubic diamond (2nd neighbor)",
                )
                plt.plot(
                    struct.positions[:, 2][cna == "Hexagonal diamond"],
                    struct.positions[:, 0][cna == "Hexagonal diamond"],
                    "x",
                    label="Hexagonal diamond",
                )
                plt.plot(
                    struct.positions[:, 2][cna == "Hexagonal diamond (1st neighbor)"],
                    struct.positions[:, 0][cna == "Hexagonal diamond (1st neighbor)"],
                    "x",
                    label="Hexagonal diamond (1st neighbor)",
                )
                plt.plot(
                    struct.positions[:, 2][cna == "Hexagonal diamond (2nd neighbor)"],
                    struct.positions[:, 0][cna == "Hexagonal diamond (2nd neighbor)"],
                    "x",
                    label="Hexagonal diamond (2nd neighbor)",
                )
            cna_str_lst = struct.positions[:, 2][cna == cna_str]
            if len(cna_str_lst) != 0:
                plt.axvline(cna_str_lst.max(), color="red")
                plt.axvline(cna_str_lst.min(), color="red")
            plt.legend()
            plt.show()
            plt.xlabel("Position in z")
            plt.ylabel("kernel density score")
            plt.title("strain: " + str(strain))
            if z_range is not None:
                plt.plot(z_range, gaussian_funct, label=cna_str)
                plt.axvline(
                    np.min(z_range_above_limit),
                    color="black",
                    linestyle="--",
                    label="ratio: " + str(ratio),
                )
                plt.axvline(np.max(z_range_above_limit), color="black", linestyle="--")
            plt.axhline(0.1, color="red")
            plt.legend()
            plt.show()
    return ratio_lst


def plot_equilibration(strain_results: List[StrainPointResult], debug_plot=True):
    if debug_plot:
        for result in strain_results:
            temperature_trace = result.parsed_output["generic"]["temperature"]
            plt.plot(temperature_trace, label="strain: " + str(result.strain))
            plt.axhline(
                np.mean(temperature_trace[-20:]),
                linestyle="--",
                color="red",
            )
            plt.axvline(
                range(len(temperature_trace))[-20],
                linestyle="--",
                color="black",
            )
            plt.legend()
            plt.xlabel("timestep")
            plt.ylabel("Temperature K")
            plt.legend()
            plt.show()


def plot_melting_point_prediction(
    strain_results: List[StrainPointResult],
    boundary_value: float = 0.25,
    debug_plot: bool = True,
) -> tuple[float, float, float, float]:
    strain_value_lst = [r.strain for r in strain_results]
    pressure_value_lst = [r.pressure for r in strain_results]
    temperature_value_lst = [r.temperature for r in strain_results]
    fit_press = np.poly1d(np.polyfit(strain_value_lst, pressure_value_lst, 1))
    fit_temp = np.poly1d(np.polyfit(strain_value_lst, temperature_value_lst, 1))
    fit_temp_from_press = np.poly1d(
        np.polyfit(pressure_value_lst, temperature_value_lst, 1)
    )
    fit_combined = np.poly1d(
        np.polyfit(fit_press(strain_value_lst), fit_temp(strain_value_lst), 1)
    )
    if debug_plot:
        plt.plot(strain_value_lst, pressure_value_lst, "o", label="pressure (strain)")
        plt.plot(strain_value_lst, fit_press(strain_value_lst), label="fit")
        plt.xlabel("Strain")
        plt.ylabel("Pressure GPa")
        plt.legend()
        plt.show()
        plt.plot(
            strain_value_lst, temperature_value_lst, "o", label="temperature (strain)"
        )
        plt.plot(strain_value_lst, fit_temp(strain_value_lst), label="fit")
        plt.xlabel("Strain")
        plt.ylabel("Temperature K")
        plt.legend()
        plt.show()
        plt.plot(
            pressure_value_lst,
            temperature_value_lst,
            "o",
            label="temperature (pressure)",
        )
        plt.plot(
            pressure_value_lst,
            fit_temp_from_press(pressure_value_lst),
            label="fit direct",
        )
        plt.plot(
            fit_press(strain_value_lst),
            fit_temp(strain_value_lst),
            label="combined fits",
        )
        plt.xlabel("Pressure GPa")
        plt.ylabel("Temperature K")
        plt.legend()
        plt.show()
    print(fit_temp_from_press(0.0), fit_combined(0.0))
    temperature_mean = (
        np.min(temperature_value_lst)
        + (np.max(temperature_value_lst) - np.min(temperature_value_lst)) * 1 / 2
    )
    temperature_left = np.min(temperature_value_lst) + (
        np.max(temperature_value_lst) - np.min(temperature_value_lst)
    ) * (1 / 2 - boundary_value)
    temperature_right = np.min(temperature_value_lst) + (
        np.max(temperature_value_lst) - np.min(temperature_value_lst)
    ) * (1 / 2 + boundary_value)
    temperature_next = fit_temp_from_press(0.0)
    return temperature_next, temperature_mean, temperature_left, temperature_right


def ratio_selection(
    strain_results: List[StrainPointResult],
    ratio_lst,
    ratio_boundary,
    debug_plot: bool = True,
):
    """
    Keep the longest contiguous run of strain points whose solid/liquid ratio falls within
    ``ratio_boundary`` of 0.5, and report whether that run leans solid (+1) or liquid (-1).
    """
    if debug_plot:
        plt.plot([r.strain for r in strain_results], ratio_lst)
        plt.axhline(0.5 + ratio_boundary, color="red", linestyle="--")
        plt.axhline(0.5, color="black", linestyle="--")
        plt.axhline(0.5 - ratio_boundary, color="red", linestyle="--")
        plt.xlabel("Strain")
        plt.ylabel("ratio solid vs. liquid")
    rat_lst, rat_col_lst = [], []
    for rat in ratio_lst:
        if (0.5 - ratio_boundary) < rat < (0.5 + ratio_boundary):
            rat_lst.append(rat)
        elif len(rat_lst) != 0:
            rat_col_lst.append(rat_lst)
            rat_lst = []
    if len(rat_lst) != 0:
        rat_col_lst.append(rat_lst)
    if len(rat_col_lst) != 0:
        rat_max_ind = np.argmax([len(lst) for lst in rat_col_lst])
        ratio_ind = [r in rat_col_lst[rat_max_ind] for r in ratio_lst]
        selected_results = [r for r, keep in zip(strain_results, ratio_ind) if keep]
        ratio_value_lst = np.array(ratio_lst)[ratio_ind]
        if debug_plot:
            strain_value_lst = [r.strain for r in selected_results]
            plt.axvline(np.min(strain_value_lst), color="blue", linestyle="--")
            plt.axvline(np.max(strain_value_lst), color="blue", linestyle="--")
            plt.show()
        sl_flag = 1 if np.mean(ratio_value_lst) > 0.5 else -1
        return selected_results, sl_flag
    else:
        sl_flag = 1 if np.mean(ratio_lst) > 0.5 else -1
        return [], sl_flag


def check_for_holes(strain_results: List[StrainPointResult], debug_plot=True):
    strain_value_lst = [r.strain for r in strain_results]
    max_lst, mean_lst = get_voronoi_volume(
        structure_lst=[r.structure for r in strain_results]
    )
    if debug_plot:
        plt.plot(strain_value_lst, mean_lst, label="mean")
        plt.plot(strain_value_lst, max_lst, label="max")
        plt.axhline(np.mean(mean_lst) * 2, color="black", linestyle="--")
        plt.legend()
        plt.xlabel("Strain")
        plt.ylabel("Voronoi Volume")
        plt.show()
    return np.array(max_lst) < np.mean(mean_lst) * 2
