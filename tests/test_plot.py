import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ase.build import bulk  # noqa: E402
import numpy as np  # noqa: E402

from interfacemethod.plot import (  # noqa: E402
    check_for_holes,
    plot_equilibration,
    plot_melting_point_prediction,
    plot_solid_liquid_ratio,
    ratio_selection,
)
from interfacemethod.result import StrainPointResult  # noqa: E402
from interfacemethod.structure import get_voronoi_volume  # noqa: E402


def make_strain_result(strain=1.0, pressure=0.0, temperature=0.0, structure=None, parsed_output=None):
    return StrainPointResult(
        strain=strain,
        pressure=pressure,
        pressure_std=0.0,
        temperature=temperature,
        temperature_std=0.0,
        structure=structure,
        parsed_output=parsed_output,
    )


class PlotTestCase(unittest.TestCase):
    def tearDown(self):
        plt.close("all")


class TestPlotSolidLiquidRatio(PlotTestCase):
    def test_pure_fcc_structure_gives_full_solid_ratio(self):
        structure = bulk("Al", cubic=True).repeat(4)
        result = make_strain_result(strain=1.0, structure=structure, parsed_output={})

        ratio_lst = plot_solid_liquid_ratio(
            [result], {"crystalstructure": "fcc"}, debug_plot=False
        )

        self.assertEqual(len(ratio_lst), 1)
        self.assertAlmostEqual(ratio_lst[0], 1.0)

    def test_pure_diamond_structure_gives_full_solid_ratio(self):
        structure = bulk("Si", cubic=True).repeat(3)
        result = make_strain_result(strain=1.0, structure=structure, parsed_output={})

        ratio_lst = plot_solid_liquid_ratio(
            [result], {"crystalstructure": "diamond"}, debug_plot=False
        )

        self.assertEqual(len(ratio_lst), 1)
        self.assertAlmostEqual(ratio_lst[0], 1.0)

    def test_structure_not_matching_target_crystalstructure_gives_zero_ratio(self):
        structure = bulk("Al", cubic=True).repeat(4)
        result = make_strain_result(strain=1.0, structure=structure, parsed_output={})

        ratio_lst = plot_solid_liquid_ratio(
            [result], {"crystalstructure": "bcc"}, debug_plot=False
        )

        self.assertEqual(ratio_lst, [0.0])

    @patch("matplotlib.pyplot.show")
    def test_debug_plot_true_does_not_raise(self, mock_show):
        structure = bulk("Al", cubic=True).repeat(4)
        result = make_strain_result(strain=1.0, structure=structure, parsed_output={})

        plot_solid_liquid_ratio([result], {"crystalstructure": "fcc"}, debug_plot=True)

        self.assertTrue(mock_show.called)


class TestPlotEquilibration(PlotTestCase):
    def test_debug_plot_false_is_a_no_op(self):
        temperature_trace = np.full(30, 500.0)
        result = make_strain_result(
            strain=1.0, parsed_output={"generic": {"temperature": temperature_trace}}
        )

        self.assertIsNone(plot_equilibration([result], debug_plot=False))

    @patch("matplotlib.pyplot.show")
    def test_debug_plot_true_does_not_raise(self, mock_show):
        temperature_trace = np.concatenate([np.full(30, 500.0), np.full(20, 510.0)])
        result = make_strain_result(
            strain=1.0, parsed_output={"generic": {"temperature": temperature_trace}}
        )

        plot_equilibration([result], debug_plot=True)

        self.assertTrue(mock_show.called)


class TestPlotMeltingPointPrediction(PlotTestCase):
    def setUp(self):
        self.strain_lst = [0.98, 0.99, 1.0, 1.01, 1.02]
        self.pressure_lst = [-2.0, -1.0, 0.0, 1.0, 2.0]
        self.temperature_lst = [700.0, 850.0, 1000.0, 1300.0, 1900.0]
        self.results = [
            make_strain_result(strain=s, pressure=p, temperature=t)
            for s, p, t in zip(self.strain_lst, self.pressure_lst, self.temperature_lst)
        ]

    def test_temperature_mean_left_right_from_min_max_and_boundary(self):
        _, temperature_mean, temperature_left, temperature_right = (
            plot_melting_point_prediction(
                self.results, boundary_value=0.25, debug_plot=False
            )
        )

        t_min, t_max = min(self.temperature_lst), max(self.temperature_lst)
        self.assertAlmostEqual(temperature_mean, t_min + (t_max - t_min) * 0.5)
        self.assertAlmostEqual(temperature_left, t_min + (t_max - t_min) * 0.25)
        self.assertAlmostEqual(temperature_right, t_min + (t_max - t_min) * 0.75)

    def test_temperature_next_from_pressure_temperature_fit(self):
        temperature_next, *_ = plot_melting_point_prediction(
            self.results, boundary_value=0.25, debug_plot=False
        )

        expected_next = np.poly1d(
            np.polyfit(self.pressure_lst, self.temperature_lst, 1)
        )(0.0)
        self.assertAlmostEqual(temperature_next, expected_next)

    @patch("matplotlib.pyplot.show")
    def test_debug_plot_true_does_not_raise(self, mock_show):
        plot_melting_point_prediction(self.results, debug_plot=True)

        self.assertTrue(mock_show.called)


class TestRatioSelection(PlotTestCase):
    def test_selects_longest_contiguous_in_range_run(self):
        ratio_lst = [0.55, 0.9, 0.52, 0.51, 0.53, 0.9, 0.49, 0.9]
        strain_results = [make_strain_result(strain=i) for i in range(len(ratio_lst))]

        selected_results, sl_flag = ratio_selection(
            strain_results, ratio_lst, ratio_boundary=0.1, debug_plot=False
        )

        self.assertEqual([r.strain for r in selected_results], [2, 3, 4])
        self.assertEqual(sl_flag, 1)

    def test_negative_flag_when_selected_run_leans_liquid(self):
        ratio_lst = [0.9, 0.48, 0.47, 0.49, 0.9]
        strain_results = [make_strain_result(strain=i) for i in range(len(ratio_lst))]

        selected_results, sl_flag = ratio_selection(
            strain_results, ratio_lst, ratio_boundary=0.1, debug_plot=False
        )

        self.assertEqual([r.strain for r in selected_results], [1, 2, 3])
        self.assertEqual(sl_flag, -1)

    def test_no_values_within_boundary_returns_empty_selection(self):
        ratio_lst = [0.9, 0.95, 0.92]
        strain_results = [make_strain_result(strain=i) for i in range(len(ratio_lst))]

        selected_results, sl_flag = ratio_selection(
            strain_results, ratio_lst, ratio_boundary=0.1, debug_plot=False
        )

        self.assertEqual(selected_results, [])
        self.assertEqual(sl_flag, 1)

    @patch("matplotlib.pyplot.show")
    def test_debug_plot_true_does_not_raise(self, mock_show):
        ratio_lst = [0.55, 0.9, 0.52, 0.51, 0.53]
        strain_results = [make_strain_result(strain=i) for i in range(len(ratio_lst))]

        ratio_selection(strain_results, ratio_lst, ratio_boundary=0.1, debug_plot=True)

        self.assertTrue(mock_show.called)


class TestCheckForHoles(PlotTestCase):
    def test_matches_voronoi_volume_threshold_formula(self):
        structure_a = bulk("Al", cubic=True).repeat(3)
        structure_b = bulk("Al", cubic=True).repeat(3)
        del structure_b[0]
        results = [
            make_strain_result(strain=0.98, structure=structure_a),
            make_strain_result(strain=1.02, structure=structure_b),
        ]

        holes = check_for_holes(results, debug_plot=False)

        max_lst, mean_lst = get_voronoi_volume([structure_a, structure_b])
        expected = np.array(max_lst) < np.mean(mean_lst) * 2
        np.testing.assert_array_equal(holes, expected)

    @patch("matplotlib.pyplot.show")
    def test_debug_plot_true_does_not_raise(self, mock_show):
        structure_a = bulk("Al", cubic=True).repeat(2)
        structure_b = bulk("Al", cubic=True).repeat(2)
        results = [
            make_strain_result(strain=0.98, structure=structure_a),
            make_strain_result(strain=1.02, structure=structure_b),
        ]

        check_for_holes(results, debug_plot=True)

        self.assertTrue(mock_show.called)


if __name__ == "__main__":
    unittest.main()
