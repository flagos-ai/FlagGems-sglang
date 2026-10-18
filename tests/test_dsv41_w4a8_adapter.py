# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Historical adapter stays opt-in and preserves exact tactic intervals."""
import copy
import importlib
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class HistoricalAdapterTests(unittest.TestCase):
    def setUp(self):
        name = 'isolated_historical_w4a8'
        package = types.ModuleType(name)
        package.__path__ = [str(Path(__file__).parents[1] / 'src/flaggems_sglang_plugin')]
        self.patch = patch.dict(sys.modules, {name: package})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.adapter = importlib.import_module(name + '.w4a8_moe')

    def table(self, experts=48, stage='w13'):
        module = self.adapter
        points = module._points(experts)
        return [(0 if index == 0 else points[index - 1], point,
                 copy.deepcopy(module.BASELINE_TACTICS[f'{experts}/{stage}/{point}']))
                for index, point in enumerate(points)]

    def test_default_off_does_not_import_torch(self):
        with patch.dict(os.environ, {}, clear=True):
            before = set(sys.modules)
            self.assertIsNone(self.adapter.install())
            self.assertEqual(set(sys.modules), before)

    def test_opt_in_requires_all_original_flags(self):
        with patch.dict(os.environ, {self.adapter.ENV: '1'}, clear=True):
            with self.assertRaises(ValueError):
                self.adapter.install()

    def test_table_points_keep_interval_geometry_and_input_immutable(self):
        for experts in (16, 48):
            for stage in ('w13', 'w2'):
                table = self.table(experts, stage)
                original = copy.deepcopy(table)
                result = self.adapter.tune_table(table, experts, stage)
                self.assertEqual(table, original)
                for point in self.adapter._points(experts):
                    selected = [v for lo, hi, v in result if lo < point <= hi]
                    self.assertEqual(len(selected), 1)
                    expected = copy.deepcopy(self.adapter.BASELINE_TACTICS[f'{experts}/{stage}/{point}'])
                    expected['use_tma_c'] = False
                    self.assertEqual(selected[0], expected)

    def test_unreviewed_native_tactic_rejected(self):
        table = self.table()
        table[0][2]['num_stages'] += 1
        with self.assertRaisesRegex(ValueError, 'Unreviewed native'):
            self.adapter.tune_table(table, 48, 'w13')

    def test_uninstall_keeps_later_binding_owner(self):
        old, owned, later = object(), object(), object()
        module = types.SimpleNamespace(first=owned, second=later)
        handle = self.adapter.Installation([(module, 'first', old, owned),
                                            (module, 'second', old, owned)])
        handle.uninstall()
        self.assertIs(module.first, old)
        self.assertIs(module.second, later)


if __name__ == '__main__':
    unittest.main()
