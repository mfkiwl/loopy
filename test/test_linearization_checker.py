from __future__ import division, print_function

__copyright__ = "Copyright (C) 2019 James Stevens"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import six  # noqa: F401
import sys
import numpy as np
import loopy as lp
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl
    as pytest_generate_tests)
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa
import logging
from loopy import (
    preprocess_kernel,
    get_one_linearized_kernel,
)

logger = logging.getLogger(__name__)

try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()


def test_lexschedule_and_islmap_creation():
    import islpy as isl
    from loopy.schedule.checker import (
        get_schedule_for_statement_pair,
        get_isl_maps_for_LexSchedule,
    )
    from loopy.schedule.checker.utils import (
        align_isl_maps_by_var_names,
    )

    # example kernel
    knl = lp.make_kernel(
        [
            "{[i]: 0<=i<pi}",
            "{[k]: 0<=k<pk}",
            "{[j]: 0<=j<pj}",
            "{[t]: 0<=t<pt}",
        ],
        """
        for i
            for k
                <>temp = b[i,k]  {id=insn_a}
            end
            for j
                a[i,j] = temp + 1  {id=insn_b,dep=insn_a}
                c[i,j] = d[i,j]  {id=insn_c}
            end
        end
        for t
            e[t] = f[t]  {id=insn_d}
        end
        """,
        name="example",
        assumptions="pi,pj,pk,pt >= 1",
        lang_version=(2018, 2)
        )
    knl = lp.add_and_infer_dtypes(
            knl,
            {"b": np.float32, "d": np.float32, "f": np.float32})
    knl = lp.prioritize_loops(knl, "i,k")
    knl = lp.prioritize_loops(knl, "i,j")

    # get a linearization
    knl = preprocess_kernel(knl)
    knl = get_one_linearized_kernel(knl)
    linearization_items = knl.linearization

    # Create LexSchedule: mapping of {statement instance: lex point}
    lex_sched_ab = get_schedule_for_statement_pair(
        knl,
        linearization_items,
        "insn_a",
        "insn_b",
        )
    lex_sched_ac = get_schedule_for_statement_pair(
        knl,
        linearization_items,
        "insn_a",
        "insn_c",
        )
    lex_sched_ad = get_schedule_for_statement_pair(
        knl,
        linearization_items,
        "insn_a",
        "insn_d",
        )
    lex_sched_bc = get_schedule_for_statement_pair(
        knl,
        linearization_items,
        "insn_b",
        "insn_c",
        )
    lex_sched_bd = get_schedule_for_statement_pair(
        knl,
        linearization_items,
        "insn_b",
        "insn_d",
        )
    lex_sched_cd = get_schedule_for_statement_pair(
        knl,
        linearization_items,
        "insn_c",
        "insn_d",
        )

    # Relationship between insn_a and insn_b ---------------------------------------

    assert lex_sched_ab.stmt_instance_before.lex_pt == [0, 'i', 0, 'k', 0]
    assert lex_sched_ab.stmt_instance_after.lex_pt == [0, 'i', 1, 'j', 0]

    # Get two isl maps representing the LexSchedule

    isl_sched_map_before, isl_sched_map_after = get_isl_maps_for_LexSchedule(
        lex_sched_ab, knl)

    # Create expected maps, align, compare

    isl_sched_map_before_expected = isl.Map(
        "[pi, pk] -> { "
        "[statement = 0, i, k] -> [l0 = 0, l1 = i, l2 = 0, l3 = k, l4 = 0] : "
        "0 <= i < pi and 0 <= k < pk }"
        )
    isl_sched_map_before_expected = align_isl_maps_by_var_names(
        isl_sched_map_before_expected, isl_sched_map_before)

    isl_sched_map_after_expected = isl.Map(
        "[pi, pj] -> { "
        "[statement = 1, i, j] -> [l0 = 0, l1 = i, l2 = 1, l3 = j, l4 = 0] : "
        "0 <= i < pi and 0 <= j < pj }"
        )
    isl_sched_map_after_expected = align_isl_maps_by_var_names(
        isl_sched_map_after_expected, isl_sched_map_after)

    assert isl_sched_map_before == isl_sched_map_before_expected
    assert isl_sched_map_after == isl_sched_map_after_expected

    # ------------------------------------------------------------------------------
    # Relationship between insn_a and insn_c ---------------------------------------

    assert lex_sched_ac.stmt_instance_before.lex_pt == [0, 'i', 0, 'k', 0]
    assert lex_sched_ac.stmt_instance_after.lex_pt == [0, 'i', 1, 'j', 0]

    # Get two isl maps representing the LexSchedule

    isl_sched_map_before, isl_sched_map_after = get_isl_maps_for_LexSchedule(
        lex_sched_ac, knl)

    # Create expected maps, align, compare

    isl_sched_map_before_expected = isl.Map(
        "[pi, pk] -> { "
        "[statement = 0, i, k] -> [l0 = 0, l1 = i, l2 = 0, l3 = k, l4 = 0] : "
        "0 <= i < pi and 0 <= k < pk }"
        )
    isl_sched_map_before_expected = align_isl_maps_by_var_names(
        isl_sched_map_before_expected, isl_sched_map_before)

    isl_sched_map_after_expected = isl.Map(
        "[pi, pj] -> { "
        "[statement = 1, i, j] -> [l0 = 0, l1 = i, l2 = 1, l3 = j, l4 = 0] : "
        "0 <= i < pi and 0 <= j < pj }"
        )
    isl_sched_map_after_expected = align_isl_maps_by_var_names(
        isl_sched_map_after_expected, isl_sched_map_after)

    assert isl_sched_map_before == isl_sched_map_before_expected
    assert isl_sched_map_after == isl_sched_map_after_expected

    # ------------------------------------------------------------------------------
    # Relationship between insn_a and insn_d ---------------------------------------

    # insn_a and insn_d could have been linearized in either order
    # (i loop could be before or after t loop)
    def perform_insn_ad_checks_with(sid_a, sid_d):
        assert lex_sched_ad.stmt_instance_before.lex_pt == [sid_a, 'i', 0, 'k', 0]
        assert lex_sched_ad.stmt_instance_after.lex_pt == [sid_d, 't', 0, 0, 0]

        # Get two isl maps representing the LexSchedule

        isl_sched_map_before, isl_sched_map_after = get_isl_maps_for_LexSchedule(
            lex_sched_ad, knl)

        # Create expected maps, align, compare

        isl_sched_map_before_expected = isl.Map(
            "[pi, pk] -> { "
            "[statement = %d, i, k] -> [l0 = %d, l1 = i, l2 = 0, l3 = k, l4 = 0] : "
            "0 <= i < pi and 0 <= k < pk }"
            % (sid_a, sid_a)
            )
        isl_sched_map_before_expected = align_isl_maps_by_var_names(
            isl_sched_map_before_expected, isl_sched_map_before)

        isl_sched_map_after_expected = isl.Map(
            "[pt] -> { "
            "[statement = %d, t] -> [l0 = %d, l1 = t, l2 = 0, l3 = 0, l4 = 0] : "
            "0 <= t < pt }"
            % (sid_d, sid_d)
            )
        isl_sched_map_after_expected = align_isl_maps_by_var_names(
            isl_sched_map_after_expected, isl_sched_map_after)

        assert isl_sched_map_before == isl_sched_map_before_expected
        assert isl_sched_map_after == isl_sched_map_after_expected

    if lex_sched_ad.stmt_instance_before.stmt.int_id == 0:
        perform_insn_ad_checks_with(0, 1)
    else:
        perform_insn_ad_checks_with(1, 0)

    # ------------------------------------------------------------------------------
    # Relationship between insn_b and insn_c ---------------------------------------

    # insn_b and insn_c could have been linearized in either order
    # (i loop could be before or after t loop)
    def perform_insn_bc_checks_with(sid_b, sid_c):
        assert lex_sched_bc.stmt_instance_before.lex_pt == [0, 'i', 0, 'j', sid_b]
        assert lex_sched_bc.stmt_instance_after.lex_pt == [0, 'i', 0, 'j', sid_c]

        # Get two isl maps representing the LexSchedule

        isl_sched_map_before, isl_sched_map_after = get_isl_maps_for_LexSchedule(
            lex_sched_bc, knl)

        # Create expected maps, align, compare

        isl_sched_map_before_expected = isl.Map(
            "[pi, pj] -> { "
            "[statement = %d, i, j] -> [l0 = 0, l1 = i, l2 = 0, l3 = j, l4 = %d] : "
            "0 <= i < pi and 0 <= j < pj }"
            % (sid_b, sid_b)
            )
        isl_sched_map_before_expected = align_isl_maps_by_var_names(
            isl_sched_map_before_expected, isl_sched_map_before)

        isl_sched_map_after_expected = isl.Map(
            "[pi, pj] -> { "
            "[statement = %d, i, j] -> [l0 = 0, l1 = i, l2 = 0, l3 = j, l4 = %d] : "
            "0 <= i < pi and 0 <= j < pj }"
            % (sid_c, sid_c)
            )
        isl_sched_map_after_expected = align_isl_maps_by_var_names(
            isl_sched_map_after_expected, isl_sched_map_after)

        assert isl_sched_map_before == isl_sched_map_before_expected
        assert isl_sched_map_after == isl_sched_map_after_expected

    if lex_sched_bc.stmt_instance_before.stmt.int_id == 0:
        perform_insn_bc_checks_with(0, 1)
    else:
        perform_insn_bc_checks_with(1, 0)

    # ------------------------------------------------------------------------------
    # Relationship between insn_b and insn_d ---------------------------------------

    # insn_b and insn_d could have been linearized in either order
    # (i loop could be before or after t loop)
    def perform_insn_bd_checks_with(sid_b, sid_d):
        assert lex_sched_bd.stmt_instance_before.lex_pt == [sid_b, 'i', 0, 'j', 0]
        assert lex_sched_bd.stmt_instance_after.lex_pt == [sid_d, 't', 0, 0, 0]

        # Get two isl maps representing the LexSchedule

        isl_sched_map_before, isl_sched_map_after = get_isl_maps_for_LexSchedule(
            lex_sched_bd, knl)

        # Create expected maps, align, compare

        isl_sched_map_before_expected = isl.Map(
            "[pi, pj] -> { "
            "[statement = %d, i, j] -> [l0 = %d, l1 = i, l2 = 0, l3 = j, l4 = 0] : "
            "0 <= i < pi and 0 <= j < pj }"
            % (sid_b, sid_b)
            )
        isl_sched_map_before_expected = align_isl_maps_by_var_names(
            isl_sched_map_before_expected, isl_sched_map_before)

        isl_sched_map_after_expected = isl.Map(
            "[pt] -> { "
            "[statement = %d, t] -> [l0 = %d, l1 = t, l2 = 0, l3 = 0, l4 = 0] : "
            "0 <= t < pt }"
            % (sid_d, sid_d)
            )
        isl_sched_map_after_expected = align_isl_maps_by_var_names(
            isl_sched_map_after_expected, isl_sched_map_after)

        assert isl_sched_map_before == isl_sched_map_before_expected
        assert isl_sched_map_after == isl_sched_map_after_expected

    if lex_sched_bd.stmt_instance_before.stmt.int_id == 0:
        perform_insn_bd_checks_with(0, 1)
    else:
        perform_insn_bd_checks_with(1, 0)

    # ------------------------------------------------------------------------------
    # Relationship between insn_c and insn_d ---------------------------------------

    # insn_c and insn_d could have been linearized in either order
    # (i loop could be before or after t loop)
    def perform_insn_cd_checks_with(sid_c, sid_d):
        assert lex_sched_cd.stmt_instance_before.lex_pt == [sid_c, 'i', 0, 'j', 0]
        assert lex_sched_cd.stmt_instance_after.lex_pt == [sid_d, 't', 0, 0, 0]

        # Get two isl maps representing the LexSchedule

        isl_sched_map_before, isl_sched_map_after = get_isl_maps_for_LexSchedule(
            lex_sched_cd, knl)

        # Create expected maps, align, compare

        isl_sched_map_before_expected = isl.Map(
            "[pi, pj] -> { "
            "[statement = %d, i, j] -> [l0 = %d, l1 = i, l2 = 0, l3 = j, l4 = 0] : "
            "0 <= i < pi and 0 <= j < pj }"
            % (sid_c, sid_c)
            )
        isl_sched_map_before_expected = align_isl_maps_by_var_names(
            isl_sched_map_before_expected, isl_sched_map_before)

        isl_sched_map_after_expected = isl.Map(
            "[pt] -> { "
            "[statement = %d, t] -> [l0 = %d, l1 = t, l2 = 0, l3 = 0, l4 = 0] : "
            "0 <= t < pt }"
            % (sid_d, sid_d)
            )
        isl_sched_map_after_expected = align_isl_maps_by_var_names(
            isl_sched_map_after_expected, isl_sched_map_after)

        assert isl_sched_map_before == isl_sched_map_before_expected
        assert isl_sched_map_after == isl_sched_map_after_expected

    if lex_sched_cd.stmt_instance_before.stmt.int_id == 0:
        perform_insn_cd_checks_with(0, 1)
    else:
        perform_insn_cd_checks_with(1, 0)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker
