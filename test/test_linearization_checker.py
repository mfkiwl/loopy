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
from loopy.kernel import KernelState
from loopy import (
    preprocess_kernel,
    get_one_linearized_kernel,
)
from loopy.schedule.checker.schedule import (
    LEX_VAR_PREFIX,
    STATEMENT_VAR_NAME,
)

logger = logging.getLogger(__name__)

try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()


# {{{ test PairwiseScheduleBuilder and map creation

def test_pairwise_schedule_and_map_creation():
    import islpy as isl
    from loopy.schedule.checker import (
        get_schedule_for_statement_pair,
    )
    from loopy.schedule.checker.utils import (
        ensure_dim_names_match_and_align,
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

    # Create PairwiseScheduleBuilder: mapping of {statement instance: lex point}
    sched_ab = get_schedule_for_statement_pair(
        knl,
        linearization_items,
        "insn_a",
        "insn_b",
        )
    sched_ac = get_schedule_for_statement_pair(
        knl,
        linearization_items,
        "insn_a",
        "insn_c",
        )
    sched_ad = get_schedule_for_statement_pair(
        knl,
        linearization_items,
        "insn_a",
        "insn_d",
        )
    sched_bc = get_schedule_for_statement_pair(
        knl,
        linearization_items,
        "insn_b",
        "insn_c",
        )
    sched_bd = get_schedule_for_statement_pair(
        knl,
        linearization_items,
        "insn_b",
        "insn_d",
        )
    sched_cd = get_schedule_for_statement_pair(
        knl,
        linearization_items,
        "insn_c",
        "insn_d",
        )

    # There are multiple potential linearization orders for this kernel, so when
    # performing our comparisons for schedule correctness, we need to know which
    # order loopy chose.
    from loopy.schedule import RunInstruction
    linearized_insn_ord = []
    for item in linearization_items:
        if isinstance(item, RunInstruction):
            linearized_insn_ord.append(item.insn_id)

    def _lex_space_string(dim_vals):
        # Return a string describing lex space dimension assignments
        # (used to create maps below)
        return ", ".join(
            ["%s%d=%s" % (LEX_VAR_PREFIX, idx, str(val))
            for idx, val in enumerate(dim_vals)])

    # Relationship between insn_a and insn_b ---------------------------------------

    assert sched_ab.stmt_instance_set_before.lex_points == [0, 'i', 0, 'k', 0]
    assert sched_ab.stmt_instance_set_after.lex_points == [0, 'i', 1, 'j', 0]

    # Get two maps from the PairwiseScheduleBuilder

    sched_map_before, sched_map_after = sched_ab.build_maps(knl)

    # Create expected maps, align, compare

    sched_map_before_expected = isl.Map(
        "[pi, pk] -> { [%s=0, i, k] -> [%s] : 0 <= i < pi and 0 <= k < pk }"
        % (
            STATEMENT_VAR_NAME,
            _lex_space_string(["0", "i", "0", "k", "0"]),
            )
        )
    sched_map_before_expected = ensure_dim_names_match_and_align(
        sched_map_before_expected, sched_map_before)

    sched_map_after_expected = isl.Map(
        "[pi, pj] -> { [%s=1, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
        % (
            STATEMENT_VAR_NAME,
            _lex_space_string(["0", "i", "1", "j", "0"]),
            )
        )
    sched_map_after_expected = ensure_dim_names_match_and_align(
        sched_map_after_expected, sched_map_after)

    assert sched_map_before == sched_map_before_expected
    assert sched_map_after == sched_map_after_expected

    # ------------------------------------------------------------------------------
    # Relationship between insn_a and insn_c ---------------------------------------

    assert sched_ac.stmt_instance_set_before.lex_points == [0, 'i', 0, 'k', 0]
    assert sched_ac.stmt_instance_set_after.lex_points == [0, 'i', 1, 'j', 0]

    # Get two maps from the PairwiseScheduleBuilder

    sched_map_before, sched_map_after = sched_ac.build_maps(knl)

    # Create expected maps, align, compare

    sched_map_before_expected = isl.Map(
        "[pi, pk] -> { [%s=0, i, k] -> [%s] : 0 <= i < pi and 0 <= k < pk }"
        % (
            STATEMENT_VAR_NAME,
            _lex_space_string(["0", "i", "0", "k", "0"]),
            )
        )
    sched_map_before_expected = ensure_dim_names_match_and_align(
        sched_map_before_expected, sched_map_before)

    sched_map_after_expected = isl.Map(
        "[pi, pj] -> { [%s=1, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
        % (
            STATEMENT_VAR_NAME,
            _lex_space_string(["0", "i", "1", "j", "0"]),
            )
        )
    sched_map_after_expected = ensure_dim_names_match_and_align(
        sched_map_after_expected, sched_map_after)

    assert sched_map_before == sched_map_before_expected
    assert sched_map_after == sched_map_after_expected

    # ------------------------------------------------------------------------------
    # Relationship between insn_a and insn_d ---------------------------------------

    # insn_a and insn_d could have been linearized in either order
    # (i loop could be before or after t loop)
    def perform_insn_ad_checks_with(a_lex_idx, d_lex_idx):
        assert sched_ad.stmt_instance_set_before.lex_points == [
            a_lex_idx, 'i', 0, 'k', 0]
        assert sched_ad.stmt_instance_set_after.lex_points == [
            d_lex_idx, 't', 0, 0, 0]

        # Get two maps from the PairwiseScheduleBuilder

        sched_map_before, sched_map_after = sched_ad.build_maps(knl)

        # Create expected maps, align, compare

        sched_map_before_expected = isl.Map(
            "[pi, pk] -> { [%s=0, i, k] -> [%s] : 0 <= i < pi and 0 <= k < pk }"
            % (
                STATEMENT_VAR_NAME,
                _lex_space_string([a_lex_idx, "i", "0", "k", "0"]),
                )
            )
        sched_map_before_expected = ensure_dim_names_match_and_align(
            sched_map_before_expected, sched_map_before)

        sched_map_after_expected = isl.Map(
            "[pt] -> { [%s=1, t] -> [%s] : 0 <= t < pt }"
            % (
                STATEMENT_VAR_NAME,
                _lex_space_string([d_lex_idx, "t", "0", "0", "0"]),
                )
            )
        sched_map_after_expected = ensure_dim_names_match_and_align(
            sched_map_after_expected, sched_map_after)

        assert sched_map_before == sched_map_before_expected
        assert sched_map_after == sched_map_after_expected

    if linearized_insn_ord.index("insn_a") < linearized_insn_ord.index("insn_d"):
        # insn_a was linearized first, check schedule accordingly
        perform_insn_ad_checks_with(0, 1)
    else:
        # insn_d was linearized first, check schedule accordingly
        perform_insn_ad_checks_with(1, 0)

    # ------------------------------------------------------------------------------
    # Relationship between insn_b and insn_c ---------------------------------------

    # insn_b and insn_c could have been linearized in either order
    # (i loop could be before or after t loop)
    def perform_insn_bc_checks_with(b_lex_idx, c_lex_idx):
        assert sched_bc.stmt_instance_set_before.lex_points == [
            0, 'i', 0, 'j', b_lex_idx]
        assert sched_bc.stmt_instance_set_after.lex_points == [
            0, 'i', 0, 'j', c_lex_idx]

        # Get two maps from the PairwiseScheduleBuilder

        sched_map_before, sched_map_after = sched_bc.build_maps(knl)

        # Create expected maps, align, compare

        sched_map_before_expected = isl.Map(
            "[pi, pj] -> { [%s=0, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
            % (
                STATEMENT_VAR_NAME,
                _lex_space_string(["0", "i", "0", "j", b_lex_idx]),
                )
            )
        sched_map_before_expected = ensure_dim_names_match_and_align(
            sched_map_before_expected, sched_map_before)

        sched_map_after_expected = isl.Map(
            "[pi, pj] -> { [%s=1, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
            % (
                STATEMENT_VAR_NAME,
                _lex_space_string(["0", "i", "0", "j", c_lex_idx]),
                )
            )
        sched_map_after_expected = ensure_dim_names_match_and_align(
            sched_map_after_expected, sched_map_after)

        assert sched_map_before == sched_map_before_expected
        assert sched_map_after == sched_map_after_expected

    if linearized_insn_ord.index("insn_b") < linearized_insn_ord.index("insn_c"):
        # insn_b was linearized first, check schedule accordingly
        perform_insn_bc_checks_with(0, 1)
    else:
        # insn_c was linearized first, check schedule accordingly
        perform_insn_bc_checks_with(1, 0)

    # ------------------------------------------------------------------------------
    # Relationship between insn_b and insn_d ---------------------------------------

    # insn_b and insn_d could have been linearized in either order
    # (i loop could be before or after t loop)
    def perform_insn_bd_checks_with(b_lex_idx, d_lex_idx):
        assert sched_bd.stmt_instance_set_before.lex_points == [
            b_lex_idx, 'i', 0, 'j', 0]
        assert sched_bd.stmt_instance_set_after.lex_points == [
            d_lex_idx, 't', 0, 0, 0]

        # Get two maps from the PairwiseScheduleBuilder

        sched_map_before, sched_map_after = sched_bd.build_maps(knl)

        # Create expected maps, align, compare

        sched_map_before_expected = isl.Map(
            "[pi, pj] -> { [%s=0, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
            % (
                STATEMENT_VAR_NAME,
                _lex_space_string([b_lex_idx, "i", "0", "j", "0"]),
                )
            )
        sched_map_before_expected = ensure_dim_names_match_and_align(
            sched_map_before_expected, sched_map_before)

        sched_map_after_expected = isl.Map(
            "[pt] -> { [%s=1, t] -> [%s] : 0 <= t < pt }"
            % (
                STATEMENT_VAR_NAME,
                _lex_space_string([d_lex_idx, "t", "0", "0", "0"]),
                )
            )
        sched_map_after_expected = ensure_dim_names_match_and_align(
            sched_map_after_expected, sched_map_after)

        assert sched_map_before == sched_map_before_expected
        assert sched_map_after == sched_map_after_expected

    if linearized_insn_ord.index("insn_b") < linearized_insn_ord.index("insn_d"):
        # insn_b was linearized first, check schedule accordingly
        perform_insn_bd_checks_with(0, 1)
    else:
        # insn_d was linearized first, check schedule accordingly
        perform_insn_bd_checks_with(1, 0)

    # ------------------------------------------------------------------------------
    # Relationship between insn_c and insn_d ---------------------------------------

    # insn_c and insn_d could have been linearized in either order
    # (i loop could be before or after t loop)
    def perform_insn_cd_checks_with(c_lex_idx, d_lex_idx):
        assert sched_cd.stmt_instance_set_before.lex_points == [
            c_lex_idx, 'i', 0, 'j', 0]
        assert sched_cd.stmt_instance_set_after.lex_points == [
            d_lex_idx, 't', 0, 0, 0]

        # Get two maps from the PairwiseScheduleBuilder

        sched_map_before, sched_map_after = sched_cd.build_maps(knl)

        # Create expected maps, align, compare

        sched_map_before_expected = isl.Map(
            "[pi, pj] -> { [%s=0, i, j] -> [%s] : 0 <= i < pi and 0 <= j < pj }"
            % (
                STATEMENT_VAR_NAME,
                _lex_space_string([c_lex_idx, "i", "0", "j", "0"]),
                )
            )
        sched_map_before_expected = ensure_dim_names_match_and_align(
            sched_map_before_expected, sched_map_before)

        sched_map_after_expected = isl.Map(
            "[pt] -> { [%s=1, t] -> [%s] : 0 <= t < pt }"
            % (
                STATEMENT_VAR_NAME,
                _lex_space_string([d_lex_idx, "t", "0", "0", "0"]),
                )
            )
        sched_map_after_expected = ensure_dim_names_match_and_align(
            sched_map_after_expected, sched_map_after)

        assert sched_map_before == sched_map_before_expected
        assert sched_map_after == sched_map_after_expected

    if linearized_insn_ord.index("insn_c") < linearized_insn_ord.index("insn_d"):
        # insn_c was linearized first, check schedule accordingly
        perform_insn_cd_checks_with(0, 1)
    else:
        # insn_d was linearized first, check schedule accordingly
        perform_insn_cd_checks_with(1, 0)

# }}}


# {{{ test statement instance ordering creation

def test_statement_instance_ordering_creation():
    import islpy as isl
    from loopy.schedule.checker import (
        get_schedule_for_statement_pair,
    )
    from loopy.schedule.checker.utils import (
        ensure_dim_names_match_and_align,
        append_marker_to_isl_map_var_names,
    )
    from loopy.schedule.checker.lexicographic_order_map import (
        get_statement_ordering_map,
    )

    # example kernel (add deps to fix loop order)
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
                c[i,j] = d[i,j]  {id=insn_c,dep=insn_b}
            end
        end
        for t
            e[t] = f[t]  {id=insn_d, dep=insn_c}
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

    def check_sio_for_insn_pair(
            insn_id_before,
            insn_id_after,
            expected_lex_order_map,
            expected_sio,
            ):

        sched_builder = get_schedule_for_statement_pair(
            knl,
            linearization_items,
            insn_id_before,
            insn_id_after,
            )

        # Get two isl maps from the PairwiseScheduleBuilder
        sched_map_before, sched_map_after = sched_builder.build_maps(knl)

        # get map representing lexicographic ordering
        sched_lex_order_map = sched_builder.get_lex_order_map_for_sched_space()

        assert sched_lex_order_map == expected_lex_order_map

        # create statement instance ordering,
        # maps each statement instance to all statement instances occuring later
        sio = get_statement_ordering_map(
            sched_map_before,
            sched_map_after,
            sched_lex_order_map,
            )

        sio_aligned = ensure_dim_names_match_and_align(sio, expected_sio)

        assert sio_aligned == expected_sio

    expected_lex_order_map = isl.Map(
        "{{ "
        "[{0}0', {0}1', {0}2', {0}3', {0}4'] -> [{0}0, {0}1, {0}2, {0}3, {0}4] :"
        "("
        "{0}0' < {0}0 "
        ") or ("
        "{0}0'={0}0 and {0}1' < {0}1 "
        ") or ("
        "{0}0'={0}0 and {0}1'={0}1 and {0}2' < {0}2 "
        ") or ("
        "{0}0'={0}0 and {0}1'={0}1 and {0}2'={0}2 and {0}3' < {0}3 "
        ") or ("
        "{0}0'={0}0 and {0}1'={0}1 and {0}2'={0}2 and {0}3'={0}3 and {0}4' < {0}4"
        ")"
        "}}".format(LEX_VAR_PREFIX))

    # Isl ignores these apostrophes, but test would still pass since it ignores
    # variable names when checking for equality. Even so, explicitly add apostrophes
    # for sanity.
    expected_lex_order_map = append_marker_to_isl_map_var_names(
        expected_lex_order_map, isl.dim_type.in_, "'")

    # Relationship between insn_a and insn_b ---------------------------------------

    expected_sio = isl.Map(
        "[pi, pj, pk] -> {{ "
        "[{0}'=0, i', k'] -> [{0}=1, i, j] : "
        "0 <= i' < pi and 0 <= k' < pk and 0 <= j < pj and 0 <= i < pi and i > i'; "
        "[{0}'=0, i', k'] -> [{0}=1, i=i', j] : "
        "0 <= i' < pi and 0 <= k' < pk and 0 <= j < pj "
        "}}".format(STATEMENT_VAR_NAME)
        )
    # isl ignores these apostrophes, so explicitly add them
    expected_sio = append_marker_to_isl_map_var_names(
        expected_sio, isl.dim_type.in_, "'")

    check_sio_for_insn_pair(
        "insn_a", "insn_b", expected_lex_order_map, expected_sio)

    # Relationship between insn_a and insn_c ---------------------------------------

    expected_sio = isl.Map(
        "[pi, pj, pk] -> {{ "
        "[{0}'=0, i', k'] -> [{0}=1, i, j] : "
        "0 <= i' < pi and 0 <= k' < pk and 0 <= j < pj and 0 <= i < pi and i > i'; "
        "[{0}'=0, i', k'] -> [{0}=1, i=i', j] : "
        "0 <= i' < pi and 0 <= k' < pk and 0 <= j < pj "
        "}}".format(STATEMENT_VAR_NAME)
        )
    # isl ignores these apostrophes, so explicitly add them
    expected_sio = append_marker_to_isl_map_var_names(
        expected_sio, isl.dim_type.in_, "'")

    check_sio_for_insn_pair(
        "insn_a", "insn_c", expected_lex_order_map, expected_sio)

    # Relationship between insn_a and insn_d ---------------------------------------

    expected_sio = isl.Map(
        "[pt, pi, pk] -> {{ "
        "[{0}'=0, i', k'] -> [{0}=1, t] : "
        "0 <= i' < pi and 0 <= k' < pk and 0 <= t < pt "
        "}}".format(STATEMENT_VAR_NAME)
        )
    # isl ignores these apostrophes, so explicitly add them
    expected_sio = append_marker_to_isl_map_var_names(
        expected_sio, isl.dim_type.in_, "'")

    check_sio_for_insn_pair(
        "insn_a", "insn_d", expected_lex_order_map, expected_sio)

    # Relationship between insn_b and insn_c ---------------------------------------

    expected_sio = isl.Map(
        "[pi, pj] -> {{ "
        "[{0}'=0, i', j'] -> [{0}=1, i, j] : "
        "0 <= i' < pi and 0 <= j' < pj and i > i' and 0 <= i < pi and 0 <= j < pj; "
        "[{0}'=0, i', j'] -> [{0}=1, i=i', j] : "
        "0 <= i' < pi and 0 <= j' < pj and j > j' and 0 <= j < pj; "
        "[{0}'=0, i', j'] -> [{0}=1, i=i', j=j'] : "
        "0 <= i' < pi and 0 <= j' < pj "
        "}}".format(STATEMENT_VAR_NAME)
        )
    # isl ignores these apostrophes, so explicitly add them
    expected_sio = append_marker_to_isl_map_var_names(
        expected_sio, isl.dim_type.in_, "'")

    check_sio_for_insn_pair(
        "insn_b", "insn_c", expected_lex_order_map, expected_sio)

    # Relationship between insn_b and insn_d ---------------------------------------

    expected_sio = isl.Map(
        "[pt, pi, pj] -> {{ "
        "[{0}'=0, i', j'] -> [{0}=1, t] : "
        "0 <= i' < pi and 0 <= j' < pj and 0 <= t < pt "
        "}}".format(STATEMENT_VAR_NAME)
        )
    # isl ignores these apostrophes, so explicitly add them
    expected_sio = append_marker_to_isl_map_var_names(
        expected_sio, isl.dim_type.in_, "'")

    check_sio_for_insn_pair(
        "insn_b", "insn_d", expected_lex_order_map, expected_sio)

    # Relationship between insn_c and insn_d ---------------------------------------

    expected_sio = isl.Map(
        "[pt, pi, pj] -> {{ "
        "[{0}'=0, i', j'] -> [{0}=1, t] : "
        "0 <= i' < pi and 0 <= j' < pj and 0 <= t < pt "
        "}}".format(STATEMENT_VAR_NAME)
        )
    # isl ignores these apostrophes, so explicitly add them
    expected_sio = append_marker_to_isl_map_var_names(
        expected_sio, isl.dim_type.in_, "'")

    check_sio_for_insn_pair(
        "insn_c", "insn_d", expected_lex_order_map, expected_sio)

# }}}


def test_linearization_checker_with_loop_prioritization():
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

    unprocessed_knl = knl.copy()

    deps = lp.create_dependencies_from_legacy_knl(unprocessed_knl)
    if hasattr(lp, "add_dependencies_v2"):
        # TODO update this after dep refactoring
        knl = lp.add_dependencies_v2(  # pylint:disable=no-member
            knl, deps)

    # get a linearization to check
    if knl.state < KernelState.PREPROCESSED:
        knl = preprocess_kernel(knl)
    knl = get_one_linearized_kernel(knl)
    linearization_items = knl.linearization

    linearization_is_valid = lp.check_linearization_validity(
        unprocessed_knl, deps, linearization_items)
    assert linearization_is_valid


def test_linearization_checker_with_matmul():
    bsize = 16
    knl = lp.make_kernel(
            "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<ell}",
            [
                "c[i, j] = sum(k, a[i, k]*b[k, j])"
            ],
            name="matmul",
            assumptions="n,m,ell >= 1",
            lang_version=(2018, 2),
            )
    knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32, b=np.float32))
    knl = lp.split_iname(knl, "i", bsize, outer_tag="g.0", inner_tag="l.1")
    knl = lp.split_iname(knl, "j", bsize, outer_tag="g.1", inner_tag="l.0")
    knl = lp.split_iname(knl, "k", bsize)
    knl = lp.add_prefetch(knl, "a", ["k_inner", "i_inner"], default_tag="l.auto")
    knl = lp.add_prefetch(knl, "b", ["j_inner", "k_inner"], default_tag="l.auto")
    knl = lp.prioritize_loops(knl, "k_outer,k_inner")

    unprocessed_knl = knl.copy()

    deps = lp.create_dependencies_from_legacy_knl(unprocessed_knl)
    if hasattr(lp, "add_dependencies_v2"):
        # TODO update this after dep refactoring
        knl = lp.add_dependencies_v2(  # pylint:disable=no-member
            knl, deps)

    # get a linearization to check
    if knl.state < KernelState.PREPROCESSED:
        knl = preprocess_kernel(knl)
    knl = get_one_linearized_kernel(knl)
    linearization_items = knl.linearization

    linearization_is_valid = lp.check_linearization_validity(
        unprocessed_knl, deps, linearization_items)
    assert linearization_is_valid


def test_linearization_checker_with_scan():
    stride = 1
    n_scan = 16
    knl = lp.make_kernel(
        "[n] -> {[i,j]: 0<=i<n and 0<=j<=%d*i}" % stride,
        """
        a[i] = sum(j, j**2)
        """,
        name="scan",
        lang_version=(2018, 2),
        )

    knl = lp.fix_parameters(knl, n=n_scan)
    knl = lp.realize_reduction(knl, force_scan=True)


def test_linearization_checker_with_dependent_domain():
    knl = lp.make_kernel(
        [
            "[n] -> {[i]: 0<=i<n}",
            "{[j]: 0<=j<=2*i}"
        ],
        """
        a[i] = sum(j, j**2) {id=scan}
        """,
        name="dependent_domain",
        lang_version=(2018, 2),
        )
    # TODO current check for unused inames is incorrectly
    # causing linearizing to fail when realize_reduction is used
    #knl = lp.realize_reduction(knl, force_scan=True)

    unprocessed_knl = knl.copy()

    deps = lp.create_dependencies_from_legacy_knl(unprocessed_knl)
    if hasattr(lp, "add_dependencies_v2"):
        # TODO update this after dep refactoring
        knl = lp.add_dependencies_v2(  # pylint:disable=no-member
            knl, deps)

    # get a linearization to check
    if knl.state < KernelState.PREPROCESSED:
        knl = preprocess_kernel(knl)
    knl = get_one_linearized_kernel(knl)
    linearization_items = knl.linearization

    linearization_is_valid = lp.check_linearization_validity(
        unprocessed_knl, deps, linearization_items)
    assert linearization_is_valid


def test_linearization_checker_with_stroud_bernstein():
    knl = lp.make_kernel(
            "{[el, i2, alpha1,alpha2]: \
                    0 <= el < nels and \
                    0 <= i2 < nqp1d and \
                    0 <= alpha1 <= deg and 0 <= alpha2 <= deg-alpha1 }",
            """
            for el,i2
                <> xi = qpts[1, i2]
                <> s = 1-xi
                <> r = xi/s
                <> aind = 0 {id=aind_init}
                for alpha1
                    <> w = s**(deg-alpha1) {id=init_w}
                    for alpha2
                        tmp[el,alpha1,i2] = tmp[el,alpha1,i2] + w * coeffs[aind] \
                                {id=write_tmp,dep=init_w:aind_init}
                        w = w * r * ( deg - alpha1 - alpha2 ) / (1 + alpha2) \
                                {id=update_w,dep=init_w:write_tmp}
                        aind = aind + 1 \
                                {id=aind_incr,dep=aind_init:write_tmp:update_w}
                    end
                end
            end
            """,
            [lp.GlobalArg("coeffs", None, shape=None), "..."],
            name="stroud_bernstein_orig", assumptions="deg>=0 and nels>=1")
    knl = lp.add_and_infer_dtypes(knl,
        dict(coeffs=np.float32, qpts=np.int32))
    knl = lp.fix_parameters(knl, nqp1d=7, deg=4)
    knl = lp.split_iname(knl, "el", 16, inner_tag="l.0")
    knl = lp.split_iname(knl, "el_outer", 2, outer_tag="g.0",
        inner_tag="ilp", slabs=(0, 1))
    knl = lp.tag_inames(knl, dict(i2="l.1", alpha1="unr", alpha2="unr"))

    unprocessed_knl = knl.copy()

    deps = lp.create_dependencies_from_legacy_knl(unprocessed_knl)
    if hasattr(lp, "add_dependencies_v2"):
        # TODO update this after dep refactoring
        knl = lp.add_dependencies_v2(  # pylint:disable=no-member
            knl, deps)

    # get a linearization to check
    if knl.state < KernelState.PREPROCESSED:
        knl = preprocess_kernel(knl)
    knl = get_one_linearized_kernel(knl)
    linearization_items = knl.linearization

    linearization_is_valid = lp.check_linearization_validity(
        unprocessed_knl, deps, linearization_items)
    assert linearization_is_valid


def test_linearization_checker_with_nop():
    knl = lp.make_kernel(
        [
            "{[b]: b_start<=b<b_end}",
            "{[c]: c_start<=c<c_end}",
        ],
        """
         for b
          <> c_end = 2
          for c
           ... nop
          end
         end
        """,
        "...",
        seq_dependencies=True)
    knl = lp.fix_parameters(knl, dim=3)

    unprocessed_knl = knl.copy()

    deps = lp.create_dependencies_from_legacy_knl(unprocessed_knl)
    if hasattr(lp, "add_dependencies_v2"):
        # TODO update this after dep refactoring
        knl = lp.add_dependencies_v2(  # pylint:disable=no-member
            knl, deps)

    # get a linearization to check
    if knl.state < KernelState.PREPROCESSED:
        knl = preprocess_kernel(knl)
    knl = get_one_linearized_kernel(knl)
    linearization_items = knl.linearization

    linearization_is_valid = lp.check_linearization_validity(
        unprocessed_knl, deps, linearization_items)
    assert linearization_is_valid


def test_linearization_checker_with_multi_domain():
    knl = lp.make_kernel(
        [
            "{[i]: 0<=i<ni}",
            "{[j]: 0<=j<nj}",
            "{[k]: 0<=k<nk}",
            "{[x,xx]: 0<=x,xx<nx}",
        ],
        """
        for x,xx
          for i
            <>acc = 0 {id=insn0}
            for j
              for k
                acc = acc + j + k {id=insn1,dep=insn0}
              end
            end
          end
        end
        """,
        name="nest_multi_dom",
        assumptions="ni,nj,nk,nx >= 1",
        lang_version=(2018, 2)
        )
    knl = lp.prioritize_loops(knl, "x,xx,i")
    knl = lp.prioritize_loops(knl, "i,j")
    knl = lp.prioritize_loops(knl, "j,k")

    unprocessed_knl = knl.copy()

    deps = lp.create_dependencies_from_legacy_knl(unprocessed_knl)
    if hasattr(lp, "add_dependencies_v2"):
        # TODO update this after dep refactoring
        knl = lp.add_dependencies_v2(  # pylint:disable=no-member
            knl, deps)

    # get a linearization to check
    if knl.state < KernelState.PREPROCESSED:
        knl = preprocess_kernel(knl)
    knl = get_one_linearized_kernel(knl)
    linearization_items = knl.linearization

    linearization_is_valid = lp.check_linearization_validity(
        unprocessed_knl, deps, linearization_items)
    assert linearization_is_valid


def test_linearization_checker_with_loop_carried_deps():
    knl = lp.make_kernel(
        "{[i]: 0<=i<n}",
        """
        <>acc0 = 0 {id=insn0}
        for i
          acc0 = acc0 + i {id=insn1,dep=insn0}
          <>acc2 = acc0 + i {id=insn2,dep=insn1}
          <>acc3 = acc2 + i {id=insn3,dep=insn2}
          <>acc4 = acc0 + i {id=insn4,dep=insn1}
        end
        """,
        name="loop_carried_deps",
        assumptions="n >= 1",
        lang_version=(2018, 2)
        )

    unprocessed_knl = knl.copy()

    deps = lp.create_dependencies_from_legacy_knl(unprocessed_knl)
    if hasattr(lp, "add_dependencies_v2"):
        # TODO update this after dep refactoring
        knl = lp.add_dependencies_v2(  # pylint:disable=no-member
            knl, deps)

    # get a linearization to check
    if knl.state < KernelState.PREPROCESSED:
        knl = preprocess_kernel(knl)
    knl = get_one_linearized_kernel(knl)
    linearization_items = knl.linearization

    linearization_is_valid = lp.check_linearization_validity(
        unprocessed_knl, deps, linearization_items)
    assert linearization_is_valid


def test_linearization_checker_and_invalid_prioritiy_detection():
    ref_knl = lp.make_kernel(
        [
            "{[h]: 0<=h<nh}",
            "{[i]: 0<=i<ni}",
            "{[j]: 0<=j<nj}",
            "{[k]: 0<=k<nk}",
        ],
        """
        <> acc = 0
        for h,i,j,k
              acc = acc + h + i + j + k
        end
        """,
        name="priorities",
        assumptions="ni,nj,nk,nh >= 1",
        lang_version=(2018, 2)
        )

    # no error:
    knl0 = lp.prioritize_loops(ref_knl, "h,i")
    knl0 = lp.prioritize_loops(ref_knl, "i,j")
    knl0 = lp.prioritize_loops(knl0, "j,k")

    unprocessed_knl = knl0.copy()

    deps = lp.create_dependencies_from_legacy_knl(unprocessed_knl)
    if hasattr(lp, "add_dependencies_v2"):
        # TODO update this after dep refactoring
        knl0 = lp.add_dependencies_v2(  # pylint:disable=no-member
            knl0, deps)

    # get a linearization to check
    if knl0.state < KernelState.PREPROCESSED:
        knl0 = preprocess_kernel(knl0)
    knl0 = get_one_linearized_kernel(knl0)
    linearization_items = knl0.linearization

    linearization_is_valid = lp.check_linearization_validity(
        unprocessed_knl, deps, linearization_items)
    assert linearization_is_valid

    # no error:
    knl1 = lp.prioritize_loops(ref_knl, "h,i,k")
    knl1 = lp.prioritize_loops(knl1, "h,j,k")

    unprocessed_knl = knl1.copy()

    deps = lp.create_dependencies_from_legacy_knl(unprocessed_knl)
    if hasattr(lp, "add_dependencies_v2"):
        # TODO update this after dep refactoring
        knl1 = lp.add_dependencies_v2(  # pylint:disable=no-member
            knl1, deps)

    # get a linearization to check
    if knl1.state < KernelState.PREPROCESSED:
        knl1 = preprocess_kernel(knl1)
    knl1 = get_one_linearized_kernel(knl1)
    linearization_items = knl1.linearization

    linearization_is_valid = lp.check_linearization_validity(
        unprocessed_knl, deps, linearization_items)
    assert linearization_is_valid

    # error (cycle):
    knl2 = lp.prioritize_loops(ref_knl, "h,i,j")
    knl2 = lp.prioritize_loops(knl2, "j,k")
    # TODO think about when legacy deps should be updated based on prio changes

    try:
        if hasattr(lp, "constrain_loop_nesting"):
            knl2 = lp.constrain_loop_nesting(knl2, "k,i")  # pylint:disable=no-member

            # legacy deps depend on priorities, so update deps using new knl
            deps = lp.create_dependencies_from_legacy_knl(knl2)
            if hasattr(lp, "add_dependencies_v2"):
                # TODO update this after dep refactoring
                knl2 = lp.add_dependencies_v2(  # pylint:disable=no-member
                    knl2, deps)
        else:
            knl2 = lp.prioritize_loops(knl2, "k,i")

            # legacy deps depend on priorities, so update deps using new knl
            deps = lp.create_dependencies_from_legacy_knl(knl2)
            if hasattr(lp, "add_dependencies_v2"):
                # TODO update this after dep refactoring
                knl2 = lp.add_dependencies_v2(  # pylint:disable=no-member
                    knl2, deps)

            unprocessed_knl = knl2.copy()

            # get a linearization to check
            if knl2.state < KernelState.PREPROCESSED:
                knl2 = preprocess_kernel(knl2)
            knl2 = get_one_linearized_kernel(knl2)
            linearization_items = knl2.linearization

            linearization_is_valid = lp.check_linearization_validity(
                unprocessed_knl, deps, linearization_items)
        # should raise error
        assert False
    except ValueError as e:
        if hasattr(lp, "constrain_loop_nesting"):
            assert "cycle detected" in str(e)
        else:
            assert "invalid priorities" in str(e)

    # error (inconsistent priorities):
    knl3 = lp.prioritize_loops(ref_knl, "h,i,j,k")
    # TODO think about when legacy deps should be updated based on prio changes
    try:
        if hasattr(lp, "constrain_loop_nesting"):
            knl3 = lp.constrain_loop_nesting(  # pylint:disable=no-member
                knl3, "h,j,i,k")

            # legacy deps depend on priorities, so update deps using new knl
            deps = lp.create_dependencies_from_legacy_knl(knl3)
            if hasattr(lp, "add_dependencies_v2"):
                # TODO update this after dep refactoring
                knl3 = lp.add_dependencies_v2(  # pylint:disable=no-member
                    knl3, deps)
        else:
            knl3 = lp.prioritize_loops(knl3, "h,j,i,k")

            # legacy deps depend on priorities, so update deps using new knl
            deps = lp.create_dependencies_from_legacy_knl(knl3)
            if hasattr(lp, "add_dependencies_v2"):
                # TODO update this after dep refactoring
                knl3 = lp.add_dependencies_v2(  # pylint:disable=no-member
                    knl3, deps)

            unprocessed_knl = knl3.copy()

            # get a linearization to check
            if knl3.state < KernelState.PREPROCESSED:
                knl3 = preprocess_kernel(knl3)
            knl3 = get_one_linearized_kernel(knl3)
            linearization_items = knl3.linearization

            linearization_is_valid = lp.check_linearization_validity(
                unprocessed_knl, deps, linearization_items)
        # should raise error
        assert False
    except ValueError as e:
        if hasattr(lp, "constrain_loop_nesting"):
            assert "cycle detected" in str(e)
        else:
            assert "invalid priorities" in str(e)

# TODO create more kernels with invalid linearizations to test linearization checker


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker