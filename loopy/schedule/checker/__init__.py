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


# {{{ Create PairwiseScheduleBuilder for statement pair

def get_schedule_for_statement_pair(
        knl,
        linearization_items,
        insn_id_before,
        insn_id_after,
        ):
    """Create a :class:`loopy.schedule.checker.schedule.PairwiseScheduleBuilder`
        representing the order of two statements as a mapping from
        :class:`loopy.schedule.checker.PairwiseScheduleStatementInstanceSet`
        to lexicographic time.

    :arg knl: A :class:`loopy.kernel.LoopKernel` containing the
        linearization items that will be used to create a schedule.

    :arg linearization_items: A list of :class:`loopy.schedule.ScheduleItem`
        (to be renamed to `loopy.schedule.LinearizationItem`) containing
        the two linearization items for which a schedule will be
        created. This list may be a partial linearization for a
        kernel since this function may be used during the linearization
        process.

    :arg insn_id_before: An instruction identifier that is unique within
        a :class:`loopy.kernel.LoopKernel`.

    :arg insn_id_after: An instruction identifier that is unique within
        a :class:`loopy.kernel.LoopKernel`.

    :returns: A :class:`loopy.schedule.checker.schedule.PairwiseScheduleBuilder`
        representing the order of two statements as a mapping from
        :class:`loopy.schedule.checker.PairwiseScheduleStatementInstanceSet`
        to lexicographic time.
    """

    # {{{ Preprocess if not already preprocessed
    from loopy import preprocess_kernel
    preproc_knl = preprocess_kernel(knl)
    # }}}

    # {{{ Find any EnterLoop inames that are tagged as concurrent
    # so that PairwiseScheduleBuilder knows to ignore them
    # (In the future, this shouldn't be necessary because there
    #  won't be any inames with ConcurrentTags in EnterLoop linearization items.
    #  Test which exercises this: test_linearization_checker_with_stroud_bernstein())
    from loopy.schedule.checker.utils import (
        get_concurrent_inames,
        get_EnterLoop_inames,
    )
    conc_inames, _ = get_concurrent_inames(preproc_knl)
    enterloop_inames = get_EnterLoop_inames(linearization_items, preproc_knl)
    conc_loop_inames = conc_inames & enterloop_inames
    if conc_loop_inames:
        from warnings import warn
        warn(
            "get_schedule_for_statement_pair encountered EnterLoop for inames %s "
            "with ConcurrentTag(s) in linearization for kernel %s. "
            "Ignoring these loops." % (conc_loop_inames, preproc_knl.name))
    # }}}

    # {{{ Create PairwiseScheduleBuilder: mapping of {statement instance: lex point}
    # include only instructions involved in this dependency
    from loopy.schedule.checker.schedule import PairwiseScheduleBuilder
    return PairwiseScheduleBuilder(
        linearization_items,
        insn_id_before,
        insn_id_after,
        loops_to_ignore=conc_loop_inames,
        )
    # }}}

# }}}


# {{{ Get isl map pair from PairwiseScheduleBuilder

def get_isl_maps_from_PairwiseScheduleBuilder(sched_builder, knl):
    """Create a pair of :class:`islpy.Map`s representing a
        sub-schedule as two mappings from statement instances to lexicographic
        time, one for the dependee statement and one for the depender.

    :arg sched_builder: A
        :class:`loopy.schedule.checker.schedule.PairwiseScheduleBuilder`
        representing the order of two statements as a mapping from
        :class:`loopy.schedule.checker.PairwiseScheduleStatementInstanceSet`
        to lexicographic time.

    :arg knl: A :class:`loopy.kernel.LoopKernel` containing the
        linearization items that will be used to create a schedule.

    :returns: A two-tuple containing two :class:`islpy.Map`s
        representing the schedule as two mappings
        from statement instances to lexicographic time, one for
        the dependee and one for the depender.
    """

    # {{{ Get iname domains
    dom_before = knl.get_inames_domain(
        knl.id_to_insn[
            sched_builder.stmt_instance_before.stmt.insn_id].within_inames)
    dom_after = knl.get_inames_domain(
        knl.id_to_insn[
            sched_builder.stmt_instance_after.stmt.insn_id].within_inames)
    # }}}

    # {{{ Get isl maps
    return sched_builder.create_isl_maps(dom_before, dom_after)
    # }}}

# }}}
