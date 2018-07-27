from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

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

import six
import re

from pytools import ImmutableRecord, memoize_method
from pymbolic.primitives import Variable

from loopy.symbolic import RuleAwareIdentityMapper, ResolvedFunction
from loopy.kernel.function_interface import (
        CallableKernel, ScalarCallable)


class FunctionResolver(RuleAwareIdentityMapper):
    """
    Mapper to convert the  ``function`` attribute of a
    :class:`pymbolic.primitives.Call` known in the kernel as instances of
    :class:`loopy.symbolic.ResolvedFunction`. A function is known in the
    *kernel*, :func:`loopy.kernel.LoopKernel.find_scoped_function_identifier`
    returns an instance of
    :class:`loopy.kernel.function_interface.InKernelCallable`.

    **Example:** If given an expression of the form ``sin(x) + unknown_function(y) +
    log(z)``, then the mapper would return ``ResolvedFunction('sin')(x) +
    unknown_function(y) + ResolvedFunction('log')(z)``.

    :arg rule_mapping_context: An instance of
        :class:`loopy.symbolic.RuleMappingContext`.
    :arg function_ids: A container with instances of :class:`str` indicating
        the function identifiers to look for while scoping functions.
    """
    def __init__(self, rule_mapping_context, kernel, program_callables_info,
            function_resolvers):
        super(FunctionResolver, self).__init__(rule_mapping_context)
        self.kernel = kernel
        self.program_callables_info = program_callables_info
        # FIXME: function_resolvesrs looks like a very bad name change it
        self.function_resolvers = function_resolvers

    def find_resolved_function_from_identifier(self, identifier):
        """
        Returns an instance of
        :class:`loopy.kernel.function_interface.InKernelCallable` if the
        :arg:`identifier` is known to any kernel function scoper, otherwise returns
        *None*.
        """
        # FIXME change docs
        for scoper in self.function_resolvers:
            # fixme: do we really need to given target for the function
            in_knl_callable = scoper(self.kernel.target, identifier)
            if in_knl_callable is not None:
                return in_knl_callable

        return None

    def map_call(self, expr, expn_state):
        from pymbolic.primitives import Call, CallWithKwargs
        from loopy.symbolic import parse_tagged_name

        name, tag = parse_tagged_name(expr.function)
        if name not in self.rule_mapping_context.old_subst_rules:
            new_call_with_kwargs = self.rec(CallWithKwargs(
                function=expr.function, parameters=expr.parameters,
                kw_parameters={}), expn_state)
            return Call(new_call_with_kwargs.function,
                    new_call_with_kwargs.parameters)
        else:
            return self.map_substitution(name, tag, expr.parameters, expn_state)

    def map_call_with_kwargs(self, expr, expn_state):

        if not isinstance(expr.function, ResolvedFunction):

            # search the kernel for the function.
            in_knl_callable = self.find_resolved_function_from_identifier(
                    expr.function.name)

            if in_knl_callable:
                # associate the newly created ResolvedFunction with the
                # resolved in-kernel callable

                self.program_callables_info, new_func_id = (
                        self.program_callables_info.with_callable(expr.function,
                            in_knl_callable, True))
                return type(expr)(
                        ResolvedFunction(new_func_id),
                        tuple(self.rec(child, expn_state)
                            for child in expr.parameters),
                        dict(
                            (key, self.rec(val, expn_state))
                            for key, val in six.iteritems(expr.kw_parameters))
                            )

        # this is an unknown function as of yet, do not modify it
        return super(FunctionResolver, self).map_call_with_kwargs(expr,
                expn_state)

    def map_reduction(self, expr, expn_state):
        self.scoped_functions.update(
                expr.operation.get_scalar_callables(self.kernel))
        return super(FunctionResolver, self).map_reduction(expr, expn_state)


def resolve_callables(name, program_callables_info, function_resolvers):

    kernel = program_callables_info[name].subkernel

    from loopy.symbolic import SubstitutionRuleMappingContext
    rule_mapping_context = SubstitutionRuleMappingContext(
            kernel.substitutions, kernel.get_var_name_generator())

    function_resolver = FunctionResolver(rule_mapping_context, kernel,
            program_callables_info, function_resolvers)

    # scoping fucntions and collecting the scoped functions
    kernel_with_functions_resolved = rule_mapping_context.finish_kernel(
            function_resolver.map_kernel(kernel))
    program_callables_info = function_resolver.program_callables_info

    new_in_knl_callable = program_callables_info[name].copy(
            subkernel=kernel_with_functions_resolved)
    program_callables_info, _ = program_callables_info.with_callable(
            Variable(name), new_in_knl_callable)

    return program_callables_info


# {{{ program definition

class Program(ImmutableRecord):
    def __init__(self,
            root_kernel_name,
            program_callables_info,
            target=None,
            function_resolvers=None):
        assert isinstance(program_callables_info, ProgramCallablesInfo)

        # FIXME: check if all sanity checks have been covered?
        # FIXME: The comments over here may need some attention.
        assert root_kernel_name in program_callables_info

        if target is None:
            target = program_callables_info[root_kernel_name].subkernel.target

        if function_resolvers is None:
            # populate the function scopers from the target and the loopy
            # specific callable scopers

            # at this point only the root kernel can be present in the
            # callables.
            assert len(program_callables_info.resolved_functions) == 1

            from loopy.library.function import loopy_specific_callable_scopers
            function_resolvers = [loopy_specific_callable_scopers] + (
                    target.get_device_ast_builder().function_scopers())

            # new function resolvers have arrived, implies we need to resolve
            # the callables identified by this set of resolvers
            program_callables_info = (
                    program_callables_info.with_edit_callables_mode())

            for name, in_knl_callable in program_callables_info.items():
                if isinstance(in_knl_callable, CallableKernel):
                    # resolve the callables in the subkernel
                    program_callables_info = (
                            resolve_callables(name, program_callables_info,
                                function_resolvers))
                elif isinstance(in_knl_callable, ScalarCallable):
                    pass
                else:
                    raise NotImplementedError("Unknown callable %s." %
                            type(in_knl_callable).__name__)

            program_callables_info = (
                    program_callables_info.with_exit_edit_callables_mode())

        super(Program, self).__init__(
                root_kernel_name=root_kernel_name,
                program_callables_info=program_callables_info,
                target=target,
                function_resolvers=function_resolvers)

        self._program_executor_cache = {}

    @property
    def name(self):
        #FIXME: discuss with @inducer if we use "name" instead of
        # "root_kernel_name"
        return self.root_kernel_name

    @property
    def root_kernel(self):
        return self.program_callables_info[self.root_kernel_name].subkernel

    def with_root_kernel(self, root_kernel):
        new_in_knl_callable = self.program_callables_info[
                self.root_kernel_name].copy(subkernel=root_kernel)
        new_resolved_functions = (
                self.program_callables_info.resolved_functions.copy())
        new_resolved_functions[self.root_kernel_name] = new_in_knl_callable

        return self.copy(
                program_callables_info=self.program_callables_info.copy(
                    resolved_functions=new_resolved_functions))

    @property
    def args(self):
        return self.root_kernel.args[:]

    # {{{ implementation arguments

    @property
    @memoize_method
    def impl_arg_to_arg(self):
        from loopy.kernel.array import ArrayBase

        result = {}

        for arg in self.args:
            if not isinstance(arg, ArrayBase):
                result[arg.name] = arg
                continue

            if arg.shape is None or arg.dim_tags is None:
                result[arg.name] = arg
                continue

            subscripts_and_names = arg.subscripts_and_names()
            if subscripts_and_names is None:
                result[arg.name] = arg
                continue

            for index, sub_arg_name in subscripts_and_names:
                result[sub_arg_name] = arg

        return result

    # }}}

    def __call__(self, *args, **kwargs):
        key = self.target.get_kernel_executor_cache_key(*args, **kwargs)
        try:
            pex = self._program_executor_cache[key]
        except KeyError:
            pex = self.target.get_kernel_executor(self, *args, **kwargs)
            self._program_executor_cache[key] = pex

        return pex(*args, **kwargs)

    def __str__(self):
        # FIXME: make this better
        print(self.program_callables_info.num_times_callables_called)
        return (
                (self.program_callables_info[
                    self.root_kernel_name].subkernel).__str__() +
                '\nResolved Functions: ' +
                (self.program_callables_info.resolved_functions.keys()).__str__() +
                '\n' + 75*'-' + '\n')

# }}}


def next_indexed_function_identifier(function):
    """
    Returns an instance of :class:`str` with the next indexed-name in the
    sequence for the name of *function*.

    *Example:* ``Variable('sin_0')`` will return ``'sin_1'``.

    :arg function: Either an instance of :class:`pymbolic.primitives.Variable`
        or :class:`loopy.reduction.ArgExtOp` or
        :class:`loopy.reduction.SegmentedOp`.
    """
    from loopy.library.reduction import ArgExtOp, SegmentedOp
    if isinstance(function, (ArgExtOp, SegmentedOp)):
        return function.copy()
    elif isinstance(function, str):
        function = Variable(function)

    assert isinstance(function, Variable)
    func_name = re.compile(r"^(?P<alpha>\S+?)_(?P<num>\d+?)$")

    match = func_name.match(function.name)

    if match is None:
        if function.name[-1] == '_':
            return "{old_name}0".format(old_name=function.name)
        else:
            return "{old_name}_0".format(old_name=function.name)

    return "{alpha}_{num}".format(alpha=match.group('alpha'),
            num=int(match.group('num'))+1)


class ResolvedFunctionRenamer(RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context, renaming_dict):
        super(ResolvedFunctionRenamer, self).__init__(
                rule_mapping_context)
        self.renaming_dict = renaming_dict

    def map_resolved_functions(self, expr, expn_state):
        if expr.name in self.renaming_dict:
            return ResolvedFunction(self.renaming_dict[expr.name])
        else:
            return super(ResolvedFunctionRenamer, self).rec(expr, expn_state)


def rename_resolved_functions_in_a_single_kernel(kernel,
        renaming_dict):
    from loopy.symbolic import SubstitutionRuleMappingContext
    rule_mapping_context = SubstitutionRuleMappingContext(
                kernel.substitutions, kernel.get_var_name_generator())
    resolved_function_renamer = ResolvedFunctionRenamer(rule_mapping_context,
            renaming_dict)
    return (
            rule_mapping_context.finish_kernel(
                resolved_function_renamer.map_kernel(kernel)))


# {{{ program callables info

class ProgramCallablesInfo(ImmutableRecord):
    def __init__(self, resolved_functions, num_times_callables_called=None,
            history_of_callable_names=None, is_being_edited=False,
            old_resolved_functions={}, num_times_hit_during_editing={},
            renames_needed_after_editing={}):

        if num_times_callables_called is None:
            num_times_callables_called = dict((func_id, 1) for func_id in
                    resolved_functions)
        if history_of_callable_names is None:
            history_of_callable_names = dict((func_id, [func_id]) for func_id in
                    resolved_functions)

        super(ProgramCallablesInfo, self).__init__(
                resolved_functions=resolved_functions,
                num_times_callables_called=num_times_callables_called,
                history_of_callable_names=history_of_callable_names,
                old_resolved_functions=old_resolved_functions,
                is_being_edited=is_being_edited,
                num_times_hit_during_editing=num_times_hit_during_editing,
                renames_needed_after_editing=renames_needed_after_editing)

    def with_edit_callables_mode(self):
        return self.copy(is_being_edited=True,
                old_resolved_functions=self.resolved_functions.copy(),
                num_times_hit_during_editing=dict((func_id, 0) for func_id in
                    self.resolved_functions))

    def with_callable(self, function, in_kernel_callable,
            resolved_for_the_first_time=False):
        """
        :arg function: An instance of :class:`pymbolic.primitives.Variable` or
            :class:`loopy.library.reduction.ReductionOpFunction`.

        :arg in_kernel_callables: An instance of
            :class:`loopy.InKernelCallable`.

        .. note::

            Assumes that each callable is touched atmost once, the internal
            working of this function fails if that is violated and raises a
            *RuntimeError*.
        """
        # FIXME: add a note about using enter and exit
        assert self.is_being_edited

        from loopy.library.reduction import ArgExtOp, SegmentedOp

        # {{{ sanity checks

        if isinstance(function, str):
            function = Variable(function)

        assert isinstance(function, (Variable, ArgExtOp, SegmentedOp))

        # }}}

        renames_needed_after_editing = self.renames_needed_after_editing.copy()
        num_times_hit_during_editing = self.num_times_hit_during_editing.copy()
        num_times_callables_called = self.num_times_callables_called.copy()

        if not resolved_for_the_first_time:
            num_times_hit_during_editing[function.name] += 1

        if in_kernel_callable in self.resolved_functions.values():
            for func_id, in_knl_callable in self.resolved_functions.items():
                if in_knl_callable == in_kernel_callable:
                    num_times_callables_called[func_id] += 1
                    if not resolved_for_the_first_time:
                        num_times_callables_called[function.name] -= 1
                        if num_times_callables_called[function.name] == 0:
                            renames_needed_after_editing[func_id] = function.name

                    return (
                            self.copy(
                                num_times_hit_during_editing=(
                                    num_times_hit_during_editing),
                                num_times_callables_called=(
                                    num_times_callables_called),
                                renames_needed_after_editing=(
                                    renames_needed_after_editing)),
                            func_id)
        else:

            # FIXME: maybe deal with the history over here?
            # FIXME: once the code logic is running beautify this part.
            # many "ifs" can be avoided
            unique_function_identifier = function.name
            if (resolved_for_the_first_time or
                    self.num_times_callables_called[function.name] > 1):
                while unique_function_identifier in self.resolved_functions:
                    unique_function_identifier = (
                            next_indexed_function_identifier(
                                unique_function_identifier))

            if not resolved_for_the_first_time:
                num_times_callables_called[function.name] -= 1

            num_times_callables_called[unique_function_identifier] = 1

            updated_resolved_functions = self.resolved_functions.copy()
            updated_resolved_functions[unique_function_identifier] = (
                    in_kernel_callable)

            return (
                    self.copy(
                        resolved_functions=updated_resolved_functions,
                        num_times_callables_called=num_times_callables_called,
                        num_times_hit_during_editing=num_times_hit_during_editing,
                        renames_needed_after_editing=renames_needed_after_editing),
                    Variable(unique_function_identifier))

    def with_exit_edit_callables_mode(self):
        assert self.is_being_edited

        num_times_callables_called = {}
        resolved_functions = {}

        for func_id, in_knl_callable in self.resolved_functions.items():
            if isinstance(in_knl_callable, CallableKernel):
                old_subkernel = in_knl_callable.subkernel
                new_subkernel = rename_resolved_functions_in_a_single_kernel(
                        old_subkernel, self.renames_needed_after_editing)
                in_knl_callable = (
                        in_knl_callable.copy(subkernel=new_subkernel))
            elif isinstance(in_knl_callable, ScalarCallable):
                pass
            else:
                raise NotImplementedError("Unknown callable type %s." %
                        type(in_knl_callable).__name__)

            if func_id in self.renames_needed_after_editing:
                new_func_id = self.renames_needed_after_editing[func_id]
                resolved_functions[new_func_id] = (
                        in_knl_callable)
                num_times_callables_called[new_func_id] = (
                        self.num_times_callables_called[func_id])

            else:
                resolved_functions[func_id] = in_knl_callable
                num_times_callables_called[func_id] = (
                        self.num_times_callables_called[func_id])

        return self.copy(
                is_being_edited=False,
                resolved_functions=resolved_functions,
                num_times_callables_called=num_times_callables_called,
                num_times_hit_during_editing={},
                renames_needed_after_editing={})

    def __getitem__(self, item):
        return self.resolved_functions[item]

    def __contains__(self, item):
        return item in self.resolved_functions

    def items(self):
        return self.resolved_functions.items()

# }}}


def make_program_from_kernel(kernel):
    callable_knl = CallableKernel(subkernel=kernel)
    resolved_functions = {kernel.name: callable_knl}
    program_callables_info = ProgramCallablesInfo(resolved_functions)

    program = Program(
            root_kernel_name=kernel.name,
            program_callables_info=program_callables_info)

    return program


# {{{ ingoring this for now

# if False and isinstance(function, (ArgExtOp, SegmentedOp)):
# FIXME: ignoring this casse for now
# FIXME: If a kernel has two flavors of ArgExtOp then they are
# overwritten and hence not supported.(for now).
# updated_resolved_functions = self.scoped_functions.copy()
# updated_resolved_functions[function] = in_kernel_callable
# return self.copy(updated_resolved_functions), function.copy()

# }}}


# vim: foldmethod=marker
