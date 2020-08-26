import numpy as np
from pytools import memoize_method
from loopy.target.c import CTarget, CASTBuilder
from loopy.types import NumpyType


# {{{ function mangler, preamble generator

def c_vecextensions_function_mangler(kernel, name, arg_dtypes):
    return None


def c_vecextensions_preamble_generator(preamble_info):
    # FIXME: typedef int vs4si __attribute__((vector_size(16))), and folks.
    if False:
        yield

# }}}


# {{{ vector types

class vec:  # noqa
    pass


def _create_vector_types():
    field_names = ["x", "y", "z", "w"]

    vec.types = {}
    vec.names_and_dtypes = []
    vec.type_to_scalar_and_count = {}

    counts = [2, 3, 4, 8, 16]

    for base_name, base_type in [
            ('sc', np.int8),
            ('uc', np.uint8),
            ('ss', np.int16),
            ('us', np.uint16),
            ('si', np.int32),
            ('ui', np.uint32),
            ('sl', np.int64),
            ('ul', np.uint64),
            ('f', np.float32),
            ('d', np.float64),
            ]:
        for count in counts:
            name = "v%s%d" % (base_name, count)

            titles = field_names[:count]

            padded_count = count
            if count == 3:
                padded_count = 4

            names = ["s%d" % i for i in range(count)]
            while len(names) < padded_count:
                names.append("padding%d" % (len(names)-count))

            if len(titles) < len(names):
                titles.extend((len(names)-len(titles))*[None])

            try:
                dtype = np.dtype(dict(
                    names=names,
                    formats=[base_type]*padded_count,
                    titles=titles))
            except NotImplementedError:
                try:
                    dtype = np.dtype([((n, title), base_type)
                                      for (n, title) in zip(names, titles)])
                except TypeError:
                    dtype = np.dtype([(n, base_type) for (n, title)
                                      in zip(names, titles)])

            setattr(vec, name, dtype)

            vec.names_and_dtypes.append((name, dtype))

            vec.types[np.dtype(base_type), count] = dtype
            vec.type_to_scalar_and_count[dtype] = np.dtype(base_type), count


_create_vector_types()


def _register_vector_types(dtype_registry):
    for name, dtype in vec.names_and_dtypes:
        dtype_registry.get_or_register_dtype(name, dtype)

# }}}


# {{{ target

class CVectorExtensionsTarget(CTarget):
    """A specialized C-target that represents vectorization through GCC/Clang
    language extensions.
    """

    def get_device_ast_builder(self):
        return CVectorExtensionsASTBuilder(self)

    @memoize_method
    def get_dtype_registry(self):
        from loopy.target.c.compyte.dtypes import (
                DTypeRegistry, fill_registry_with_c99_stdint_types,
                fill_registry_with_c99_complex_types)
        from loopy.target.c import DTypeRegistryWrapper

        result = DTypeRegistry()
        fill_registry_with_c99_stdint_types(result)
        fill_registry_with_c99_complex_types(result)

        _register_vector_types(result)
        return DTypeRegistryWrapper(result)

    def is_vector_dtype(self, dtype):
        return (isinstance(dtype, NumpyType)
                and dtype.numpy_dtype in list(vec.types.values()))

    def vector_dtype(self, base, count):
        return NumpyType(
                vec.types[base.numpy_dtype, count],
                target=self)

# }}}


# {{{ AST builder

class CVectorExtensionsASTBuilder(CASTBuilder):
    # {{{ library

    def function_manglers(self):
        return (
                [
                    c_vecextensions_function_mangler,
                ] +
                super(CVectorExtensionsASTBuilder, self).function_manglers())

    def preamble_generators(self):
        from loopy.library.reduction import reduction_preamble_generator

        return (
                super(CVectorExtensionsASTBuilder, self).preamble_generators() + [
                    c_vecextensions_preamble_generator,
                    reduction_preamble_generator,
                    ])

    # }}}

    def add_vector_access(self, access_expr, index):
        return access_expr[index]

# }}}
