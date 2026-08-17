# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type and value helpers shared by the CCCL LLDB pretty printers."""

from __future__ import annotations

import re

import lldb

_ABI_NAMESPACE_PATTERN = re.compile(r"::__(?:\d+|version_bump_ver\d+_)(?=::)")
_TUPLE_PATTERN = re.compile(r"^cuda::std::tuple<.*>$")
_TUPLE_LEAF_INDEX_PATTERN = re.compile(r"__tuple_leaf<(\d+),")


def strip_reference(value_type: lldb.SBType) -> lldb.SBType:
    """Return the type behind any reference, typedef, or cv-qualifier.

    GetDereferencedType() is a no-op on a non-reference, so it needs no guard.
    """
    return value_type.GetCanonicalType().GetDereferencedType().GetUnqualifiedType()


def strip_reference_value(value: lldb.SBValue) -> lldb.SBValue:
    """Return the referenced value, or the value itself if it is not a reference."""
    if value.GetType().IsReferenceType():
        return value.Dereference()
    return value


def canonical_type_name(value_type: lldb.SBType) -> str:
    """Return the display name of the type behind any reference or typedef."""
    return strip_reference(value_type).GetDisplayTypeName() or ""


def public_type_name(value_type: lldb.SBType) -> str:
    """Return the complete type name without CUDA ABI inline namespaces.

    GetName() keeps default template arguments that GetDisplayTypeName() hides.
    """
    value_type = strip_reference(value_type)
    type_name = value_type.GetName() or value_type.GetDisplayTypeName() or ""
    return _ABI_NAMESPACE_PATTERN.sub("", type_name)


def is_cuda_tuple_type(value_type: lldb.SBType) -> bool:
    """Return whether the type behind any reference or alias is a cuda::std::tuple."""
    return _TUPLE_PATTERN.fullmatch(canonical_type_name(value_type)) is not None


def tuple_leaf_bases(base_type: lldb.SBType) -> list[tuple[int, lldb.SBTypeMember]]:
    """Return each ``__tuple_leaf`` base class paired with its tuple index.

    Enumerated from the *type*, not a value's children: LLDB's child
    enumeration silently omits zero-size (empty base class optimization)
    subobjects, but the type system still reports them with a correct offset.
    """
    leaves = []
    for i in range(base_type.GetNumberOfDirectBaseClasses()):
        member = base_type.GetDirectBaseClassAtIndex(i)
        match = _TUPLE_LEAF_INDEX_PATTERN.search(member.GetName() or "")
        if match is None:
            continue
        leaves.append((int(match.group(1)), member))
    leaves.sort(key=lambda pair: pair[0])
    return leaves


def _tuple_base_field_type(tuple_type: lldb.SBType) -> lldb.SBType | None:
    """Return the ``__base_`` member's type, or ``None`` (e.g. cuda::std::tuple<>)."""
    for i in range(tuple_type.GetNumberOfFields()):
        field = tuple_type.GetFieldAtIndex(i)
        if field.GetName() == "__base_":
            return field.GetType()
    return None


def tuple_type_name(tuple_type: lldb.SBType) -> str:
    """Reconstruct a tuple's display name, template arguments included.

    GCC's debug info drops a variadic class template's own template
    arguments, collapsing every specialization's name to
    ``cuda::std::tuple<>`` (Clang is unaffected). Each ``__tuple_leaf<N, T>``
    base class still carries its element type ``T`` though, so the argument
    list is rebuilt from those instead.
    """
    canonical = tuple_type.GetCanonicalType().GetUnqualifiedType()
    display_name = canonical.GetDisplayTypeName() or canonical.GetName() or ""
    prefix = display_name.split("<", 1)[0]
    base_type = _tuple_base_field_type(canonical)
    leaves = tuple_leaf_bases(base_type) if base_type is not None else []
    element_names = [
        _tuple_element_type_name(member.GetType().GetTemplateArgumentType(1))
        for _, member in leaves
    ]
    return f"{prefix}<{', '.join(element_names)}>"


def _tuple_element_type_name(element_type: lldb.SBType) -> str:
    """Return a tuple element's display name, recursing into nested tuples.

    A nested tuple element loses its own arguments to the same GCC
    limitation. The unqualified name is only used to test for that case;
    the returned text keeps the element's own qualifiers (e.g. ``const
    char *``), unlike the outer tuple's own top-level qualifier, which is
    dropped everywhere for consistency (see ``is_cuda_tuple_type``).
    """
    recognized_name = (
        element_type.GetCanonicalType().GetUnqualifiedType().GetDisplayTypeName() or ""
    )
    if _TUPLE_PATTERN.fullmatch(recognized_name) is not None:
        return tuple_type_name(element_type)
    return element_type.GetDisplayTypeName() or element_type.GetName() or ""
