# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printer for cuda::std::optional."""

from __future__ import annotations

import re

import cccl_common

import lldb

_OPTIONAL_PATTERN = re.compile(r"^cuda::std::optional<.+>$")
InternalDict = dict[str, object]


def is_cuda_optional(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    return (
        _OPTIONAL_PATTERN.fullmatch(cccl_common.canonical_type_name(value_type))
        is not None
    )


def _is_engaged(value: lldb.SBValue) -> bool | None:
    """Return the engaged state, or ``None`` when it cannot be determined.

    The reference specialization stores a pointer in ``__value_``; the other
    specializations keep an ``__engaged_`` flag next to their storage. A
    member that is missing or unreadable (e.g. behind a dangling pointer)
    must not be reported as disengaged.
    """
    pointer = value.GetChildMemberWithName("__value_")
    if pointer.IsValid():
        if pointer.GetError().Fail():
            return None
        return pointer.GetValueAsUnsigned(0) != 0
    engaged = value.GetChildMemberWithName("__engaged_")
    if not engaged.IsValid() or engaged.GetError().Fail():
        return None
    return engaged.GetValueAsUnsigned(0) != 0


def _clone_contained(value: lldb.SBValue) -> lldb.SBValue:
    if not value.IsValid():
        return lldb.SBValue()
    # Casting to the canonical type drops alias spellings like remove_cv_t,
    # which GCC's debug info collapses to one argument-less name that LLDB's
    # name-keyed formatter matching then applies to unrelated values.
    return value.Cast(value.GetType().GetCanonicalType()).Clone("value")


def _contained_value(value: lldb.SBValue) -> lldb.SBValue:
    if not _is_engaged(value):
        return lldb.SBValue()
    pointer = value.GetChildMemberWithName("__value_")
    if pointer.IsValid():
        return _clone_contained(pointer.Dereference())
    storage = value.GetChildMemberWithName("__storage_")
    return _clone_contained(storage.GetChildMemberWithName("__val_"))


def optional_summary(value: lldb.SBValue, _internal_dict: InternalDict) -> str:
    value = cccl_common.strip_reference_value(value).GetNonSyntheticValue()
    engaged = _is_engaged(value)
    if engaged is None:
        return ""
    return "" if engaged else "nullopt"


class OptionalSyntheticProvider:
    """Expose an engaged cuda::std::optional value as one synthetic child."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value.GetNonSyntheticValue()
        self.child = lldb.SBValue()
        self.update()

    def update(self) -> bool:
        self.child = _contained_value(self.value)
        # False tells LLDB to rebuild children after each stop; a cached child
        # count would go stale when the optional becomes engaged on resume.
        return False

    def num_children(self) -> int:
        return int(self.child.IsValid())

    def has_children(self) -> bool:
        return self.child.IsValid()

    def get_type_name(self) -> str:
        value_type = cccl_common.strip_reference(self.value.GetType())
        type_name = cccl_common.public_type_name(value_type)
        payload_type = value_type.GetTemplateArgumentType(0)
        if not cccl_common.is_cuda_tuple_type(payload_type):
            return type_name
        # A tuple payload's name falls to GCC's collapsed variadic spelling
        # (optional<cuda::std::tuple<>>); rebuild it from the payload type.
        prefix = type_name.split("<", 1)[0]
        return f"{prefix}<{cccl_common.tuple_type_name(payload_type)}>"

    def get_child_index(self, name: str) -> int:
        return 0 if name == "value" else -1

    def get_child_at_index(self, index: int) -> lldb.SBValue | None:
        if index == 0 and self.child.IsValid():
            return self.child
        return None


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register the cuda::std::optional formatter in an LLDB category."""
    debugger.HandleCommand(
        f"type summary add --category {category} --expand --python-function {module}.optional_summary "
        f"--recognizer-function {module}.is_cuda_optional"
    )
    debugger.HandleCommand(
        f"type synthetic add --category {category} --python-class {module}.OptionalSyntheticProvider "
        f"--recognizer-function {module}.is_cuda_optional"
    )
