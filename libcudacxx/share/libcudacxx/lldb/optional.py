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


def _contained_value(value: lldb.SBValue) -> lldb.SBValue:
    value_type = cccl_common.strip_reference(value.GetType()).GetTemplateArgumentType(0)
    if value_type.IsReferenceType():
        pointer = value.GetChildMemberWithName("__value_")
        if not pointer.IsValid() or pointer.GetValueAsUnsigned(0) == 0:
            return lldb.SBValue()
        return pointer.Dereference().Clone("value")

    engaged = value.GetChildMemberWithName("__engaged_")
    if not engaged.IsValid() or engaged.GetValueAsUnsigned(0) == 0:
        return lldb.SBValue()
    storage = value.GetChildMemberWithName("__storage_")
    if not storage.IsValid():
        return lldb.SBValue()
    return storage.GetChildMemberWithName("__val_").Clone("value")


def optional_summary(value: lldb.SBValue, _internal_dict: InternalDict) -> str:
    value = cccl_common.strip_reference_value(value).GetNonSyntheticValue()
    return "" if _contained_value(value).IsValid() else "nullopt"


class OptionalSyntheticProvider:
    """Expose an engaged cuda::std::optional value as one synthetic child."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value.GetNonSyntheticValue()
        self.child = lldb.SBValue()
        self.update()

    def update(self) -> bool:
        self.child = _contained_value(self.value)
        return self.child.IsValid()

    def num_children(self) -> int:
        return int(self.child.IsValid())

    def has_children(self) -> bool:
        return self.child.IsValid()

    def get_type_name(self) -> str:
        return cccl_common.public_type_name(self.value.GetType())

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
