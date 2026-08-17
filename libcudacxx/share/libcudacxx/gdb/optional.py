# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printer for cuda::std::optional."""

from __future__ import annotations

import re
from collections.abc import Iterator
from types import ModuleType

import cccl_common

import gdb
import gdb.printing

_OPTIONAL_PATTERN = re.compile(r"^cuda::std::optional<.+>$")


def _is_cuda_optional(value_type: gdb.Type) -> bool:
    # Anchored to the complete name so pointers and arrays of optionals stay
    # unclaimed.
    type_name = cccl_common.canonical_type_name(value_type)
    return _OPTIONAL_PATTERN.fullmatch(type_name) is not None


class OptionalPrinter:
    """Expose the contained value of cuda::std::optional to GDB."""

    def __init__(self, value: gdb.Value) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value
        self.type = cccl_common.canonical_type(value.type)
        self.type_name = cccl_common.public_type_name(self.type)
        self.contained = self._contained_value()

    def _contained_value(self) -> gdb.Value | None:
        # The reference specialization stores a pointer in __value_; the other
        # specializations keep an __engaged_ flag next to their storage.
        if any(field.name == "__value_" for field in self.type.fields()):
            pointer = self.value["__value_"]
            return None if int(pointer) == 0 else pointer.dereference()
        if not bool(self.value["__engaged_"]):
            return None
        return self.value["__storage_"]["__val_"]

    def children(self) -> Iterator[tuple[str, gdb.Value]]:
        if self.contained is not None:
            yield "value", self.contained

    def to_string(self) -> str:
        if self.contained is None:
            # Same word as the LLDB printer, framed as a GDB-style annotation.
            return f"{self.type_name} [nullopt]"
        return self.type_name


class OptionalPrinterLookup(gdb.printing.PrettyPrinter):
    """Select cuda::std::optional by its public class name."""

    def __init__(self) -> None:
        super().__init__("cuda::std::optional")

    def __call__(self, value: gdb.Value) -> OptionalPrinter | None:
        if _is_cuda_optional(value.type):
            return OptionalPrinter(value)
        return None


def register(objfile: ModuleType) -> None:
    """Register the cuda::std::optional printer with GDB."""
    gdb.printing.register_pretty_printer(objfile, OptionalPrinterLookup(), replace=True)
