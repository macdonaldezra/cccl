# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printer for cuda::std::optional."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType

import cccl_common

import gdb
import gdb.printing


def _is_cuda_optional(value_type: gdb.Type) -> bool:
    value_type = cccl_common.canonical_type(value_type)
    type_name = cccl_common.public_type_name(value_type)
    return cccl_common.template_name(type_name) == "cuda::std::optional"


class OptionalPrinter:
    """Expose the contained value of cuda::std::optional to GDB."""

    def __init__(self, value: gdb.Value) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value
        self.type = cccl_common.canonical_type(value.type)
        self.type_name = cccl_common.public_type_name(self.type)
        self.value_type = self.type.template_argument(0)

    def _contained_value(self) -> gdb.Value | None:
        if self.value_type.code == gdb.TYPE_CODE_REF:
            pointer = self.value["__value_"]
            return None if int(pointer) == 0 else pointer.dereference()
        if not bool(self.value["__engaged_"]):
            return None
        return self.value["__storage_"]["__val_"]

    def children(self) -> Iterator[tuple[str, gdb.Value]]:
        contained = self._contained_value()
        if contained is not None:
            yield "value", contained

    def to_string(self) -> str:
        if self._contained_value() is None:
            return f"{self.type_name} = nullopt"
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
