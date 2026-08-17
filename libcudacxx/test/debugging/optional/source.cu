// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/array>
#include <cuda/std/optional>
#include <cuda/std/tuple>

// Give the inspected parameter a stack location that survives optimization, so the
// debugger can read it in this frame. Without this the parameter stays in a
// caller-clobbered register and reads as unavailable at -O3.
#define KEEP_FOR_DEBUGGER(value) asm volatile("" : : "g"(&(value)) : "memory")

struct nontrivial
{
  int value;

  nontrivial(int value)
      : value(value)
  {}

  ~nontrivial() {}
};

using optional_alias        = cuda::std::optional<int>;
using nested_optional       = cuda::std::optional<cuda::std::optional<int>>;
using tuple_optional        = cuda::std::optional<cuda::std::tuple<int, int>>;
using array_optional        = cuda::std::optional<cuda::std::array<int, 3>>;
using empty_array_optional  = cuda::std::optional<cuda::std::array<int, 0>>;
using nested_array_optional = cuda::std::optional<cuda::std::array<cuda::std::array<int, 2>, 2>>;

[[gnu::noinline]] void inspect_empty(const cuda::std::optional<int>& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_int(const cuda::std::optional<int>& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_zero(const cuda::std::optional<int>& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_false(const cuda::std::optional<bool>& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_const(const cuda::std::optional<const int>& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_alias(const optional_alias& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_nontrivial_empty(const cuda::std::optional<nontrivial>& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_nontrivial(const cuda::std::optional<nontrivial>& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_nested_empty(const nested_optional& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_nested_inner_empty(const nested_optional& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_nested(const nested_optional& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_tuple(const tuple_optional& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_array(const array_optional& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_empty_array(const empty_array_optional& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_nested_array(const nested_array_optional& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_null_pointer(const cuda::std::optional<int*>& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_before_reset(const cuda::std::optional<int>& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_after_reset(const cuda::std::optional<int>& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_before_emplace(const cuda::std::optional<int>& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_after_emplace(const cuda::std::optional<int>& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_reference_empty(const cuda::std::optional<int&>& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_reference(const cuda::std::optional<int&>& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_reference_before_update(const cuda::std::optional<int&>& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_reference_after_update(const cuda::std::optional<int&>& value)
{
  KEEP_FOR_DEBUGGER(value);
}

int main()
{
  const cuda::std::optional<int> empty{};
  const cuda::std::optional<int> integer{42};
  const cuda::std::optional<int> zero{0};
  const cuda::std::optional<bool> false_value{false};
  const cuda::std::optional<const int> const_value{17};
  const optional_alias alias{-9};
  const cuda::std::optional<nontrivial> nontrivial_empty{};
  const cuda::std::optional<nontrivial> nontrivial_value{cuda::std::in_place, 73};
  const nested_optional nested_empty{};
  const nested_optional nested_inner_empty{cuda::std::in_place};
  const nested_optional nested{cuda::std::in_place, 31};
  const cuda::std::optional<cuda::std::tuple<int, int>> tuple{cuda::std::in_place, 4, -6};
  const array_optional array{cuda::std::in_place, cuda::std::array<int, 3>{-3, 0, 27}};
  const empty_array_optional empty_array{cuda::std::in_place};
  const nested_array_optional nested_array{
    cuda::std::in_place, cuda::std::array<cuda::std::array<int, 2>, 2>{{{{1, -2}}, {{0, 45}}}}};
  const cuda::std::optional<int*> null_pointer{nullptr};

  inspect_empty(empty);
  inspect_int(integer);
  inspect_zero(zero);
  inspect_false(false_value);
  inspect_const(const_value);
  inspect_alias(alias);
  inspect_nontrivial_empty(nontrivial_empty);
  inspect_nontrivial(nontrivial_value);
  inspect_nested_empty(nested_empty);
  inspect_nested_inner_empty(nested_inner_empty);
  inspect_nested(nested);
  inspect_tuple(tuple);
  inspect_array(array);
  inspect_empty_array(empty_array);
  inspect_nested_array(nested_array);
  inspect_null_pointer(null_pointer);

  cuda::std::optional<int> reset{8};
  inspect_before_reset(reset);
  reset.reset();
  inspect_after_reset(reset);

  cuda::std::optional<int> emplaced;
  inspect_before_emplace(emplaced);
  emplaced.emplace(95);
  inspect_after_emplace(emplaced);

  int referenced = 12;
  const cuda::std::optional<int&> reference_empty{};
  const cuda::std::optional<int&> reference{referenced};
  inspect_reference_empty(reference_empty);
  inspect_reference(reference);

  int updated = 7;
  const cuda::std::optional<int&> reference_updated{updated};
  inspect_reference_before_update(reference_updated);
  updated = -44;
  inspect_reference_after_update(reference_updated);
}
