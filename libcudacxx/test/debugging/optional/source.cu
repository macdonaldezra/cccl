// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/array>
#include <cuda/std/optional>
#include <cuda/std/tuple>

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

#define INSPECT(name, type)                      \
  [[gnu::noinline]] void name(const type& value) \
  {                                              \
    KEEP_FOR_DEBUGGER(value);                    \
  }

INSPECT(inspect_empty, cuda::std::optional<int>)
INSPECT(inspect_int, cuda::std::optional<int>)
INSPECT(inspect_zero, cuda::std::optional<int>)
INSPECT(inspect_false, cuda::std::optional<bool>)
INSPECT(inspect_const, cuda::std::optional<const int>)
INSPECT(inspect_alias, optional_alias)
INSPECT(inspect_nontrivial_empty, cuda::std::optional<nontrivial>)
INSPECT(inspect_nontrivial, cuda::std::optional<nontrivial>)
INSPECT(inspect_nested_empty, nested_optional)
INSPECT(inspect_nested_inner_empty, nested_optional)
INSPECT(inspect_nested, nested_optional)
INSPECT(inspect_tuple, tuple_optional)
INSPECT(inspect_array, array_optional)
INSPECT(inspect_empty_array, empty_array_optional)
INSPECT(inspect_nested_array, nested_array_optional)
INSPECT(inspect_null_pointer, cuda::std::optional<int*>)
INSPECT(inspect_before_reset, cuda::std::optional<int>)
INSPECT(inspect_after_reset, cuda::std::optional<int>)
INSPECT(inspect_before_emplace, cuda::std::optional<int>)
INSPECT(inspect_after_emplace, cuda::std::optional<int>)
INSPECT(inspect_reference_empty, cuda::std::optional<int&>)
INSPECT(inspect_reference, cuda::std::optional<int&>)
INSPECT(inspect_reference_before_update, cuda::std::optional<int&>)
INSPECT(inspect_reference_after_update, cuda::std::optional<int&>)

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
  inspect_reference_before_update(reference);
  referenced = -44;
  inspect_reference_after_update(reference);
}
