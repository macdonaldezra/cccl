libcudacxx_add_pretty_printer_test(
  NAME optional
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_empty optional.empty value
    --case inspect_int optional.int value
    --case inspect_zero optional.zero value
    --case inspect_false optional.false value
    --case inspect_const optional.const value
    --case inspect_alias optional.alias value
    --case inspect_nontrivial_empty optional.nontrivial.empty value
    --case inspect_nontrivial optional.nontrivial value
    --case inspect_nested_empty optional.nested.empty value
    --case inspect_nested_inner_empty optional.nested.inner_empty value
    --case inspect_nested optional.nested value
    --case inspect_tuple optional.tuple value
    --case inspect_array optional.array value
    --case inspect_empty_array optional.array.empty value
    --case inspect_nested_array optional.array.nested value
    --case inspect_null_pointer optional.pointer.null value
    --case inspect_before_reset optional.update.before_reset value
    --case inspect_after_reset optional.update.after_reset value
    --case inspect_before_emplace optional.update.before_emplace value
    --case inspect_after_emplace optional.update.after_emplace value
    --case inspect_reference_empty optional.reference.empty value
    --case inspect_reference optional.reference value
    --case inspect_reference_before_update optional.reference.before_update value
    --case inspect_reference_after_update optional.reference.after_update value
    # gersemi: on
)
