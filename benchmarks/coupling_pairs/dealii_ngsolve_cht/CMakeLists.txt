cmake_minimum_required(VERSION 3.13)

find_package(deal.II 9.0 REQUIRED HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR})
deal_ii_initialize_cached_variables()

project(heat_slab_dealii CXX)

add_executable(heat_slab_dealii heat_slab_dealii.cc)
deal_ii_setup_target(heat_slab_dealii)
