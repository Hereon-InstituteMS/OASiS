cmake_minimum_required(VERSION 3.13)

find_package(deal.II 9.0 REQUIRED HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
             $ENV{HOME}/dealii/build $ENV{HOME}/dealii)
deal_ii_initialize_cached_variables()

project(heat_iface_dealii CXX)

add_executable(heat_iface_dealii heat_iface_dealii.cc)
deal_ii_setup_target(heat_iface_dealii)
