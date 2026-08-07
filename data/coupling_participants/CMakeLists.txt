cmake_minimum_required(VERSION 3.13)

find_package(deal.II 9.0 REQUIRED HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
             $ENV{HOME}/dealii/build $ENV{HOME}/dealii)
deal_ii_initialize_cached_variables()

project(heat_iface_dealii CXX)

add_executable(heat_iface_dealii heat_iface_dealii.cc)
deal_ii_setup_target(heat_iface_dealii)

# ── the preCICE participant, built ONLY if libprecice is actually here ──────
#
# deal.II has no Python API, so a preCICE participant has to be a compiled
# executable that links libprecice.  This target is OPTIONAL on purpose: the
# `couple` participant above must keep building on an install with no preCICE,
# and a hard `find_package(precice)` would break it there.
#
# PRECICE_ROOT defaults to /opt/precice; override with -DPRECICE_ROOT=<path>.
# A hand-rolled `g++ -I<dealii>/include` does NOT work — deal.II's bundled
# headers need the flags deal_ii_setup_target() supplies — so the preCICE
# include/lib are added ON TOP of deal.II's own target setup, not instead.
set(PRECICE_ROOT "/opt/precice" CACHE PATH "preCICE install prefix")
find_library(PRECICE_LIBRARY NAMES precice
             HINTS ${PRECICE_ROOT}/lib ${PRECICE_ROOT}/lib64)
find_path(PRECICE_INCLUDE_DIR precice/precice.hpp
          HINTS ${PRECICE_ROOT}/include)

if(PRECICE_LIBRARY AND PRECICE_INCLUDE_DIR)
  message(STATUS "preCICE found: ${PRECICE_LIBRARY} — building "
                 "precice_heat_dealii")
  add_executable(precice_heat_dealii precice_heat_dealii.cc)
  deal_ii_setup_target(precice_heat_dealii)
  target_include_directories(precice_heat_dealii PRIVATE ${PRECICE_INCLUDE_DIR})
  target_link_libraries(precice_heat_dealii ${PRECICE_LIBRARY})
else()
  message(STATUS "preCICE not found under ${PRECICE_ROOT} — skipping "
                 "precice_heat_dealii (the `couple` participant still builds)")
endif()
