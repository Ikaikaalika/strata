cmake_minimum_required(VERSION 3.20)
project(lokahi_engine LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wno-unused-parameter)

set(LOKAHI_CORE_SOURCES
  src/c_api.cpp
  src/gemma3.cpp
  src/json.cpp
  src/model.cpp
  src/quant.cpp
  src/safetensors.cpp
  src/thread_pool.cpp
)

add_library(lokahi_core STATIC ${LOKAHI_CORE_SOURCES})
target_include_directories(lokahi_core
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
find_package(Threads REQUIRED)
target_link_libraries(lokahi_core PUBLIC Threads::Threads)

# Shared library exposing the C ABI (include/lokahi/lokahi.h).
add_library(lokahi SHARED src/c_api_export.cpp)
target_link_libraries(lokahi PRIVATE lokahi_core)
if(APPLE)
  target_link_options(lokahi PRIVATE -Wl,-all_load)
else()
  target_link_options(lokahi PRIVATE -Wl,--whole-archive $<TARGET_FILE:lokahi_core> -Wl,--no-whole-archive)
endif()

add_executable(lokahi_cli tools/lokahi_cli.cpp)
set_target_properties(lokahi_cli PROPERTIES OUTPUT_NAME lokahi)
target_link_libraries(lokahi_cli PRIVATE lokahi_core)

enable_testing()
add_executable(lokahi_engine_tests tests/engine_tests.cpp)
target_link_libraries(lokahi_engine_tests PRIVATE lokahi_core)
add_test(NAME lokahi_engine_tests COMMAND lokahi_engine_tests)
