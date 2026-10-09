# Writes INPUT as a C++ raw string literal named SYMBOL into OUTPUT.
file(READ "${INPUT}" CONTENT)
string(FIND "${CONTENT}" ")lokahi\"" COLLISION)
if(NOT COLLISION EQUAL -1)
  message(FATAL_ERROR "${INPUT} contains the raw-string delimiter")
endif()
get_filename_component(OUTPUT_DIR "${OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${OUTPUT_DIR}")
file(WRITE "${OUTPUT}" "// Generated from ${INPUT}; do not edit.\n#pragma once\nstatic const char ${SYMBOL}[] = R\"lokahi(${CONTENT})lokahi\";\n")
