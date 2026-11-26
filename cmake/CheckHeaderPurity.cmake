# Verifies that all public keel headers in KEEL_INCLUDE_DIR (except *interop_std.h):
# 1. Only contain `#include <cstddef>`, `#include <cstdint>`, `#include <initializer_list>`, or `#include <new>` as angle-bracket includes.
# 2. When compiled standalone with `/showIncludes` (MSVC) or `-H` (Clang/GCC), do not transitively include
#    any system header outside `<cstddef> + <cstdint> + <initializer_list> + <new>` (and upstream KEEL_DEP_INC_DIRS).

if(NOT DEFINED KEEL_INCLUDE_DIR)
    message(FATAL_ERROR "KEEL_INCLUDE_DIR must be passed to CheckHeaderPurity.cmake")
endif()

file(GLOB KEEL_PUBLIC_HEADERS "${KEEL_INCLUDE_DIR}/keel/*.h")
list(FILTER KEEL_PUBLIC_HEADERS EXCLUDE REGEX "interop_std\\.h$")

set(ALLOWED_STD_HEADERS "cstddef" "cstdint" "initializer_list" "new")

foreach(HDR IN LISTS KEEL_PUBLIC_HEADERS)
    get_filename_component(HDR_NAME "${HDR}" NAME)
    file(STRINGS "${HDR}" INCLUDE_LINES REGEX "^[ \t]*#[ \t]*include[ \t]*<[^>]+>")
    foreach(LINE IN LISTS INCLUDE_LINES)
        string(REGEX REPLACE "^[ \t]*#[ \t]*include[ \t]*<([^>]+)>.*$" "\\1" INC_NAME "${LINE}")
        if(NOT INC_NAME IN_LIST ALLOWED_STD_HEADERS)
            message(FATAL_ERROR "Header purity violation in ${HDR_NAME}: forbidden '#include <${INC_NAME}>'. Only <cstddef>, <cstdint>, <initializer_list>, <new> are allowed.")
        endif()
    endforeach()
endforeach()

if(DEFINED COMPILER_PATH AND EXISTS "${COMPILER_PATH}")
    set(SCRATCH_DIR "${CMAKE_BINARY_DIR}/header_purity_check")
    file(MAKE_DIRECTORY "${SCRATCH_DIR}")

    set(USE_MSVC_CLI FALSE)
    if(COMPILER_ID MATCHES "MSVC" OR COMPILER_PATH MATCHES "clang-cl(\\.exe)?$")
        set(USE_MSVC_CLI TRUE)
    endif()

    set(SYS_INC_FLAGS "")
    if(DEFINED STD_INC_DIRS AND NOT STD_INC_DIRS STREQUAL "")
        string(REPLACE "|" ";" STD_INC_LIST "${STD_INC_DIRS}")
        foreach(SDIR IN LISTS STD_INC_LIST)
            if(USE_MSVC_CLI)
                list(APPEND SYS_INC_FLAGS "/I${SDIR}")
            else()
                list(APPEND SYS_INC_FLAGS "-isystem" "${SDIR}")
            endif()
        endforeach()
    endif()

    set(DEP_INC_FLAGS "")
    set(ALLOWED_KEEL_PREFIXES "")
    file(TO_CMAKE_PATH "${KEEL_INCLUDE_DIR}" KEEL_INC_NORM)
    if(USE_MSVC_CLI)
        string(TOLOWER "${KEEL_INC_NORM}" KEEL_INC_NORM)
        list(APPEND DEP_INC_FLAGS "/I${KEEL_INCLUDE_DIR}")
    else()
        list(APPEND DEP_INC_FLAGS "-I${KEEL_INCLUDE_DIR}")
    endif()
    list(APPEND ALLOWED_KEEL_PREFIXES "${KEEL_INC_NORM}")

    if(DEFINED KEEL_DEP_INC_DIRS AND NOT KEEL_DEP_INC_DIRS STREQUAL "")
        string(REPLACE "|" ";" DEP_LIST "${KEEL_DEP_INC_DIRS}")
        foreach(DDIR IN LISTS DEP_LIST)
            file(TO_CMAKE_PATH "${DDIR}" DDIR_NORM)
            if(USE_MSVC_CLI)
                string(TOLOWER "${DDIR_NORM}" DDIR_NORM)
                list(APPEND DEP_INC_FLAGS "/I${DDIR}")
            else()
                list(APPEND DEP_INC_FLAGS "-I${DDIR}")
            endif()
            list(APPEND ALLOWED_KEEL_PREFIXES "${DDIR_NORM}")
        endforeach()
    endif()

    set(BASELINE_SRC "${SCRATCH_DIR}/baseline.cpp")
    file(WRITE "${BASELINE_SRC}" "#include <cstddef>\n#include <cstdint>\n#include <initializer_list>\n#include <new>\nint main() { return 0; }\n")

    if(USE_MSVC_CLI)
        execute_process(
            COMMAND "${COMPILER_PATH}" /nologo /std:c++20 /Zs /showIncludes ${SYS_INC_FLAGS} "${BASELINE_SRC}"
            OUTPUT_VARIABLE BASELINE_OUT
            ERROR_VARIABLE BASELINE_ERR
            RESULT_VARIABLE BASELINE_RES
        )
    else()
        execute_process(
            COMMAND "${COMPILER_PATH}" -std=c++20 -fsyntax-only -H ${SYS_INC_FLAGS} "${BASELINE_SRC}"
            OUTPUT_VARIABLE BASELINE_OUT
            ERROR_VARIABLE BASELINE_ERR
            RESULT_VARIABLE BASELINE_RES
        )
    endif()
    set(COMBINED_BASELINE "${BASELINE_OUT}\n${BASELINE_ERR}")
    if(NOT BASELINE_RES EQUAL 0)
        message(FATAL_ERROR "Failed to compile baseline for header purity check:\n${COMBINED_BASELINE}")
    endif()

    set(ALLOWED_PATHS "")
    string(REPLACE "\r\n" "\n" COMBINED_BASELINE "${COMBINED_BASELINE}")
    string(REPLACE "\n" ";" BASELINE_LINES "${COMBINED_BASELINE}")
    foreach(BLINE IN LISTS BASELINE_LINES)
        if(USE_MSVC_CLI)
            if(BLINE MATCHES "Note: including file:[ \t]*(.+)$")
                string(STRIP "${CMAKE_MATCH_1}" INC_PATH)
                file(TO_CMAKE_PATH "${INC_PATH}" INC_PATH_NORM)
                string(TOLOWER "${INC_PATH_NORM}" INC_PATH_NORM)
                list(APPEND ALLOWED_PATHS "${INC_PATH_NORM}")
            endif()
        else()
            if(BLINE MATCHES "^\\.+ ([^ ]+)$")
                string(STRIP "${CMAKE_MATCH_1}" INC_PATH)
                file(TO_CMAKE_PATH "${INC_PATH}" INC_PATH_NORM)
                list(APPEND ALLOWED_PATHS "${INC_PATH_NORM}")
            endif()
        endif()
    endforeach()
    list(REMOVE_DUPLICATES ALLOWED_PATHS)

    foreach(HDR IN LISTS KEEL_PUBLIC_HEADERS)
        get_filename_component(HDR_NAME "${HDR}" NAME)
        set(TEST_SRC "${SCRATCH_DIR}/check_${HDR_NAME}.cpp")
        file(WRITE "${TEST_SRC}" "#include \"keel/${HDR_NAME}\"\nint main() { return 0; }\n")

        if(USE_MSVC_CLI)
            execute_process(
                COMMAND "${COMPILER_PATH}" /nologo /std:c++20 /Zs /showIncludes ${SYS_INC_FLAGS} ${DEP_INC_FLAGS} "${TEST_SRC}"
                OUTPUT_VARIABLE HDR_OUT
                ERROR_VARIABLE HDR_ERR
                RESULT_VARIABLE HDR_RES
            )
        else()
            execute_process(
                COMMAND "${COMPILER_PATH}" -std=c++20 -fsyntax-only -H ${SYS_INC_FLAGS} ${DEP_INC_FLAGS} "${TEST_SRC}"
                OUTPUT_VARIABLE HDR_OUT
                ERROR_VARIABLE HDR_ERR
                RESULT_VARIABLE HDR_RES
            )
        endif()

        set(COMBINED_HDR "${HDR_OUT}\n${HDR_ERR}")
        if(NOT HDR_RES EQUAL 0)
            message(FATAL_ERROR "Standalone compilation failed for keel/${HDR_NAME}:\n${COMBINED_HDR}")
        endif()

        string(REPLACE "\r\n" "\n" COMBINED_HDR "${COMBINED_HDR}")
        string(REPLACE "\n" ";" HDR_LINES "${COMBINED_HDR}")
        foreach(HLINE IN LISTS HDR_LINES)
            set(EXTRACTED "")
            if(USE_MSVC_CLI)
                if(HLINE MATCHES "Note: including file:[ \t]*(.+)$")
                    string(STRIP "${CMAKE_MATCH_1}" EXTRACTED)
                    file(TO_CMAKE_PATH "${EXTRACTED}" EXTRACTED)
                    string(TOLOWER "${EXTRACTED}" EXTRACTED)
                endif()
            else()
                if(HLINE MATCHES "^\\.+ ([^ ]+)$")
                    string(STRIP "${CMAKE_MATCH_1}" EXTRACTED)
                    file(TO_CMAKE_PATH "${EXTRACTED}" EXTRACTED)
                endif()
            endif()

            if(NOT EXTRACTED STREQUAL "")
                set(IS_KEEL_FAMILY FALSE)
                foreach(PREF IN LISTS ALLOWED_KEEL_PREFIXES)
                    string(FIND "${EXTRACTED}" "${PREF}" POS_PREF)
                    if(POS_PREF EQUAL 0)
                        set(IS_KEEL_FAMILY TRUE)
                        break()
                    endif()
                endforeach()
                if(NOT IS_KEEL_FAMILY)
                    if(NOT EXTRACTED IN_LIST ALLOWED_PATHS)
                        message(FATAL_ERROR "Transitive header purity violation in keel/${HDR_NAME}: unexpected system header '${EXTRACTED}'")
                    endif()
                endif()
            endif()
        endforeach()
    endforeach()
endif()

list(LENGTH KEEL_PUBLIC_HEADERS NUM_KEEL_HEADERS)
message(STATUS "Header purity check passed for all ${NUM_KEEL_HEADERS} public header(s).")
