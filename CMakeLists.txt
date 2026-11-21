cmake_minimum_required(VERSION 3.20)
project(keel_wavelet VERSION 0.1.0 LANGUAGES CXX)

option(KEEL_WAVELET_NO_EXCEPTIONS_RTTI "Compile keel_wavelet without C++ exceptions and RTTI" OFF)
option(KEEL_WAVELET_BUILD_TESTS "Build keel_wavelet unit tests and header purity checks" ${PROJECT_IS_TOP_LEVEL})

if(NOT TARGET keel::keel)
    find_package(keel QUIET CONFIG)
endif()
if(NOT TARGET keel::keel)
    include(FetchContent)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../keel/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL)
        set(FETCHCONTENT_SOURCE_DIR_KEEL "${CMAKE_CURRENT_SOURCE_DIR}/../keel" CACHE PATH "Local keel checkout")
    endif()
    if(KEEL_WAVELET_NO_EXCEPTIONS_RTTI)
        set(KEEL_NO_EXCEPTIONS_RTTI ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(
        keel
        GIT_REPOSITORY https://github.com/keelcpp/keel.git
        GIT_TAG        869a5c07a5d77341e5f6acbac5f8e024b274ea08
    )
    FetchContent_MakeAvailable(keel)
endif()

if(NOT TARGET keel::fft)
    find_package(keel_fft QUIET CONFIG)
    if(TARGET keel::keel_fft AND NOT TARGET keel::fft)
        add_library(keel::fft ALIAS keel::keel_fft)
    endif()
endif()
if(NOT TARGET keel::fft)
    include(FetchContent)
    set(KEEL_FFT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../fft/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL_FFT)
        set(FETCHCONTENT_SOURCE_DIR_KEEL_FFT "${CMAKE_CURRENT_SOURCE_DIR}/../fft" CACHE PATH "Local keel_fft checkout")
    endif()
    if(KEEL_WAVELET_NO_EXCEPTIONS_RTTI)
        set(KEEL_FFT_NO_EXCEPTIONS_RTTI ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(
        keel_fft
        GIT_REPOSITORY https://github.com/keelcpp/fft.git
        GIT_TAG        abd91d6693d5f2a56311ecc4fbb38e2555ee0486
    )
    FetchContent_MakeAvailable(keel_fft)
endif()

add_library(keel_wavelet STATIC
    src/filters.cpp
    src/dwt.cpp
    src/multilevel.cpp
    src/swt.cpp
    src/denoise.cpp
    src/shape.cpp
)
add_library(keel::wavelet ALIAS keel_wavelet)

target_link_libraries(keel_wavelet
    PUBLIC  keel::keel
    PRIVATE keel::fft
)

target_compile_features(keel_wavelet PUBLIC cxx_std_20)
set_target_properties(keel_wavelet PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(keel_wavelet PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(keel_wavelet PRIVATE /W4 /utf-8)
    if(KEEL_WAVELET_NO_EXCEPTIONS_RTTI)
        string(REGEX REPLACE "/EH[a-z]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REGEX REPLACE "/GR" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        target_compile_options(keel_wavelet PUBLIC /EHs-c- /GR-)
        target_compile_definitions(keel_wavelet PUBLIC _HAS_EXCEPTIONS=0)
    endif()
else()
    target_compile_options(keel_wavelet PRIVATE -Wall -Wextra -Wpedantic)
    if(KEEL_WAVELET_NO_EXCEPTIONS_RTTI)
        target_compile_options(keel_wavelet PUBLIC -fno-exceptions -fno-rtti)
    endif()
endif()

if(COMMAND keel_enforce_guidelines)
    keel_enforce_guidelines(keel_wavelet
        SCOPE ALL
        BAN_RAW_INTS ALL
        BAN_RAW_FLOATS ALL
        HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include"
        STANDALONE_HEADERS ${KEEL_WAVELET_BUILD_TESTS}
        TEST_NAME test_guidelines_headers
        WARNINGS OFF
    )
endif()

include(CMakePackageConfigHelpers)
install(TARGETS keel_wavelet EXPORT keel_waveletTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/keel DESTINATION include)
install(EXPORT keel_waveletTargets
    FILE keel_waveletTargets.cmake
    NAMESPACE keel::
    DESTINATION lib/cmake/keel_wavelet
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/keel_waveletConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/keel_waveletConfig.cmake"
    INSTALL_DESTINATION lib/cmake/keel_wavelet
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/keel_waveletConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/keel_waveletConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/keel_waveletConfigVersion.cmake"
    DESTINATION lib/cmake/keel_wavelet
)

if(KEEL_WAVELET_BUILD_TESTS)
    enable_testing()

    add_executable(test_wavelet tests/test_wavelet.cpp)
    target_link_libraries(test_wavelet PRIVATE keel::wavelet)
    if(MSVC)
        target_compile_options(test_wavelet PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_wavelet PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME test_wavelet COMMAND test_wavelet)

    get_target_property(KEEL_CORE_INC_DIRS keel::keel INTERFACE_INCLUDE_DIRECTORIES)
    set(KEEL_CORE_INC_CLEAN "")
    foreach(IDIR IN LISTS KEEL_CORE_INC_DIRS)
        string(REGEX REPLACE "\\$<BUILD_INTERFACE:([^>]+)>" "\\1" IDIR_STRIPPED "${IDIR}")
        if(NOT IDIR_STRIPPED MATCHES "\\$<INSTALL_INTERFACE:" AND EXISTS "${IDIR_STRIPPED}")
            list(APPEND KEEL_CORE_INC_CLEAN "${IDIR_STRIPPED}")
        endif()
    endforeach()
    string(REPLACE ";" "|" KEEL_DEP_INC_JOINED "${KEEL_CORE_INC_CLEAN}")
    string(REPLACE ";" "|" KEEL_STD_INC_JOINED "${CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES}")

    add_test(
        NAME test_header_purity
        COMMAND ${CMAKE_COMMAND}
            "-DKEEL_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/include"
            "-DKEEL_DEP_INC_DIRS=${KEEL_DEP_INC_JOINED}"
            "-DCOMPILER_PATH=${CMAKE_CXX_COMPILER}"
            "-DCOMPILER_ID=${CMAKE_CXX_COMPILER_ID}"
            "-DSTD_INC_DIRS=${KEEL_STD_INC_JOINED}"
            "-DCMAKE_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckHeaderPurity.cmake"
    )
endif()
