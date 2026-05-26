# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_subdirectory(rocisa)

include(CheckCXXCompilerFlag)

option(TENSILELITE_ENABLE_NATIVE_CPU "Enable -march=native for GCC/Clang builds" ON)

set(HOST_ISA_COMPILE_FLAGS "")

if(MSVC)
    check_cxx_compiler_flag("/arch:AVX512" COMPILER_SUPPORTS_AVX512_MSVC)
    check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2_MSVC)

    if(COMPILER_SUPPORTS_AVX512_MSVC)
        list(APPEND HOST_ISA_COMPILE_FLAGS "/arch:AVX512")
    elseif(COMPILER_SUPPORTS_AVX2_MSVC)
        list(APPEND HOST_ISA_COMPILE_FLAGS "/arch:AVX2")
    endif()
else()
    if(TENSILELITE_ENABLE_NATIVE_CPU)
        check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    endif()

    check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
    check_cxx_compiler_flag("-mavx512bf16" COMPILER_SUPPORTS_AVX512BF16)
    check_cxx_compiler_flag("-mavx512dq" COMPILER_SUPPORTS_AVX512DQ)
    check_cxx_compiler_flag("-mavx512vl" COMPILER_SUPPORTS_AVX512VL)
    check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

    if(TENSILELITE_ENABLE_NATIVE_CPU AND COMPILER_SUPPORTS_MARCH_NATIVE)
        list(APPEND HOST_ISA_COMPILE_FLAGS "-march=native")
    elseif(COMPILER_SUPPORTS_AVX512F AND COMPILER_SUPPORTS_AVX512BF16 AND COMPILER_SUPPORTS_AVX512DQ AND COMPILER_SUPPORTS_AVX512VL AND COMPILER_SUPPORTS_FMA)
        list(APPEND HOST_ISA_COMPILE_FLAGS "-mavx512f" "-mavx512bf16" "-mavx512dq" "-mavx512vl" "-mfma")
    elseif(COMPILER_SUPPORTS_AVX2)
        list(APPEND HOST_ISA_COMPILE_FLAGS "-mavx2")
        if(COMPILER_SUPPORTS_FMA)
            list(APPEND HOST_ISA_COMPILE_FLAGS "-mfma")
        endif()
    endif()
endif()
 
if(TENSILELITE_ENABLE_AUTOBUILD)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    include(tensilelite_auto_build)
endif()

if(TENSILELITE_ENABLE_HOST)
    add_library(tensilelite-host OBJECT)
    add_library(tensilelite::tensilelite-host ALIAS tensilelite-host)

    set_target_properties(tensilelite-host
        PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON
    )
    target_compile_options(tensilelite-host PUBLIC ${HOST_ISA_COMPILE_FLAGS})
    target_compile_features(tensilelite-host PUBLIC cxx_std_17)
    target_compile_definitions(tensilelite-host
        PUBLIC
            $<BUILD_INTERFACE:TENSILE_DEFAULT_SERIALIZATION>
            $<BUILD_INTERFACE:TENSILE_USE_HIP>
            $<BUILD_INTERFACE:__HIP_HCC_COMPAT_MODE__=1> # if we don't have this we fail tests on 942_304cu
    )

    if(HIPBLASLT_ENABLE_YAML)
        target_link_libraries(tensilelite-host PRIVATE LLVMObjectYAML)
        target_include_directories(tensilelite-host PRIVATE ${LLVM_INCLUDE_DIRS})
        target_compile_definitions(tensilelite-host PUBLIC $<BUILD_INTERFACE:TENSILE_YAML>)
    else()
        target_link_libraries(tensilelite-host PRIVATE ${hipblaslt_msgpack_target})
        target_compile_definitions(tensilelite-host PUBLIC $<BUILD_INTERFACE:TENSILE_MSGPACK>)
    endif()

    target_include_directories(tensilelite-host
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            "${CMAKE_CURRENT_BINARY_DIR}/include"
    )

    target_link_libraries(tensilelite-host
        PUBLIC
            $<BUILD_INTERFACE:rocisa::rocisa-cpp>
            roc::origami
        PRIVATE
            hip::device
    )

    add_subdirectory(src)
    add_subdirectory(include)

    if(TENSILELITE_ENABLE_CLIENT)
        add_subdirectory(client)
    endif()

    if(TENSILELITE_BUILD_TESTING OR BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()
