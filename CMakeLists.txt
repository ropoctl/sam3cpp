cmake_minimum_required(VERSION 3.20)
project(sam3cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()

if (APPLE)
    set(SAM3CPP_ENABLE_METAL_DEFAULT ON)
else()
    set(SAM3CPP_ENABLE_METAL_DEFAULT OFF)
endif()

option(SAM3CPP_ENABLE_METAL "Build ggml with Metal backend support" ${SAM3CPP_ENABLE_METAL_DEFAULT})
option(SAM3CPP_BUILD_PYTHON "Build the pybind11 Python module" OFF)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build bundled ggml as static libraries" FORCE)
set(GGML_STATIC ON CACHE BOOL "Statically link bundled ggml libraries" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "Disable dynamic ggml backend loading" FORCE)
if (NOT APPLE AND SAM3CPP_ENABLE_METAL)
    message(WARNING "SAM3CPP_ENABLE_METAL is only supported on Apple platforms; disabling it")
    set(SAM3CPP_ENABLE_METAL OFF CACHE BOOL "Build ggml with Metal backend support" FORCE)
endif()
set(GGML_METAL ${SAM3CPP_ENABLE_METAL} CACHE BOOL "Enable ggml Metal backend" FORCE)
if (APPLE AND SAM3CPP_ENABLE_METAL)
    execute_process(
        COMMAND xcodebuild -showComponent metalToolchain -json
        RESULT_VARIABLE SAM3CPP_METAL_COMPONENT_RESULT
        OUTPUT_VARIABLE SAM3CPP_METAL_COMPONENT_JSON
        ERROR_QUIET
    )
    if (SAM3CPP_METAL_COMPONENT_RESULT EQUAL 0
        AND SAM3CPP_METAL_COMPONENT_JSON MATCHES "\"status\"[ \t\r\n]*:[ \t\r\n]*\"installed\""
        AND SAM3CPP_METAL_COMPONENT_JSON MATCHES "\"toolchainIdentifier\"[ \t\r\n]*:[ \t\r\n]*\"([^\"]+)\"")
        set(SAM3CPP_METAL_TOOLCHAIN_ID "${CMAKE_MATCH_1}")
        message(STATUS "Using Apple Metal toolchain component: ${SAM3CPP_METAL_TOOLCHAIN_ID}")
        set(SAM3CPP_XCRUN_PREFIX ${CMAKE_COMMAND} -E env "TOOLCHAINS=${SAM3CPP_METAL_TOOLCHAIN_ID}" xcrun -sdk macosx)
    else()
        set(SAM3CPP_XCRUN_PREFIX xcrun -sdk macosx)
    endif()

    execute_process(
        COMMAND ${SAM3CPP_XCRUN_PREFIX} --find metal
        RESULT_VARIABLE SAM3CPP_METAL_RESULT
        OUTPUT_VARIABLE SAM3CPP_METAL_PATH
        ERROR_QUIET
    )
    execute_process(
        COMMAND ${SAM3CPP_XCRUN_PREFIX} --find metallib
        RESULT_VARIABLE SAM3CPP_METALLIB_RESULT
        OUTPUT_VARIABLE SAM3CPP_METALLIB_PATH
        ERROR_QUIET
    )
    if (SAM3CPP_METAL_RESULT EQUAL 0 AND SAM3CPP_METALLIB_RESULT EQUAL 0)
        string(STRIP "${SAM3CPP_METAL_PATH}" SAM3CPP_METAL_PATH)
        string(STRIP "${SAM3CPP_METALLIB_PATH}" SAM3CPP_METALLIB_PATH)
        message(STATUS "Metal compiler: ${SAM3CPP_METAL_PATH}")
        message(STATUS "Metal linker: ${SAM3CPP_METALLIB_PATH}")
        set(GGML_METAL_EMBED_LIBRARY OFF CACHE BOOL "Prefer precompiled default.metallib over runtime source compilation" FORCE)
    else()
        message(WARNING "metal/metallib are unavailable; falling back to runtime Metal source compilation")
        set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "Fallback to runtime Metal source compilation" FORCE)
    endif()
endif()
set(GGML_CUDA OFF CACHE BOOL "Disable ggml CUDA backend" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "Disable ggml Vulkan backend" FORCE)
set(GGML_BLAS OFF CACHE BOOL "Disable ggml BLAS backend" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "Disable ggml OpenMP backend" FORCE)

find_package(ZLIB REQUIRED)

add_subdirectory(third_party/ggml)

add_library(sam3cpp
    src/dummy_prompt.cpp
    src/gguf_model.cpp
    src/decoder.cpp
    src/encoder_fusion.cpp
    src/geometry_encoder.cpp
    src/grounding_head.cpp
    src/pipeline.cpp
    src/segmentation_head.cpp
    src/tokenizer.cpp
    src/text_encoder.cpp
    src/vision_trunk.cpp
    src/vision_neck.cpp
)

target_include_directories(sam3cpp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/examples
)

target_link_libraries(sam3cpp
    PUBLIC
        ggml
        ZLIB::ZLIB
)

target_compile_definitions(sam3cpp
    PRIVATE
        SAM3CPP_DEFAULT_BPE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/bpe_simple_vocab_16e6.txt.gz"
)

add_library(sam3cpp_static STATIC
    src/dummy_prompt.cpp
    src/gguf_model.cpp
    src/decoder.cpp
    src/encoder_fusion.cpp
    src/geometry_encoder.cpp
    src/grounding_head.cpp
    src/pipeline.cpp
    src/segmentation_head.cpp
    src/tokenizer.cpp
    src/text_encoder.cpp
    src/vision_trunk.cpp
    src/vision_neck.cpp
)

target_include_directories(sam3cpp_static
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/examples
)

target_link_libraries(sam3cpp_static
    PUBLIC
        ggml
        ZLIB::ZLIB
)

target_compile_definitions(sam3cpp_static
    PRIVATE
        SAM3CPP_DEFAULT_BPE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/bpe_simple_vocab_16e6.txt.gz"
)
set_target_properties(sam3cpp_static PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(sam3cpp_c SHARED src/c_api.cpp)
target_include_directories(sam3cpp_c PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sam3cpp_c PUBLIC sam3cpp_static)
set_target_properties(sam3cpp_c PROPERTIES OUTPUT_NAME sam3cpp_c)

add_library(sam3cpp_c_static STATIC src/c_api.cpp)
target_include_directories(sam3cpp_c_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sam3cpp_c_static PUBLIC sam3cpp_static)
set_target_properties(sam3cpp_c_static PROPERTIES OUTPUT_NAME sam3cpp_c)
set_target_properties(sam3cpp_c_static PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (SAM3CPP_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_sam3cpp bindings/python/src/module.cpp)
    target_include_directories(_sam3cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(_sam3cpp PRIVATE sam3cpp_static)

    if (APPLE AND SAM3CPP_ENABLE_METAL AND NOT GGML_METAL_EMBED_LIBRARY)
        if (TARGET ggml-metal-lib)
            add_dependencies(_sam3cpp ggml-metal-lib)
        endif()
        add_custom_command(
            TARGET _sam3cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib"
                    "${CMAKE_CURRENT_SOURCE_DIR}/bindings/python/sam3cpp/default.metallib"
            COMMENT "Copy default.metallib next to the Python package"
        )
        install(FILES "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib" DESTINATION sam3cpp)
    endif()

    install(TARGETS _sam3cpp LIBRARY DESTINATION sam3cpp)
endif()

add_executable(sam3-inspect tools/sam3_inspect.cpp)
target_link_libraries(sam3-inspect PRIVATE sam3cpp)

add_executable(sam3-text-encode tools/sam3_text_encode.cpp)
target_link_libraries(sam3-text-encode PRIVATE sam3cpp)

add_executable(sam3-vision-neck tools/sam3_vision_neck.cpp)
target_link_libraries(sam3-vision-neck PRIVATE sam3cpp)

add_executable(sam3-vision-trunk tools/sam3_vision_trunk.cpp)
target_include_directories(sam3-vision-trunk PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/examples)
target_link_libraries(sam3-vision-trunk PRIVATE sam3cpp)

add_executable(sam3-call-image tools/sam3_call_image.cpp)
target_link_libraries(sam3-call-image PRIVATE sam3cpp)

add_executable(sam3-dummy-prompt tools/sam3_dummy_prompt.cpp)
target_link_libraries(sam3-dummy-prompt PRIVATE sam3cpp)

add_executable(sam3-encoder-fusion tools/sam3_encoder_fusion.cpp)
target_link_libraries(sam3-encoder-fusion PRIVATE sam3cpp)

add_executable(sam3-decoder tools/sam3_decoder.cpp)
target_link_libraries(sam3-decoder PRIVATE sam3cpp)

add_executable(sam3-grounding-head tools/sam3_grounding_head.cpp)
target_link_libraries(sam3-grounding-head PRIVATE sam3cpp)

add_executable(sam3-segmentation-head tools/sam3_segmentation_head.cpp)
target_link_libraries(sam3-segmentation-head PRIVATE sam3cpp)

add_executable(sam3-profile tools/sam3_profile.cpp)
target_link_libraries(sam3-profile PRIVATE sam3cpp)
