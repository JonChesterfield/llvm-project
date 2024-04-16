add_library(gpu_loader OBJECT Main.cpp)

target_include_directories(gpu_loader PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LIBC_SOURCE_DIR}/include
  ${LIBC_SOURCE_DIR}
)

# This utility needs to be compiled for the host system when cross compiling.
if(LLVM_RUNTIMES_TARGET OR LIBC_TARGET_TRIPLE)
  target_compile_options(gpu_loader PUBLIC --target=${LLVM_HOST_TRIPLE})
  target_link_libraries(gpu_loader PUBLIC "--target=${LLVM_HOST_TRIPLE}")
endif()

# /usr/lib/x86_64-linux-gnu/cmake/hsa-runtime64 works, no " or ;
# probably need to patch llvm/runtimes/CMakeLists.txt to pass down to
# the recursive build, I think that's why the top level setting is being ignored
# CMAKE_PREFIX_PATH that is
find_package(hsa-runtime64 QUIET 1.2.0 HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm /usr/lib/x86_64-linux-gnu/cmake/hsa-runtime64)
if(hsa-runtime64_FOUND AND LIBC_TARGET_ARCHITECTURE_IS_AMDGPU)
  add_subdirectory(amdgpu)
endif()

# The CUDA loader requires LLVM to traverse the ELF image for symbols.
find_package(CUDAToolkit 11.2 QUIET)
if(CUDAToolkit_FOUND)
  add_subdirectory(nvptx)
endif()

if(TARGET amdhsa-loader AND LIBC_TARGET_ARCHITECTURE_IS_AMDGPU)
  add_custom_target(libc.utils.gpu.loader)
  add_dependencies(libc.utils.gpu.loader amdhsa-loader)
  set_target_properties(
    libc.utils.gpu.loader
    PROPERTIES
      TARGET amdhsa-loader
      EXECUTABLE "$<TARGET_FILE:amdhsa-loader>"
  )
elseif(TARGET nvptx-loader AND LIBC_TARGET_ARCHITECTURE_IS_NVPTX)
  add_custom_target(libc.utils.gpu.loader)
  add_dependencies(libc.utils.gpu.loader nvptx-loader)
  set_target_properties(
    libc.utils.gpu.loader
    PROPERTIES
      TARGET nvptx-loader
      EXECUTABLE "$<TARGET_FILE:nvptx-loader>"
  )
endif()

foreach(gpu_loader_tgt amdhsa-loader nvptx-loader)
  if(TARGET ${gpu_loader_tgt})
    install(TARGETS ${gpu_loader_tgt}
            DESTINATION ${CMAKE_INSTALL_BINDIR}
            COMPONENT libc)
  endif()
endforeach()
