cmake_minimum_required(VERSION 3.16)
project(fkp_edge_sim C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Optimization flags matching typical embedded toolchain settings
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O2 -Wall -Wextra -Wpedantic)
endif()

# FKP edge simulation executable (local verification only)
add_executable(fkp_sim
    src/main.c
)

target_include_directories(fkp_sim PRIVATE include)
target_link_libraries(fkp_sim m)   # link math library for expf()

# ── ARM Cortex-M cross-compilation target (optional) ─────────────────────
# Uncomment and set TOOLCHAIN_PREFIX to cross-compile for STM32 / ESP32-C3
#
# set(TOOLCHAIN_PREFIX arm-none-eabi-)
# set(CMAKE_C_COMPILER   ${TOOLCHAIN_PREFIX}gcc)
# set(CMAKE_AR           ${TOOLCHAIN_PREFIX}ar)
# set(CMAKE_OBJCOPY      ${TOOLCHAIN_PREFIX}objcopy)
#
# add_executable(fkp_mcu src/main.c)
# target_compile_options(fkp_mcu PRIVATE
#     -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard
#     -mthumb -O3 -Wall --specs=nosys.specs
# )
# target_include_directories(fkp_mcu PRIVATE include)

# ── CMSIS-DSP acceleration (optional, ARM only) ───────────────────────────
# Link CMSIS-DSP for arm_mat_vec_mult_f32 hardware acceleration
# find_library(CMSIS_DSP_LIB CMSIS-DSP)
# if(CMSIS_DSP_LIB)
#     target_link_libraries(fkp_mcu ${CMSIS_DSP_LIB})
#     target_compile_definitions(fkp_mcu PRIVATE USE_CMSIS_DSP)
# endif()

message(STATUS "FKP edge simulation target: fkp_sim")
message(STATUS "Run 'make fkp_sim' to build the local C simulation.")
