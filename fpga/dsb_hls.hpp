#pragma once

#ifndef SBM_FPGA_MAX_SPINS
#define SBM_FPGA_MAX_SPINS 256
#endif
#ifndef SBM_FPGA_PC
#define SBM_FPGA_PC 16
#endif
#ifndef SBM_FPGA_PR
#define SBM_FPGA_PR 16
#endif
#ifndef SBM_FPGA_PB
#define SBM_FPGA_PB 1
#endif

#define SBM_FPGA_PARALLEL_ROWS (SBM_FPGA_PR * SBM_FPGA_PB)

#ifdef SBM_USE_XILINX_AP_FIXED
#include <ap_fixed.h>
using sbm_fpga_value = ap_fixed<24, 8, AP_RND, AP_SAT>;
#else
using sbm_fpga_value = float;
#endif

extern "C" void dsb_hls(
    int n, int steps, const sbm_fpga_value* dense_j,
    const sbm_fpga_value* fields, sbm_fpga_value* x, sbm_fpga_value* y,
    sbm_fpga_value a0, sbm_fpga_value dt, sbm_fpga_value c0,
    sbm_fpga_value gamma);
