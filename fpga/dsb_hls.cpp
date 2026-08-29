#include "dsb_hls.hpp"

#ifdef SBM_USE_XILINX_AP_FIXED
using accumulator_type = ap_fixed<40, 20, AP_RND, AP_SAT>;
#else
using accumulator_type = float;
#endif

extern "C" void dsb_hls(
    int n, int steps, const sbm_fpga_value* dense_j,
    const sbm_fpga_value* fields, sbm_fpga_value* x, sbm_fpga_value* y,
    sbm_fpga_value a0, sbm_fpga_value dt, sbm_fpga_value c0,
    sbm_fpga_value gamma) {
#pragma HLS INTERFACE m_axi port=dense_j offset=slave bundle=jmem
#pragma HLS INTERFACE m_axi port=fields offset=slave bundle=state
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=state
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=state
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static sbm_fpga_value j[SBM_FPGA_MAX_SPINS][SBM_FPGA_MAX_SPINS];
    static sbm_fpga_value state_x[SBM_FPGA_MAX_SPINS];
    static sbm_fpga_value state_y[SBM_FPGA_MAX_SPINS];
    static sbm_fpga_value field_cache[SBM_FPGA_MAX_SPINS];
    static bool signs[SBM_FPGA_MAX_SPINS];
#pragma HLS ARRAY_PARTITION variable=j cyclic factor=SBM_FPGA_PARALLEL_ROWS dim=1
#pragma HLS ARRAY_RESHAPE variable=j cyclic factor=SBM_FPGA_PC dim=2
#pragma HLS ARRAY_PARTITION variable=state_x cyclic factor=SBM_FPGA_PARALLEL_ROWS dim=1
#pragma HLS ARRAY_PARTITION variable=state_y cyclic factor=SBM_FPGA_PARALLEL_ROWS dim=1
#pragma HLS ARRAY_PARTITION variable=field_cache cyclic factor=SBM_FPGA_PARALLEL_ROWS dim=1
#pragma HLS ARRAY_RESHAPE variable=signs cyclic factor=SBM_FPGA_PC dim=1

    if (n <= 0 || n > SBM_FPGA_MAX_SPINS || steps <= 0) return;

    for (int row = 0; row < n; ++row) {
        state_x[row] = x[row];
        state_y[row] = y[row];
        field_cache[row] = fields[row];
        for (int column = 0; column < n; ++column) {
#pragma HLS PIPELINE II=1
            j[row][column] = dense_j[row * n + column];
        }
    }

    const sbm_fpga_value da = a0 / steps;
    sbm_fpga_value a = 0;
    for (int step = 0; step < steps; ++step) {
        for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
            signs[i] = state_x[i] >= 0;
        }

        for (int row_base = 0; row_base < n; row_base += SBM_FPGA_PARALLEL_ROWS) {
            accumulator_type accum[SBM_FPGA_PARALLEL_ROWS];
#pragma HLS ARRAY_PARTITION variable=accum complete
            for (int r = 0; r < SBM_FPGA_PARALLEL_ROWS; ++r) {
#pragma HLS UNROLL
                const int row = row_base + r;
                accum[r] = row < n ? field_cache[row] : sbm_fpga_value(0);
            }

            for (int column_base = 0; column_base < n; column_base += SBM_FPGA_PC) {
#pragma HLS PIPELINE II=1
                for (int r = 0; r < SBM_FPGA_PARALLEL_ROWS; ++r) {
#pragma HLS UNROLL
                    for (int c = 0; c < SBM_FPGA_PC; ++c) {
#pragma HLS UNROLL
                        const int row = row_base + r;
                        const int column = column_base + c;
                        if (row < n && column < n) {
                            accum[r] += signs[column] ? j[row][column] : -j[row][column];
                        }
                    }
                }
            }

            for (int r = 0; r < SBM_FPGA_PARALLEL_ROWS; ++r) {
#pragma HLS UNROLL
                const int i = row_base + r;
                if (i < n) {
                    const sbm_fpga_value previous_y = state_y[i];
                    sbm_fpga_value next_y =
                        previous_y + ((a - a0) * state_x[i] + c0 * accum[r]) * dt;
                    sbm_fpga_value next_x = state_x[i] + a0 * next_y * dt;
                    if (next_x > 1 || next_x < -1) {
                        next_x = next_x >= 0 ? sbm_fpga_value(1) : sbm_fpga_value(-1);
                        next_y = 0;
                    }
                    state_x[i] = next_x;
                    state_y[i] = next_y + gamma * previous_y * dt;
                }
            }
        }
        a += da;
    }

    for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
        x[i] = state_x[i];
        y[i] = state_y[i];
    }
}
