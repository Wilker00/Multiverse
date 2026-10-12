/*
 * engine3d/c_rasterizer.c
 *
 * Native barycentric triangle kernel matching engine3d/rasterizer.py.
 * The Python Rasterizer is the runtime path. Compile this unit when a
 * C toolchain is available:
 *   gcc -O3 -c engine3d/c_rasterizer.c
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

static int iclamp(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

void engine3d_clear(uint8_t *color, float *depth, int width, int height,
                    uint8_t r, uint8_t g, uint8_t b) {
    int n = width * height;
    for (int i = 0; i < n; ++i) {
        color[i * 3 + 0] = r;
        color[i * 3 + 1] = g;
        color[i * 3 + 2] = b;
        depth[i] = INFINITY;
    }
}

void engine3d_raster_triangle(
    uint8_t *color,
    float *depth,
    int width,
    int height,
    float x0, float y0, float z0,
    float x1, float y1, float z1,
    float x2, float y2, float z2,
    uint8_t r, uint8_t g, uint8_t b
) {
    float area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
    if (area <= 0.0f) {
        return;
    }
    int min_x = iclamp((int)floorf(fminf(x0, fminf(x1, x2))), 0, width - 1);
    int max_x = iclamp((int)ceilf(fmaxf(x0, fmaxf(x1, x2))), 0, width - 1);
    int min_y = iclamp((int)floorf(fminf(y0, fminf(y1, y2))), 0, height - 1);
    int max_y = iclamp((int)ceilf(fmaxf(y0, fmaxf(y1, y2))), 0, height - 1);
    float inv_area = 1.0f / area;

    for (int y = min_y; y <= max_y; ++y) {
        for (int x = min_x; x <= max_x; ++x) {
            float px = (float)x;
            float py = (float)y;
            float w0 = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1);
            float w1 = (x0 - x2) * (py - y2) - (y0 - y2) * (px - x2);
            float w2 = (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0);
            if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) {
                continue;
            }
            float z = (w0 * z0 + w1 * z1 + w2 * z2) * inv_area;
            int idx = y * width + x;
            if (z < depth[idx]) {
                depth[idx] = z;
                color[idx * 3 + 0] = r;
                color[idx * 3 + 1] = g;
                color[idx * 3 + 2] = b;
            }
        }
    }
}
