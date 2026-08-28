// lc_kernels.c — горячие elementwise ядра для nano_lc.py через ctypes.
// f32, семантика 1:1 с numpy-версиями. gcc -O3 -march=native векторизует сам.
#include <math.h>
#include <stdint.h>

// ---------------------------------------------------------------- gelu (tanh-approx)
void k_gelu_fwd(const float* x, float* y, float* t_cache, int64_t n) {
    const float c0 = 0.7978845608028654f, c1 = 0.044715f;
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        float u = c0 * (v + c1 * v * v * v);
        float t = tanhf(u);
        t_cache[i] = t;
        y[i] = 0.5f * v * (1.f + t);
    }
}
void k_gelu_bwd(const float* x, const float* t, const float* dy, float* dx, int64_t n) {
    const float c0 = 0.7978845608028654f, c1 = 3.f * 0.044715f;
    for (int64_t i = 0; i < n; i++) {
        float tt = t[i];
        float du = c0 * (1.f + c1 * x[i] * x[i]);
        dx[i] = dy[i] * (0.5f * (1.f + tt) + 0.5f * x[i] * (1.f - tt * tt) * du);
    }
}

// ----------------------------------------- layernorm по последней оси: x (R,N)
// x̂=(x−μ)·rstd; y = x̂·g + b; x̂ кэшируем для bwd
void k_ln_fwd(const float* x, const float* g, const float* b, float* y, float* xhat,
              float* mu_c, float* rs_c, int64_t R, int N, float eps) {
    for (int64_t r = 0; r < R; r++) {
        const float* xr = x + r * N; float* yr = y + r * N; float* hr = xhat + r * N;
        float m = 0.f; for (int j = 0; j < N; j++) m += xr[j]; m /= N;
        float v = 0.f; for (int j = 0; j < N; j++) { float d = xr[j] - m; v += d * d; }
        v /= N;
        float rs = 1.f / sqrtf(v + eps);
        mu_c[r] = m; rs_c[r] = rs;
        for (int j = 0; j < N; j++) { float h = (xr[j] - m) * rs; hr[j] = h; yr[j] = h * g[j] + b[j]; }
    }
}
// dx = rstd/N·(N·(dy·g) − Σ(dy·g) − x̂·Σ(dy·g·x̂)); dg, db ПРИБАВЛЯЮТСЯ (не зануляются!)
void k_ln_bwd(const float* g, const float* dy, const float* xhat, const float* rs_c,
              float* dx, float* dg, float* db, int64_t R, int N) {
    float Nf = (float)N;
    // dg/db извне приходят с прежним значением (аккумуляция), обнуляем тут:
    for (int j = 0; j < N; j++) { dg[j] = 0.f; db[j] = 0.f; }
    for (int64_t r = 0; r < R; r++) {
        const float* dyr = dy + r * N; const float* hr = xhat + r * N; float* dxr = dx + r * N;
        float rs = rs_c[r];
        float s1 = 0.f, s2 = 0.f;
        for (int j = 0; j < N; j++) {
            float d = dyr[j] * g[j];
            dxr[j] = d;
            s1 += d; s2 += d * hr[j];
            dg[j] += dyr[j] * hr[j]; db[j] += dyr[j];
        }
        float k = rs / Nf;
        for (int j = 0; j < N; j++) dxr[j] = k * (Nf * dxr[j] - s1 - hr[j] * s2);
    }
}

// ----------------------------------------- softmax+CE (слито): lg (R,V) → на месте dz
// loss = mean NLL; dz = (p − onehot)·invn, invn = 1/R. Возвращает loss (double).
double k_sce(float* lg, const int64_t* y, int64_t R, int64_t V, float invn) {
    double nll = 0.0;
    for (int64_t r = 0; r < R; r++) {
        float* lr = lg + r * V;
        float ty = lr[y[r]];                                   // логит цели до перезаписи
        float mx = lr[0]; for (int64_t j = 1; j < V; j++) if (lr[j] > mx) mx = lr[j];
        float sm = 0.f;
        for (int64_t j = 0; j < V; j++) { float e = expf(lr[j] - mx); lr[j] = e; sm += e; }
        nll += log((double)sm) + (double)mx - (double)ty;
        float k = invn / sm;
        for (int64_t j = 0; j < V; j++) lr[j] *= k;
        lr[y[r]] -= invn;
    }
    return nll / (double)R;
}
