// lc_kernels.c — горячие elementwise ядра для nano_lc.py через ctypes.
// f32, семантика 1:1 с numpy-версиями. gcc -O3 -march=native векторизует сам.
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

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

// ----------------------------------------- EMA-миксер: h_t = a⊙h_{t−1} + (1−a)⊙x_t
// параллелим по B·D, последовательны по T. X (B,T,D) row-major.
void k_ema_fwd(const float* X, const float* a, const float* sc, float* Y, float* H,
               int B, int T, int D) {
    int BD = B * D;
    for (int b = 0; b < B; b++) {
        const float* xb = X + (size_t)b * T * D;
        float* hb = H + (size_t)b * T * D; float* yb = Y + (size_t)b * T * D;
        for (int d = 0; d < D; d++) hb[d] = (1.f - a[d]) * xb[d];
        for (int t = 1; t < T; t++) {
            const float* xt = xb + (size_t)t * D; float* ht = hb + (size_t)t * D;
            const float* hp = ht - D;
            for (int d = 0; d < D; d++) ht[d] = a[d] * hp[d] + (1.f - a[d]) * xt[d];
        }
        for (int t = 0; t < T; t++) {
            float* yt = yb + (size_t)t * D; float* ht = hb + (size_t)t * D;
            for (int d = 0; d < D; d++) yt[d] = ht[d] * sc[d];
        }
    }
    (void)BD;
}
// lam_t = dH_t + a·lam_{t+1}; dX=(1−a)lam; da=Σ lam(hprev−x); dth=da·a(1−a); dsc=Σ dY·H
void k_ema_bwd(const float* X, const float* H, const float* a, const float* sc,
               const float* dY, float* dX, float* dth, float* dsc, int B, int T, int D) {
    for (int d = 0; d < D; d++) { dth[d] = 0.f; dsc[d] = 0.f; }
    static __thread float* lambuf = 0; static __thread int lamcap = 0;
    if (lamcap < D) { lambuf = (float*)realloc(lambuf, D * 4); lamcap = D; }
    for (int b = 0; b < B; b++) {
        const float* xb = X + (size_t)b * T * D;
        const float* hb = H + (size_t)b * T * D;
        const float* yb = dY + (size_t)b * T * D;
        float* dxb = dX + (size_t)b * T * D;
        for (int t = 0; t < T; t++) {
            const float* yt = yb + (size_t)t * D; const float* ht = hb + (size_t)t * D;
            for (int d = 0; d < D; d++) dsc[d] += yt[d] * ht[d];
        }
        float* lm = lambuf;
        for (int d = 0; d < D; d++) lm[d] = 0.f;
        for (int t = T - 1; t >= 0; t--) {
            const float* yt = yb + (size_t)t * D;
            const float* ht = hb + (size_t)t * D;
            const float* xt = xb + (size_t)t * D;
            float* dxt = dxb + (size_t)t * D;
            const float* hp = t > 0 ? ht - D : 0;
            for (int d = 0; d < D; d++) {
                lm[d] = yt[d] * sc[d] + a[d] * lm[d];
                dxt[d] = (1.f - a[d]) * lm[d];
                float hprev = t > 0 ? hp[d] : 0.f;
                dth[d] += lm[d] * (hprev - xt[d]);
            }
        }
    }
    for (int d = 0; d < D; d++) dth[d] *= a[d] * (1.f - a[d]);
}
