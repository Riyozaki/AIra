// lc_stream.c — потоковый (инкрементальный) C-движок для nano-EMA / EMA+ADR моделей.
// Точность режима доказана на numpy-эталоне (np_infer_incr.py): чистая EMA — побитово,
// композит — допуск по PPL ±неск.%. Веса LCW2: int8 q(in,out) + fp16 s_out(out), голова fp16→fp32.
//   lc_stream m.lcw2 bench N      — жадная генерация N токенов, печать tok/s
//   lc_stream m.lcw2 ppl val.npy  — PPL (чанки T со сбросом состояний; границы пропускаются)
// Сборка: gcc -O3 -march=native -o lc_stream lc_stream.c -lm
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct { char name[64]; uint8_t dt; uint32_t nd; uint32_t dims[4]; void* data; } Tensor;
static Tensor TENS[256]; static int NT = 0;
static uint32_t L, D, V, T; static float TAU; static int QQ8 = 0; static int QQ8MASK = 7;

static inline float f16(uint16_t h) {
    int s = (h >> 15) & 1, e = (h >> 10) & 31; uint32_t m = h & 1023; float f;
    if (e == 0) f = ldexpf((float)m, -24);
    else if (e == 31) f = m ? NAN : INFINITY;
    else f = ldexpf((float)(m | 1024), e - 25);
    return s ? -f : f;
}
static Tensor* find0(const char* n) {
    for (int i = 0; i < NT; i++) if (!strcmp(TENS[i].name, n)) return &TENS[i];
    return NULL;
}
static Tensor* find(const char* n) { Tensor* t = find0(n); if (!t) { fprintf(stderr, "missing %s\n", n); exit(1);} return t; }

// out = (x @ Q) ⊙ s_out — AVX512: 16 int8 → int32 → ps, FMA по 16 выходам за такт
#include <immintrin.h>
static void mat_q(const float* x, const Tensor* Qt, float* out, int in, int od) {
    const int8_t* W = Qt->data; char nm[72]; snprintf(nm, 72, "%s.s", Qt->name);
    const uint16_t* so = find(nm)->data;
    memset(out, 0, od * sizeof(float));
    for (int i = 0; i < in; i++) {
        float xi = x[i]; if (xi == 0.f) continue;
        const int8_t* w = W + (size_t)i * od;
        __m512 xv = _mm512_set1_ps(xi);
        int j = 0;
        for (; j + 16 <= od; j += 16) {
            __m128i qb = _mm_loadu_si128((const __m128i*)(w + j));
            __m512 qf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qb));
            __m512 acc = _mm512_loadu_ps(out + j);
            _mm512_storeu_ps(out + j, _mm512_fmadd_ps(xv, qf, acc));
        }
        for (; j < od; j++) out[j] += xi * (float)w[j];
    }
    for (int j = 0; j < od; j++) out[j] *= f16(so[j]);
}
// голова: fp32-предекодированная (D,V)
static void mat_f32(const float* x, const float* W, float* out, int in, int od) {
    memset(out, 0, od * sizeof(float));
    for (int i = 0; i < in; i++) {
        float xi = x[i]; if (xi == 0.f) continue;
        const float* w = W + (size_t)i * od;
        for (int j = 0; j < od; j++) out[j] += xi * w[j];
    }
}
static void lnrow(float* x, const uint16_t* g, const uint16_t* b) {
    float mu = 0, va = 0;
    for (uint32_t i = 0; i < D; i++) mu += x[i]; mu /= D;
    for (uint32_t i = 0; i < D; i++) { float d = x[i] - mu; va += d * d; } va /= D;
    float ist = 1.f / sqrtf(va + 1e-5f);
    for (uint32_t i = 0; i < D; i++) x[i] = f16(g[i]) * (x[i] - mu) * ist + f16(b[i]);
}
static inline float gelu(float x) {
    return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static float* HSTATE[16];
static int TT = 0;
static double T_BODY = 0, T_HEAD = 0;
static inline double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9*t.tv_nsec; }
static void reset_states(void) { for (uint32_t l = 0; l < L; l++) memset(HSTATE[l], 0, D * 4); TT = 0; }

// u8×i8 dot по n (n % 64 == 0 → AVX512 maddubs)
static int32_t dot8_u8(const uint8_t* a, const int8_t* b, int n) {
    if (n % 64 == 0) {
        __m512i acc = _mm512_setzero_si512();
        for (int c = 0; c < n; c += 64) {
            __m512i xb = _mm512_loadu_si512((const void*)(a + c));
            __m512i qb = _mm512_loadu_si512((const void*)(b + c));
            __m512i pr = _mm512_maddubs_epi16(xb, qb);
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(pr, _mm512_set1_epi16(1)));
        }
        return _mm512_reduce_add_epi32(acc);
    }
    int32_t d = 0; for (int i = 0; i < n; i++) d += (int)a[i] * (int)b[i]; return d;
}

// int8×int8 матвектор: W хранится (od,in) строками по входу (dot-порядок), активации квантуются absmax.
static void mat_qq8(const float* x, const Tensor* Qt, float* out, int in, int od) {
    const int8_t* W = Qt->data; char nm[72];
    snprintf(nm, 72, "%s.s", Qt->name);  const uint16_t* so = find(nm)->data;
    snprintf(nm, 72, "%s.qs", Qt->name); const int32_t* qs = find(nm)->data;
    float ax = 0.f; for (int i = 0; i < in; i++) { float a = fabsf(x[i]); if (a > ax) ax = a; }
    float sx = ax > 0 ? ax / 127.f : 1.f;
    static uint8_t xq[4096];
    for (int i = 0; i < in; i++) xq[i] = (uint8_t)((int)lrintf(x[i] / sx) + 128);
    int j = 0;
    if (in % 64 == 0) {
        for (; j + 4 <= od; j += 4) {                // 4 строки: один xb-load на четвёрку
            __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
            const int8_t* w0 = W + (size_t)(j+0)*in; const int8_t* w1 = W + (size_t)(j+1)*in;
            const int8_t* w2 = W + (size_t)(j+2)*in; const int8_t* w3 = W + (size_t)(j+3)*in;
            for (int c = 0; c < in; c += 64) {
                __m512i xb = _mm512_loadu_si512((const void*)(xq + c));
                a0 = _mm512_add_epi32(a0, _mm512_madd_epi16(_mm512_maddubs_epi16(xb, _mm512_loadu_si512((const void*)(w0+c))), _mm512_set1_epi16(1)));
                a1 = _mm512_add_epi32(a1, _mm512_madd_epi16(_mm512_maddubs_epi16(xb, _mm512_loadu_si512((const void*)(w1+c))), _mm512_set1_epi16(1)));
                a2 = _mm512_add_epi32(a2, _mm512_madd_epi16(_mm512_maddubs_epi16(xb, _mm512_loadu_si512((const void*)(w2+c))), _mm512_set1_epi16(1)));
                a3 = _mm512_add_epi32(a3, _mm512_madd_epi16(_mm512_maddubs_epi16(xb, _mm512_loadu_si512((const void*)(w3+c))), _mm512_set1_epi16(1)));
            }
            out[j+0] = (float)(_mm512_reduce_add_epi32(a0) - 128*qs[j+0]) * (sx * f16(so[j+0]));
            out[j+1] = (float)(_mm512_reduce_add_epi32(a1) - 128*qs[j+1]) * (sx * f16(so[j+1]));
            out[j+2] = (float)(_mm512_reduce_add_epi32(a2) - 128*qs[j+2]) * (sx * f16(so[j+2]));
            out[j+3] = (float)(_mm512_reduce_add_epi32(a3) - 128*qs[j+3]) * (sx * f16(so[j+3]));
        }
    }
    for (; j < od; j++)
        out[j] = (float)(dot8_u8(xq, W + (size_t)j*in, in) - 128*qs[j]) * (sx * f16(so[j]));
}


static void step(const int tok, float* x /*D*/) {
    const uint16_t* E = find("E")->data; const uint16_t* pos = find("pos")->data;
    if (TT >= (int)T) reset_states();
    for (uint32_t j = 0; j < D; j++) x[j] = f16(E[tok * D + j]) + f16(pos[TT * D + j]);
    static float y[1024], mix[1024], h2[1024], f1[4096], o[1024];
    for (uint32_t l = 0; l < L; l++) {
        char nm[72];
        snprintf(nm, 72, "b%u.rw", l); Tensor* rwt = find0(nm);
        float g = 1.f;
        if (rwt) {
            const uint16_t* rw = rwt->data;
            snprintf(nm, 72, "b%u.rb", l); float rb = f16(*(const uint16_t*)find(nm)->data);
            float s = rb; for (uint32_t j = 0; j < D; j++) s += x[j] * f16(rw[j]);
            g = 1.f / (1.f + expf(-s));
            if (g < TAU) continue;                  // не прошёл ворота — вход наследуется
        }
        snprintf(nm, 72, "b%u.ln1", l); const uint16_t* g1 = find(nm)->data;
        snprintf(nm, 72, "b%u.ln1.bias", l); const uint16_t* b1 = find(nm)->data;
        memcpy(y, x, D * 4); lnrow(y, g1, b1);
        snprintf(nm, 72, "b%u.th", l); const uint16_t* th = find(nm)->data;
        snprintf(nm, 72, "b%u.sc", l); const uint16_t* sc = find(nm)->data;
        float* h = HSTATE[l];
        for (uint32_t j = 0; j < D; j++) {
            float a = 1.f / (1.f + expf(-f16(th[j])));
            h[j] = a * h[j] + (1.f - a) * y[j];
            y[j] = h[j] * f16(sc[j]);
        }
        snprintf(nm, 72, "b%u.Wm", l);
        if (QQ8 && (QQ8MASK & 1)) mat_qq8(y, find(nm), mix, D, D); else mat_q(y, find(nm), mix, D, D);
        for (uint32_t j = 0; j < D; j++) h2[j] = x[j] + g * mix[j];
        memcpy(y, h2, D * 4);
        snprintf(nm, 72, "b%u.ln2", l); const uint16_t* g2 = find(nm)->data;
        snprintf(nm, 72, "b%u.ln2.bias", l); const uint16_t* b2 = find(nm)->data;
        lnrow(y, g2, b2);
        uint32_t F = QQ8 ? find("b0.fc1")->dims[0] : find("b0.fc1")->dims[1];
        snprintf(nm, 72, "b%u.fc1", l);
        if (QQ8 && (QQ8MASK & 2)) mat_qq8(y, find(nm), f1, D, F); else mat_q(y, find(nm), f1, D, F);
        for (uint32_t j = 0; j < F; j++) f1[j] = gelu(f1[j]);
        snprintf(nm, 72, "b%u.fc2", l);
        if (QQ8 && (QQ8MASK & 4)) mat_qq8(f1, find(nm), o, F, D); else mat_q(f1, find(nm), o, F, D);
        for (uint32_t j = 0; j < D; j++) x[j] = h2[j] + g * o[j];
    }
    TT++;
}
// голова int8×int8: активации квантуются динамически (absmax→127), качество ≡ fp32 (numpy: Δ+0.0004 ната)
static void logits_of(const float* x, float* lg /*V*/) {
    static float z[1024];
    memcpy(z, x, D * 4);
    lnrow(z, find("lnf")->data, find("lnf.bias")->data);
    float ax = 0.f; for (uint32_t i = 0; i < D; i++) { float a = fabsf(z[i]); if (a > ax) ax = a; }
    float sx = ax > 0 ? ax / 127.f : 1.f;
    static uint8_t xq[1024];
    __m512i sh = _mm512_set1_epi8(-128);
    for (uint32_t i = 0; i < D; i++) xq[i] = (uint8_t)((int)lrintf(z[i] / sx) + 128);
    const int8_t* Q = find("EtT")->data;                 // (V,D) row-major
    const uint16_t* so = find("EtT.s")->data;
    const int32_t* qs = find("EtT.qsum")->data;
    if (D % 64 == 0) {
        for (uint32_t v = 0; v < V; v++) {
            const int8_t* qrow = Q + (size_t)v * D;
            __m512i acc = _mm512_setzero_si512();
            for (uint32_t c = 0; c < D; c += 64) {
                __m512i xb = _mm512_loadu_si512((const void*)(xq + c));
                __m512i qb = _mm512_loadu_si512((const void*)(qrow + c));
                __m512i pr = _mm512_maddubs_epi16(xb, qb);          // u8×i8 → i16 попарные
                acc = _mm512_add_epi32(acc, _mm512_madd_epi16(pr, _mm512_set1_epi16(1)));
            }
            int32_t dot = _mm512_reduce_add_epi32(acc) - 128 * qs[v];
            lg[v] = (float)dot * sx * f16(so[v]);
        }
    } else {                                             // откат: плотный скаляр
        for (uint32_t v = 0; v < V; v++) {
            const int8_t* qrow = Q + (size_t)v * D;
            int32_t dot = -128 * qs[v];
            for (uint32_t i = 0; i < D; i++) dot += (int)(xq[i]) * qrow[i];
            lg[v] = (float)dot * sx * f16(so[v]);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: lc_stream m.lcw2 <bench N | ppl file.npy>\n"); return 1; }
    FILE* f = fopen(argv[1], "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    char magic[4]; fread(magic, 1, 4, f);
    if (memcmp(magic, "LCW2", 4)) { fprintf(stderr, "bad magic\n"); return 1; }
    uint32_t hdr[5]; fread(hdr, 4, 5, f); fread(&TAU, 4, 1, f);
    L = hdr[0]; D = hdr[1]; V = hdr[3]; T = hdr[4];
    uint32_t nt; fread(&nt, 4, 1, f); NT = nt;
    for (int i = 0; i < (int)nt; i++) {
        uint64_t nl; fread(&nl, 8, 1, f);
        Tensor* t = &TENS[i]; fread(t->name, 1, nl, f); t->name[nl] = 0;
        fread(&t->dt, 1, 1, f); fread(&t->nd, 4, 1, f); fread(t->dims, 4, t->nd, f);
        size_t bytes = 1; for (uint32_t j = 0; j < t->nd; j++) bytes *= t->dims[j];
        bytes *= (t->dt == 2 ? 1 : (t->dt == 0 ? 4 : 2));
        t->data = malloc(bytes); fread(t->data, 1, bytes, f);
    }
    fclose(f);
    { const char* m = getenv("QQ8MASK"); if (m) QQ8MASK = atoi(m); } QQ8 = getenv("QQ8OFF") ? 0 : (find0("b0.Wm.qs") != NULL);
    fprintf(stderr, "QQ8=%d\n", QQ8);
    for (uint32_t l = 0; l < L; l++) HSTATE[l] = malloc(D * 4);
    static float x[1024]; static float* lg = NULL; lg = malloc(V * 4);
    if (!strcmp(argv[2], "bench")) {
        int n = atoi(argv[3]); int cur = 2;
        struct timespec a, b;
        reset_states();
        clock_gettime(CLOCK_MONOTONIC, &a);
        for (int gg = 0; gg < n; gg++) {
            double t0 = now_s(); step(cur, x); double t1 = now_s();
            logits_of(x, lg); double t2 = now_s();
            T_BODY += t1 - t0; T_HEAD += t2 - t1;
                int best = 1; float bl = lg[1];
            for (uint32_t j = 2; j < V; j++) if (lg[j] > bl) { bl = lg[j]; best = j; }
            cur = best;
        }
        clock_gettime(CLOCK_MONOTONIC, &b);
        double dt = (double)(b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec);
        printf("stream int8: %d tokens in %.3fs -> %.1f tok/s | body %.1f%% head %.1f%%\n",
               n, dt, n / dt, 100*T_BODY/dt, 100*T_HEAD/dt);
    } else if (!strcmp(argv[2], "ppl")) {
        FILE* vf = fopen(argv[3], "rb");
        fseek(vf, 8, SEEK_SET); uint16_t hl; fread(&hl, 2, 1, vf); fseek(vf, 10 + hl, SEEK_SET);
        static uint16_t toks[1 << 20]; size_t m = fread(toks, 2, 1 << 20, vf); fclose(vf);
        double nll = 0; long cnt = 0; reset_states();
        size_t lim = m > 4096 ? 4096 : m;
        for (size_t ti = 0; ti + 1 < lim; ti++) {
            step(toks[ti], x);
            logits_of(x, lg);
            float mx = lg[0]; for (uint32_t j = 1; j < V; j++) if (lg[j] > mx) mx = lg[j];
            float sm = 0; for (uint32_t j = 0; j < V; j++) sm += expf(lg[j] - mx);
            if ((int)(TT - 1) % (int)T == (int)T - 1) continue;
            nll += log(sm) + mx - lg[toks[ti + 1]]; cnt++;
        }
        printf("stream ppl = %.2f over %ld tokens\n", exp(nll / cnt), cnt);
    }
    return 0;
}
