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
static uint32_t L, D, V, T; static float TAU;

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

// out = (x @ Q) ⊙ s_out — плотный int8-путь (конверсия+мад: автовекторизуется)
static void mat_q(const float* x, const Tensor* Qt, float* out, int in, int od) {
    const int8_t* W = Qt->data; char nm[72]; snprintf(nm, 72, "%s.s", Qt->name);
    const uint16_t* so = find(nm)->data;
    memset(out, 0, od * sizeof(float));
    for (int i = 0; i < in; i++) {
        float xi = x[i]; if (xi == 0.f) continue;
        const int8_t* w = W + (size_t)i * od;
        for (int j = 0; j < od; j++) out[j] += xi * (float)w[j];
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
static void reset_states(void) { for (uint32_t l = 0; l < L; l++) memset(HSTATE[l], 0, D * 4); TT = 0; }

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
        snprintf(nm, 72, "b%u.Wm", l); mat_q(y, find(nm), mix, D, D);
        for (uint32_t j = 0; j < D; j++) h2[j] = x[j] + g * mix[j];
        memcpy(y, h2, D * 4);
        snprintf(nm, 72, "b%u.ln2", l); const uint16_t* g2 = find(nm)->data;
        snprintf(nm, 72, "b%u.ln2.bias", l); const uint16_t* b2 = find(nm)->data;
        lnrow(y, g2, b2);
        uint32_t F = find("b0.fc1")->dims[1];
        snprintf(nm, 72, "b%u.fc1", l); mat_q(y, find(nm), f1, D, F);
        for (uint32_t j = 0; j < F; j++) f1[j] = gelu(f1[j]);
        snprintf(nm, 72, "b%u.fc2", l); mat_q(f1, find(nm), o, F, D);
        for (uint32_t j = 0; j < D; j++) x[j] = h2[j] + g * o[j];
    }
    TT++;
}
static void logits_of(const float* x, float* lg /*V*/) {
    static float z[1024];
    memcpy(z, x, D * 4);
    lnrow(z, find("lnf")->data, find("lnf.bias")->data);
    Tensor* et = find("Et");
    if (et->dt == 2) mat_q(z, et, lg, D, V);
    else mat_f32(z, et->data, lg, D, V);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: lc_stream m.lcw2 <bench N | ppl file.npy>\n"); return 1; }
    FILE* f = fopen(argv[1], "rb");
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
        bytes *= (t->dt == 2 ? 1 : 2);
        t->data = malloc(bytes); fread(t->data, 1, bytes, f);
    }
    fclose(f);
    if (find("Et")->dt != 2) {   // голова fp16 → fp32 один раз (int8-вариант не трогаем)
        Tensor* et = find("Et");
        size_t n = 1; for (uint32_t j = 0; j < et->nd; j++) n *= et->dims[j];
        const uint16_t* src = et->data; float* dst = malloc(n * 4);
        for (size_t i = 0; i < n; i++) dst[i] = f16(src[i]);
        free(et->data); et->data = dst; et->dt = 0;
    }
    for (uint32_t l = 0; l < L; l++) HSTATE[l] = malloc(D * 4);
    static float x[1024]; static float* lg = NULL; lg = malloc(V * 4);
    if (!strcmp(argv[2], "bench")) {
        int n = atoi(argv[3]); int cur = 2;
        struct timespec a, b;
        reset_states();
        clock_gettime(CLOCK_MONOTONIC, &a);
        for (int gg = 0; gg < n; gg++) {
            step(cur, x);
            logits_of(x, lg);
            int best = 1; float bl = lg[1];
            for (uint32_t j = 2; j < V; j++) if (lg[j] > bl) { bl = lg[j]; best = j; }
            cur = best;
        }
        clock_gettime(CLOCK_MONOTONIC, &b);
        double dt = (double)(b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec);
        printf("stream int8: %d tokens in %.3fs -> %.1f tok/s\n", n, dt, n / dt);
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
