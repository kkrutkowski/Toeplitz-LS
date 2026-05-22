#ifndef NANOFFT_TRIG_H
#define NANOFFT_TRIG_H

static inline void nanofft_triple_angle(FLOAT c, FLOAT s, FLOAT *c3, FLOAT *s3) {
    *c3 = SUB(MUL(FCAST(4.0), MUL(MUL(c, c), c)), MUL(FCAST(3.0), c));
    *s3 = SUB(MUL(FCAST(3.0), s), MUL(FCAST(4.0), MUL(MUL(s, s), s)));
}

#endif
