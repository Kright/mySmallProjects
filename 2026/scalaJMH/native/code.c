
/**
 a - array of 16 doubles, which represents a 4x4 matrix
 b - array of 16 doubles, which represents a 4x4 matrix
 result - array of 16 doubles, result of multiplication
 */
void matrix4x4_multiply(const double * restrict a, const double * restrict b, double * restrict result) {
    for (int row = 0; row < 4; row++) {
        for (int column = 0; column < 4; column++) {
            double sum = 0.0;
            for (int i = 0; i < 4; i++) {
                sum += a[row * 4 + i] * b[i * 4 + column];
            }
            result[row * 4 + column] = sum;
        }
    }
}

double getDoubleZero() {
    return 0.0;
}

double callFunction(double (*callback)()) {
    return callback();
}
