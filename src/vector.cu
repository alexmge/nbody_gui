#include "vector.cuh"

__device__ __host__ Vector::Vector() {
    x = 0;
    y = 0;
}

__device__ __host__ Vector::Vector(float x, float y) {
    this->x = x;
    this->y = y;
}

__device__ __host__ Vector::Vector(int x, int y) {
    this->x = (float) x;
    this->y = (float) y;
}

__device__ __host__ Vector::Vector(double x, double y) {
    this->x = (float) x;
    this->y = (float) y;
}

__device__ __host__ Vector Vector::operator+(const Vector &v){
    return Vector(x + v.x, y + v.y);
}

__device__ __host__ Vector Vector::operator-(const Vector &v){
    return Vector(x - v.x, y - v.y);
}

__device__ __host__ Vector Vector::operator*(float s) {
    return Vector(x * s, y * s);
}

__device__ __host__ Vector Vector::operator/(float s) {
    return Vector(x / s, y / s);
}

__device__ __host__ Vector Vector::operator-() {
    return Vector(-x, -y);
}

__device__ __host__ Vector Vector::operator+=(const Vector &v) {
    x += v.x;
    y += v.y;
    return *this;
}

__device__ __host__ Vector Vector::operator=(const Vector &v) {
    x = v.x;
    y = v.y;
    return *this;
}