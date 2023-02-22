#pragma once

// just the methods declarations
class Vector
{
public:
    float x;
    float y;

    __device__ __host__ Vector(float x, float y);
    __device__ __host__ Vector(int x, int y);
    __device__ __host__ Vector(double x, double y);
    __device__ __host__ Vector();
    __device__ __host__ Vector operator+(const Vector& v);
    __device__ __host__ Vector operator-(const Vector& v);
    __device__ __host__ Vector operator-();
    __device__ __host__ Vector operator*(float f);
    __device__ __host__ Vector operator/(float f);
    __device__ __host__ Vector operator=(const Vector& v);
    __device__ __host__ Vector operator+=(const Vector& v);
};
