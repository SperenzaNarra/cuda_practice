#include "vec3double.cuh"
#include <math.h>


vec3double::vec3double(const vec3double &vec) : x(vec.x), y(vec.y), z(vec.z) {}

vec3double::vec3double()        : x(0), y(0), z(0) {}
vec3double::vec3double(int i)   : x(i), y(i), z(i) {}
vec3double::vec3double(float f) : x(f), y(f), z(f) {}
vec3double::vec3double(double d): x(d), y(d), z(d) {}

double vec3double::length2()
{
    return x*x + y*y + z*z;
}

double vec3double::length()
{
    return sqrt(length2());
}

vec3double vec3double::normal()
{
    return *this / length();
}

__host__ __device__ double dot(vec3double vec1, vec3double vec2)
{
    return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}

__host__ __device__ vec3double cross(vec3double vec1, vec3double vec2)
{
    return vec3double
    (
        vec1.y * vec2.z - vec1.z * vec2.y,
        vec1.z * vec2.x - vec1.x * vec2.z,
        vec1.x * vec2.y - vec1.y * vec2.x
    );
}

vec3double::vec3double(int x, int y, int z)         : x(x), y(y), z(z) {}
vec3double::vec3double(float x, float y, float z)   : x(x), y(y), z(z) {}
vec3double::vec3double(double x, double y, double z): x(x), y(y), z(z) {}

vec3double::vec3double(int x, int y, float z)   : x(x), y(y), z(z) {}
vec3double::vec3double(int x, float y, int z)   : x(x), y(y), z(z) {}
vec3double::vec3double(int x, float y, float z) : x(x), y(y), z(z) {}
vec3double::vec3double(float x, int y, int z)   : x(x), y(y), z(z) {}
vec3double::vec3double(float x, int y, float z) : x(x), y(y), z(z) {}
vec3double::vec3double(float x, float y, int z) : x(x), y(y), z(z) {}

vec3double::vec3double(float x, float y, double z)  : x(x), y(y), z(z) {}
vec3double::vec3double(float x, double y, float z)  : x(x), y(y), z(z) {}
vec3double::vec3double(float x, double y, double z) : x(x), y(y), z(z) {}
vec3double::vec3double(double x, float y, float z)  : x(x), y(y), z(z) {}
vec3double::vec3double(double x, float y, double z) : x(x), y(y), z(z) {}
vec3double::vec3double(double x, double y, float z) : x(x), y(y), z(z) {}

vec3double::vec3double(double x, double y, int z)   : x(x), y(y), z(z) {}
vec3double::vec3double(double x, int y, double z)   : x(x), y(y), z(z) {}
vec3double::vec3double(double x, int y, int z)      : x(x), y(y), z(z) {}
vec3double::vec3double(int x, double y, double z)   : x(x), y(y), z(z) {}
vec3double::vec3double(int x, double y, int z)      : x(x), y(y), z(z) {}
vec3double::vec3double(int x, int y, double z)      : x(x), y(y), z(z) {}

vec3double vec3double::operator-() const { return vec3double(-x, -y, -z); }

vec3double& vec3double::operator+=(vec3double vec) { x += vec.x; y += vec.y; z += vec.z; return *this; }
vec3double& vec3double::operator+=(int i)   { x += (double)i; y += (double)i; z += (double)i; return *this; }
vec3double& vec3double::operator+=(float f) { x += (double)f; y += (double)f; z += (double)f; return *this; }
vec3double& vec3double::operator+=(double d){ x += d; y += d; z += d; return *this; }

vec3double& vec3double::operator-=(vec3double vec) { x -= vec.x; y -= vec.y; z -= vec.z; return *this; }
vec3double& vec3double::operator-=(int i)   { x -= (double)i; y -= (double)i; z -= (double)i; return *this; }
vec3double& vec3double::operator-=(float f) { x -= (double)f; y -= (double)f; z -= (double)f; return *this; }
vec3double& vec3double::operator-=(double d){ x -= d; y -= d; z -= d; return *this; }

vec3double& vec3double::operator*=(vec3double vec) { x *= vec.x; y *= vec.y; z *= vec.z; return *this; }
vec3double& vec3double::operator*=(int i)   { x *= (double)i; y *= (double)i; z *= (double)i; return *this; }
vec3double& vec3double::operator*=(float f) { x *= (double)f; y *= (double)f; z *= (double)f; return *this; }
vec3double& vec3double::operator*=(double d){ x *= d; y *= d; z *= d; return *this; }

vec3double& vec3double::operator/=(vec3double vec) { x /= vec.x; y /= vec.y; z /= vec.z; return *this; }
vec3double& vec3double::operator/=(int i)   { x /= (double)i; y /= (double)i; z /= (double)i; return *this; }
vec3double& vec3double::operator/=(float f) { x /= (double)f; y /= (double)f; z /= (double)f; return *this; }
vec3double& vec3double::operator/=(double d){ x /= d; y /= d; z /= d; return *this; }

__host__ __device__ vec3double operator+(vec3double vec1, vec3double vec2) { return vec3double(vec1.x+vec2.x, vec1.y+vec2.y, vec1.z+vec2.z); }
__host__ __device__ vec3double operator+(vec3double vec, int i)     { return vec3double(vec.x+i, vec.y+i, vec.z+i); }
__host__ __device__ vec3double operator+(vec3double vec, float f)   { return vec3double(vec.x+f, vec.y+f, vec.z+f); }
__host__ __device__ vec3double operator+(vec3double vec, double d)  { return vec3double(vec.x+d, vec.y+d, vec.z+d); }
__host__ __device__ vec3double operator+(int i, vec3double vec)     { return vec3double(vec.x+i, vec.y+i, vec.z+i); }
__host__ __device__ vec3double operator+(float f, vec3double vec)   { return vec3double(vec.x+f, vec.y+f, vec.z+f); }
__host__ __device__ vec3double operator+(double d, vec3double vec)  { return vec3double(vec.x+d, vec.y+d, vec.z+d); }

__host__ __device__ vec3double operator-(vec3double vec1, vec3double vec2) { return vec3double(vec1.x-vec2.x, vec1.y-vec2.y, vec1.z-vec2.z); }
__host__ __device__ vec3double operator-(vec3double vec, int i)     { return vec3double(        vec.x-i,            vec.y-i,            vec.z-i); }
__host__ __device__ vec3double operator-(vec3double vec, float f)   { return vec3double(        vec.x-f,            vec.y-f,            vec.z-f); }
__host__ __device__ vec3double operator-(vec3double vec, double d)  { return vec3double(        vec.x-d,            vec.y-d,            vec.z-d); }
__host__ __device__ vec3double operator-(int i, vec3double vec)     { return vec3double((double)i-vec.x, (double)   i-vec.y, (double)   i-vec.z); }
__host__ __device__ vec3double operator-(float f, vec3double vec)   { return vec3double((double)f-vec.x, (double)   f-vec.y, (double)   f-vec.z); }
__host__ __device__ vec3double operator-(double d, vec3double vec)  { return vec3double(        d-vec.x,            d-vec.y,            d-vec.z); }

__host__ __device__ vec3double operator*(vec3double vec1, vec3double vec2) { return vec3double(vec1.x*vec2.x, vec1.y*vec2.y, vec1.z*vec2.z); }
__host__ __device__ vec3double operator*(vec3double vec, int i)     { return vec3double(vec.x*i, vec.y*i, vec.z*i); }
__host__ __device__ vec3double operator*(vec3double vec, float f)   { return vec3double(vec.x*f, vec.y*f, vec.z*f); }
__host__ __device__ vec3double operator*(vec3double vec, double d)  { return vec3double(vec.x*d, vec.y*d, vec.z*d); }
__host__ __device__ vec3double operator*(int i, vec3double vec)     { return vec3double(vec.x*i, vec.y*i, vec.z*i); }
__host__ __device__ vec3double operator*(float f, vec3double vec)   { return vec3double(vec.x*f, vec.y*f, vec.z*f); }
__host__ __device__ vec3double operator*(double d, vec3double vec)  { return vec3double(vec.x*d, vec.y*d, vec.z*d); }

__host__ __device__ vec3double operator/(vec3double vec1, vec3double vec2) { return vec3double(vec1.x/vec2.x, vec1.y/vec2.y, vec1.z/vec2.z); }
__host__ __device__ vec3double operator/(vec3double vec, int i)     { return vec3double(        vec.x/i,            vec.y/i,            vec.z/i); }
__host__ __device__ vec3double operator/(vec3double vec, float f)   { return vec3double(        vec.x/f,            vec.y/f,            vec.z/f); }
__host__ __device__ vec3double operator/(vec3double vec, double d)  { return vec3double(        vec.x/d,            vec.y/d,            vec.z/d); }
__host__ __device__ vec3double operator/(int i, vec3double vec)     { return vec3double((double)i/vec.x, (double)   i/vec.y, (double)   i/vec.z); }
__host__ __device__ vec3double operator/(float f, vec3double vec)   { return vec3double((double)f/vec.x, (double)   f/vec.y, (double)   f/vec.z); }
__host__ __device__ vec3double operator/(double d, vec3double vec)  { return vec3double(        d/vec.x,            d/vec.y,            d/vec.z); }