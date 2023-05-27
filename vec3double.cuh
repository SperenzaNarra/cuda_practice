#ifndef VEC3DOUBLE_CUH
#define VEC3DOUBLE_CUH

struct vec3double
{
    union {double x; double u; double r;};
    union {double y; double v; double g;};
    union {double z; double w; double b;};

    __host__ __device__ double length();
    __host__ __device__ double length2();
    __host__ __device__ vec3double normalized();

    __host__ __device__ vec3double();
    __host__ __device__ vec3double(const vec3double &vec);
    __host__ __device__ vec3double(double d);
    __host__ __device__ vec3double(double x, double y, double z);

    __host__ __device__ vec3double operator-() const;

    __host__ __device__ vec3double& operator+=(vec3double vec);

    __host__ __device__ vec3double& operator-=(vec3double vec);

    __host__ __device__ vec3double& operator*=(vec3double vec);
    __host__ __device__ vec3double& operator*=(double d);

    __host__ __device__ vec3double& operator/=(vec3double vec);
    __host__ __device__ vec3double& operator/=(double d);
};

__host__ __device__ double dot(vec3double vec1, vec3double vec2);
__host__ __device__ vec3double cross(vec3double vec1, vec3double vec2);

__host__ __device__ vec3double operator+(vec3double vec1, vec3double vec2);

__host__ __device__ vec3double operator-(vec3double vec1, vec3double vec2);

__host__ __device__ vec3double operator*(vec3double vec1, vec3double vec2);
__host__ __device__ vec3double operator*(vec3double vec, double d);
__host__ __device__ vec3double operator*(double d, vec3double vec);

__host__ __device__ vec3double operator/(vec3double vec1, vec3double vec2);
__host__ __device__ vec3double operator/(vec3double vec, double d);

#endif // VEC3DOUBLE_CUH