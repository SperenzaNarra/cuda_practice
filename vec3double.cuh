#ifndef VEC3DOUBLE_CUH
#define VEC3DOUBLE_CUH

struct vec3double
{
    union {double x; double u; double r;};
    union {double y; double v; double g;};
    union {double z; double w; double b;};

    __host__ __device__ double length();
    __host__ __device__ double length2();
    __host__ __device__ vec3double normal();

    __host__ __device__ vec3double();
    __host__ __device__ vec3double(const vec3double &vec);
    __host__ __device__ vec3double(int i);
    __host__ __device__ vec3double(float f);
    __host__ __device__ vec3double(double d);
    
    __host__ __device__ vec3double(int x, int y, int z);
    __host__ __device__ vec3double(float x, float y, float z);
    __host__ __device__ vec3double(double x, double y, double z);

    __host__ __device__ vec3double(int x, int y, float z);
    __host__ __device__ vec3double(int x, float y, int z);
    __host__ __device__ vec3double(int x, float y, float z);
    __host__ __device__ vec3double(float x, int y, int z);
    __host__ __device__ vec3double(float x, int y, float z);
    __host__ __device__ vec3double(float x, float y, int z);

    __host__ __device__ vec3double(float x, float y, double z);
    __host__ __device__ vec3double(float x, double y, float z);
    __host__ __device__ vec3double(float x, double y, double z);
    __host__ __device__ vec3double(double x, float y, float z);
    __host__ __device__ vec3double(double x, float y, double z);
    __host__ __device__ vec3double(double x, double y, float z);

    __host__ __device__ vec3double(double x, double y, int z);
    __host__ __device__ vec3double(double x, int y, double z);
    __host__ __device__ vec3double(double x, int y, int z);
    __host__ __device__ vec3double(int x, double y, double z);
    __host__ __device__ vec3double(int x, double y, int z);
    __host__ __device__ vec3double(int x, int y, double z);

    __host__ __device__ vec3double operator-() const;

    __host__ __device__ vec3double& operator+=(vec3double vec);
    __host__ __device__ vec3double& operator+=(int i);
    __host__ __device__ vec3double& operator+=(float f);
    __host__ __device__ vec3double& operator+=(double d);

    __host__ __device__ vec3double& operator-=(vec3double vec);
    __host__ __device__ vec3double& operator-=(int i);
    __host__ __device__ vec3double& operator-=(float f);
    __host__ __device__ vec3double& operator-=(double d);

    __host__ __device__ vec3double& operator*=(vec3double vec);
    __host__ __device__ vec3double& operator*=(int i);
    __host__ __device__ vec3double& operator*=(float f);
    __host__ __device__ vec3double& operator*=(double d);

    __host__ __device__ vec3double& operator/=(vec3double vec);
    __host__ __device__ vec3double& operator/=(int i);
    __host__ __device__ vec3double& operator/=(float f);
    __host__ __device__ vec3double& operator/=(double d);
};

__host__ __device__ double dot(vec3double vec1, vec3double vec2);
__host__ __device__ vec3double cross(vec3double vec1, vec3double vec2);

__host__ __device__ vec3double operator+(vec3double vec1, vec3double vec2);
__host__ __device__ vec3double operator+(vec3double vec, int i);
__host__ __device__ vec3double operator+(vec3double vec, float f);
__host__ __device__ vec3double operator+(vec3double vec, double d);
__host__ __device__ vec3double operator+(int i, vec3double vec);
__host__ __device__ vec3double operator+(float f, vec3double vec);
__host__ __device__ vec3double operator+(double d, vec3double vec);

__host__ __device__ vec3double operator-(vec3double vec1, vec3double vec2);
__host__ __device__ vec3double operator-(vec3double vec, int i);
__host__ __device__ vec3double operator-(vec3double vec, float f);
__host__ __device__ vec3double operator-(vec3double vec, double d);
__host__ __device__ vec3double operator-(int i, vec3double vec);
__host__ __device__ vec3double operator-(float f, vec3double vec);
__host__ __device__ vec3double operator-(double d, vec3double vec);

__host__ __device__ vec3double operator*(vec3double vec1, vec3double vec2);
__host__ __device__ vec3double operator*(vec3double vec, int i);
__host__ __device__ vec3double operator*(vec3double vec, float f);
__host__ __device__ vec3double operator*(vec3double vec, double d);
__host__ __device__ vec3double operator*(int i, vec3double vec);
__host__ __device__ vec3double operator*(float f, vec3double vec);
__host__ __device__ vec3double operator*(double d, vec3double vec);

__host__ __device__ vec3double operator/(vec3double vec1, vec3double vec2);
__host__ __device__ vec3double operator/(vec3double vec, int i);
__host__ __device__ vec3double operator/(vec3double vec, float f);
__host__ __device__ vec3double operator/(vec3double vec, double d);
__host__ __device__ vec3double operator/(int i, vec3double vec);
__host__ __device__ vec3double operator/(float f, vec3double vec);
__host__ __device__ vec3double operator/(double d, vec3double vec);

#endif // VEC3DOUBLE_CUH