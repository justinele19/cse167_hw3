import hw3_common;

// ===== HW 3.1 BEGIN (helper) =====
// Ray-sphere intersection: solve |o + t*d - c|^2 = r^2 for t.
// Returns the closest t in (t_min, t_max). Works with unnormalized d.
bool intersect_sphere(Sphere sphere, float3 ray_org, float3 ray_dir,
                      float t_min, float t_max, out float t_hit) {
    // Expand into a*t^2 + b*t + c = 0
    float3 oc = ray_org - sphere.center;
    float a = dot(ray_dir, ray_dir);
    float b = 2.0 * dot(oc, ray_dir);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;

    t_hit = t_max;
    float2 roots;
    if (!solve_quadratic(a, b, c, roots)) {
        return false; // ray misses the sphere
    }

    // Roots are not sorted: try the nearer one first, then the farther one
    float t0 = min(roots.x, roots.y);
    float t1 = max(roots.x, roots.y);
    if (t0 > t_min && t0 < t_max) {
        t_hit = t0;
        return true;
    }
    if (t1 > t_min && t1 < t_max) {
        t_hit = t1;
        return true;
    }
    return false;
}
// ===== HW 3.1 END (helper) =====

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Sphere> spheres,
                  uniform int num_spheres,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(background, 1.0);

    // ===== HW 3.1 BEGIN =====
    // Shoot a ray from the origin through the pixel center.
    // Image plane is at z = -1, x in [-a, a], y in [-1, 1].
    float aspect = float(width) / float(height);
    float u = (float(tid.x) + 0.5) / float(width);  // 0 = left, 1 = right
    float v = (float(tid.y) + 0.5) / float(height); // 0 = top, 1 = bottom
    float3 ray_org = float3(0.0, 0.0, 0.0);
    float3 ray_dir = float3((2.0 * u - 1.0) * aspect, 1.0 - 2.0 * v, -1.0);

    // Find the closest hit among all spheres.
    // Passing closest_t as t_max means only nearer hits are accepted.
    float closest_t = 1e30;
    int hit_id = -1;
    for (int i = 0; i < num_spheres; i++) {
        float t;
        if (intersect_sphere(spheres[i], ray_org, ray_dir, 0.0, closest_t, t)) {
            closest_t = t;
            hit_id = i;
        }
    }

    // On hit, visualize the normal: map [-1, 1] -> [0, 1]
    if (hit_id >= 0) {
        float3 p = ray_org + closest_t * ray_dir;
        float3 n = normalize(p - spheres[hit_id].center);
        image[tid] = float4((n + 1.0) / 2.0, 1.0);
    }
    // ===== HW 3.1 END =====
}
