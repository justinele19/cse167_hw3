import hw3_common;

// ===== HW 3.3 BEGIN (helpers) =====
// Ray-sphere intersection (same as 3.1).
// Returns the closest t in (t_min, t_max). Works with unnormalized ray_dir.
bool intersect_sphere(Sphere sphere, float3 ray_org, float3 ray_dir,
                      float t_min, float t_max, out float t_hit) {
    float3 oc = ray_org - sphere.center;
    float a = dot(ray_dir, ray_dir);
    float b = 2.0 * dot(oc, ray_dir);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;

    t_hit = t_max;
    float2 roots;
    if (!solve_quadratic(a, b, c, roots)) {
        return false;
    }
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

// Find the closest shape the ray hits in (t_min, t_max) (same as 3.2).
// Returns the shape index, or -1 if nothing is hit.
int intersect_scene(StructuredBuffer<Shape> shapes, int num_shapes,
                    StructuredBuffer<Sphere> spheres,
                    float3 ray_org, float3 ray_dir,
                    float t_min, float t_max, out float t_hit) {
    int hit_id = -1;
    t_hit = t_max;
    for (int i = 0; i < num_shapes; i++) {
        if (shapes[i].type != ShapeType.Sphere) {
            continue;
        }
        float t;
        if (intersect_sphere(spheres[shapes[i].index], ray_org, ray_dir, t_min, t_hit, t)) {
            t_hit = t;
            hit_id = i;
        }
    }
    return hit_id;
}
// ===== HW 3.3 END (helpers) =====

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<PointLight> lights,
                  uniform int num_lights,
                  uniform int max_depth,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(background, 1.0);

    // ===== HW 3.3 BEGIN =====
    // Camera ray through the pixel center (same setup as 3.1)
    float aspect = float(width) / float(height);
    float u = (float(tid.x) + 0.5) / float(width);
    float v = (float(tid.y) + 0.5) / float(height);
    float3 ray_org = float3(0.0, 0.0, 0.0);
    float3 ray_dir = float3((2.0 * u - 1.0) * aspect, 1.0 - 2.0 * v, -1.0);

    // Recursion rewritten as a loop.
    // throughput = how much of the current bounce reaches the camera
    // (product of the Kr values of all the mirrors hit so far).
    float3 color = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    for (int b = 0; b < max_depth; b++) {
        float t;
        int hit_id = intersect_scene(shapes, num_shapes, spheres,
                                     ray_org, ray_dir, EPSILON, 1e30, t);
        if (hit_id < 0) {
            // Ray escaped (camera ray or mirror ray): add the background
            color += throughput * background;
            break;
        }

        Shape shape = shapes[hit_id];
        Sphere sphere = spheres[shape.index];
        float3 p = ray_org + t * ray_dir;
        float3 n = normalize(p - sphere.center);

        // Direct lighting with shadows (same as 3.2)
        float3 contribution = float3(0.0, 0.0, 0.0);
        for (int i = 0; i < num_lights; i++) {
            float3 to_light = lights[i].position - p;
            float dist = length(to_light);
            float3 l = to_light / dist;

            float t_shadow;
            if (intersect_scene(shapes, num_shapes, spheres,
                                p, l, EPSILON, dist - EPSILON, t_shadow) >= 0) {
                continue; // in shadow
            }
            contribution += shape.Kd * lights[i].intensity * max(dot(n, l), 0.0) / (dist * dist);
        }
        color += throughput * contribution;

        // Next bounce is weighted by this surface's mirror color
        throughput *= shape.Kr;
        if (all(throughput == float3(0.0, 0.0, 0.0))) {
            break; // not a mirror: nothing more can be added
        }

        // Continue from the hit point in the mirror direction
        // (EPSILON as t_min in the next loop avoids self intersection)
        ray_org = p;
        ray_dir = reflect(ray_dir, n);
    }
    image[tid] = float4(color, 1.0);
    // ===== HW 3.3 END =====
}
