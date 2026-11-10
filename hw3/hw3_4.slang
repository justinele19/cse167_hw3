import hw3_common;

// ===== HW 3.4 BEGIN (helpers) =====
// Ray-sphere intersection (same as 3.1).
// Returns the closest t in (t_min, t_max). Works with unnormalized ray_dir,
// which matters here: object-space rays are NOT normalized.
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

// Find the closest shape hit by a WORLD-space ray in (t_min, t_max).
// Each sphere is tested in its own object space. Because we don't normalize
// the transformed direction, t is the same in object and world space,
// so t values from different objects can be compared directly.
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
        // World -> object space (point uses w=1, direction uses w=0)
        float3 obj_org = xform_pt(shapes[i].world_to_obj, ray_org);
        float3 obj_dir = xform_vec(shapes[i].world_to_obj, ray_dir);

        float t;
        if (intersect_sphere(spheres[shapes[i].index], obj_org, obj_dir, t_min, t_hit, t)) {
            t_hit = t;
            hit_id = i;
        }
    }
    return hit_id;
}

// World-space unit normal of a transformed sphere at world point p.
float3 sphere_normal(Shape shape, Sphere sphere, float3 p) {
    // Normal in object space
    float3 n_obj = xform_pt(shape.world_to_obj, p) - sphere.center;
    // Normals use the inverse transpose of obj_to_world,
    // which is just transpose(world_to_obj)
    return normalize(xform_vec(transpose(shape.world_to_obj), n_obj));
}
// ===== HW 3.4 END (helpers) =====

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform Camera camera,
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

    // ===== HW 3.4 BEGIN =====
    // Image plane at z = -1 (focal length 1).
    // Width comes from the horizontal FOV, height from the aspect ratio.
    float aspect = float(width) / float(height);
    float plane_w = 2.0 * tan(radians(camera.fov_x) / 2.0);
    float plane_h = plane_w / aspect;
    float u = (float(tid.x) + 0.5) / float(width);
    float v = (float(tid.y) + 0.5) / float(height);
    float3 cam_dir = float3((u - 0.5) * plane_w, (0.5 - v) * plane_h, -1.0);

    // Camera space -> world space
    float3 ray_org = xform_pt(camera.cam_to_world, float3(0.0, 0.0, 0.0));
    float3 ray_dir = xform_vec(camera.cam_to_world, cam_dir);

    // Same mirror-bounce loop as 3.3, everything below is in world space
    float3 color = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    for (int b = 0; b < max_depth; b++) {
        float t;
        int hit_id = intersect_scene(shapes, num_shapes, spheres,
                                     ray_org, ray_dir, EPSILON, 1e30, t);
        if (hit_id < 0) {
            color += throughput * background;
            break;
        }

        Shape shape = shapes[hit_id];
        Sphere sphere = spheres[shape.index];
        // t is shared between spaces, so the world-space hit point is just o + t*d
        float3 p = ray_org + t * ray_dir;
        float3 n = sphere_normal(shape, sphere, p);

        // Direct lighting with shadows
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

        throughput *= shape.Kr;
        if (all(throughput == float3(0.0, 0.0, 0.0))) {
            break;
        }
        ray_org = p;
        ray_dir = reflect(ray_dir, n);
    }
    image[tid] = float4(color, 1.0);
    // ===== HW 3.4 END =====
}
