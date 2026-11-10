import hw3_common;

// ===== HW 3.6 BEGIN (helpers) =====
// Everything here is the same as 3.5, except intersect_mesh, which now walks
// a BVH (a tree of boxes around groups of triangles) instead of checking
// every triangle.

// Ray-sphere intersection (same as 3.5).
// Finds the closest point where the ray hits the sphere, between t_min and t_max.
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

// Ray-mesh intersection using the mesh's BVH.
// The tree is stored as a flat list (parent first, then its children).
// If the ray misses a node's box, we jump to next_index and skip everything
// inside that box. If it hits, we step to the next node in the list, which
// is the node's first child. Leaves hold a few triangles that we test directly.
bool intersect_mesh(TriangleMesh mesh,
                    StructuredBuffer<float3> vertices,
                    StructuredBuffer<uint3> triangles,
                    StructuredBuffer<BVHNode> bvh_nodes,
                    float3 ray_org, float3 ray_dir,
                    float t_min, float t_max,
                    out float t_hit, out uint hit_tri, out float3 hit_bary) {
    t_hit = t_max;
    hit_tri = 0;
    hit_bary = float3(0.0, 0.0, 0.0);
    bool hit = false;

    // Start at this mesh's root; the root's next_index marks the end of its tree
    uint node_id = mesh.bvh_root_index;
    uint end_id = uint(bvh_nodes[node_id].next_index);

    while (node_id < end_id) {
        BVHNode node = bvh_nodes[node_id];

        // Missed the box (or it's farther than our closest hit): skip its whole subtree
        if (!intersect_aabb(node.bbox_min, node.bbox_max,
                            ray_org, ray_dir, t_min, t_hit)) {
            node_id = uint(node.next_index);
            continue;
        }

        if (node.is_leaf) {
            // Leaf: test its few triangles, keeping the closest hit (same as 3.5)
            for (uint j = node.triangle_begin; j < node.triangle_end; j++) {
                uint3 tri = triangles[j] + mesh.vertex_offset;
                float3 b;
                float t;
                if (intersect_tri(vertices[tri.x], vertices[tri.y], vertices[tri.z],
                                  ray_org, ray_dir, t_min, t_hit, b, t)) {
                    t_hit = t;
                    hit_tri = j;
                    hit_bary = b;
                    hit = true;
                }
            }
            node_id = uint(node.next_index);
        } else {
            // Inner node we hit: go down into its first child
            node_id = node_id + 1;
        }
    }
    return hit;
}

// Find the closest shape (sphere or mesh) hit by a world-space ray (same as 3.5,
// just passing the BVH along). Returns the shape index, or -1 if nothing is hit.
int intersect_scene(StructuredBuffer<Shape> shapes, int num_shapes,
                    StructuredBuffer<Sphere> spheres,
                    StructuredBuffer<TriangleMesh> meshes,
                    StructuredBuffer<float3> vertices,
                    StructuredBuffer<uint3> triangles,
                    StructuredBuffer<BVHNode> bvh_nodes,
                    float3 ray_org, float3 ray_dir,
                    float t_min, float t_max,
                    out float t_hit, out uint hit_tri, out float3 hit_bary) {
    int hit_id = -1;
    t_hit = t_max;
    hit_tri = 0;
    hit_bary = float3(0.0, 0.0, 0.0);

    for (int i = 0; i < num_shapes; i++) {
        // Move the ray into this shape's own space (direction not normalized,
        // so t means the same thing for every object)
        float3 obj_org = xform_pt(shapes[i].world_to_obj, ray_org);
        float3 obj_dir = xform_vec(shapes[i].world_to_obj, ray_dir);

        if (shapes[i].type == ShapeType.Sphere) {
            float t;
            if (intersect_sphere(spheres[shapes[i].index], obj_org, obj_dir, t_min, t_hit, t)) {
                t_hit = t;
                hit_id = i;
            }
        } else if (shapes[i].type == ShapeType.TriangleMesh) {
            float t;
            uint tri;
            float3 bary;
            if (intersect_mesh(meshes[shapes[i].index], vertices, triangles, bvh_nodes,
                               obj_org, obj_dir, t_min, t_hit, t, tri, bary)) {
                t_hit = t;
                hit_id = i;
                hit_tri = tri;
                hit_bary = bary;
            }
        }
    }
    return hit_id;
}

// World-space unit normal of a transformed sphere at world point p (same as 3.5).
float3 sphere_normal(Shape shape, Sphere sphere, float3 p) {
    float3 n_obj = xform_pt(shape.world_to_obj, p) - sphere.center;
    return normalize(xform_vec(transpose(shape.world_to_obj), n_obj));
}

// World-space unit normal at the hit point, for either shape type (same as 3.5).
// For meshes we blend the three corner normals so the surface looks smooth.
float3 surface_normal(Shape shape,
                      StructuredBuffer<Sphere> spheres,
                      StructuredBuffer<TriangleMesh> meshes,
                      StructuredBuffer<float3> vertex_normals,
                      StructuredBuffer<uint3> triangles,
                      float3 p, uint hit_tri, float3 hit_bary) {
    if (shape.type == ShapeType.Sphere) {
        return sphere_normal(shape, spheres[shape.index], p);
    }
    uint3 tri = triangles[hit_tri] + meshes[shape.index].vertex_offset;
    float3 n_obj = hit_bary.x * vertex_normals[tri.x] +
                   hit_bary.y * vertex_normals[tri.y] +
                   hit_bary.z * vertex_normals[tri.z];
    return normalize(xform_vec(transpose(shape.world_to_obj), n_obj));
}
// ===== HW 3.6 END (helpers) =====

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform Camera camera,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<TriangleMesh> meshes,
                  StructuredBuffer<float3> vertices,
                  StructuredBuffer<float3> vertex_normals,
                  StructuredBuffer<uint3> triangles,
                  StructuredBuffer<BVHNode> bvh_nodes,
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

    // ===== HW 3.6 BEGIN =====
    // Camera ray through the pixel center (same as 3.4/3.5)
    float aspect = float(width) / float(height);
    float plane_w = 2.0 * tan(radians(camera.fov_x) / 2.0);
    float plane_h = plane_w / aspect;
    float u = (float(tid.x) + 0.5) / float(width);
    float v = (float(tid.y) + 0.5) / float(height);
    float3 cam_dir = float3((u - 0.5) * plane_w, (0.5 - v) * plane_h, -1.0);

    // Move the ray from camera space to world space
    float3 ray_org = xform_pt(camera.cam_to_world, float3(0.0, 0.0, 0.0));
    float3 ray_dir = xform_vec(camera.cam_to_world, cam_dir);

    // Mirror bounce loop (same as 3.5).
    // throughput = how much of each bounce still reaches the camera.
    float3 color = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    for (int b = 0; b < max_depth; b++) {
        // Find the closest thing this ray hits
        float t;
        uint hit_tri;
        float3 hit_bary;
        int hit_id = intersect_scene(shapes, num_shapes, spheres, meshes, vertices, triangles,
                                     bvh_nodes, ray_org, ray_dir, EPSILON, 1e30,
                                     t, hit_tri, hit_bary);
        if (hit_id < 0) {
            // Ray flew off into the background
            color += throughput * background;
            break;
        }

        // Hit point and its surface normal
        Shape shape = shapes[hit_id];
        float3 p = ray_org + t * ray_dir;
        float3 n = surface_normal(shape, spheres, meshes, vertex_normals, triangles,
                                  p, hit_tri, hit_bary);

        // Diffuse lighting from each point light, skipping lights that are blocked
        float3 contribution = float3(0.0, 0.0, 0.0);
        for (int i = 0; i < num_lights; i++) {
            float3 to_light = lights[i].position - p;
            float dist = length(to_light);
            float3 l = to_light / dist;

            // Shadow ray: is anything between p and the light?
            float t_shadow;
            uint shadow_tri;
            float3 shadow_bary;
            if (intersect_scene(shapes, num_shapes, spheres, meshes, vertices, triangles,
                                bvh_nodes, p, l, EPSILON, dist - EPSILON,
                                t_shadow, shadow_tri, shadow_bary) >= 0) {
                continue; // in shadow
            }
            contribution += shape.Kd * lights[i].intensity * max(dot(n, l), 0.0) / (dist * dist);
        }
        color += throughput * contribution;

        // Next bounce is weighted by this surface's mirror color; stop if it isn't a mirror
        throughput *= shape.Kr;
        if (all(throughput == float3(0.0, 0.0, 0.0))) {
            break;
        }
        ray_org = p;
        ray_dir = reflect(ray_dir, n);
    }
    image[tid] = float4(color, 1.0);
    // ===== HW 3.6 END =====
}
