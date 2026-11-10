import json
import numpy as np
import buffer
import math
import os
import slangpy as spy
import trimesh

SPHERE = 0  # Must match Slang's ShapeType in hw3_common.slang
TRIANGLE_MESH = 1

def parse_triangle_mesh(filename):
    if filename.lower().endswith('.ply'):
        mesh = trimesh.load_mesh(filename, process=False)

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.uint32)
        vertex_normals = np.asarray(
            mesh.vertex_normals, dtype=np.float32)

    else:
        with open(filename) as f:
            mesh = json.load(f)

        vertices = np.array(mesh['vertices'], dtype=np.float32)
        faces = np.array(mesh['faces'], dtype=np.uint32)

        if 'vertex_normals' in mesh:
            vertex_normals = np.array(
                mesh['vertex_normals'], dtype=np.float32)
        else:
            # Generate smooth vertex normals.
            tri_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                process=False)
            vertex_normals = np.asarray(
                tri_mesh.vertex_normals, dtype=np.float32)

    return vertices, faces, vertex_normals

def compose_transformations(transformations):
    """
        Given a list of transformations, compose them into a matrix.
    """
    # HW 3.4: copy paste your HW2.3 code here

    F = np.eye(4, dtype=np.float32)

    # ===== HW 3.4 BEGIN (compose_transformations) =====
    def angle_of(value):
        # Rotation angles are stored like [30] (degrees); accept a bare number too
        if isinstance(value, (list, tuple)):
            value = value[0]
        return math.radians(value)

    for transform in transformations:
        M = np.eye(4, dtype=np.float32)

        if 'scale' in transform:
            sx, sy, sz = transform['scale']
            M[0, 0], M[1, 1], M[2, 2] = sx, sy, sz

        elif 'translate' in transform:
            M[:3, 3] = transform['translate']

        elif 'rotate_x' in transform:
            c, s = math.cos(angle_of(transform['rotate_x'])), math.sin(angle_of(transform['rotate_x']))
            M[1, 1], M[1, 2] = c, -s
            M[2, 1], M[2, 2] = s, c

        elif 'rotate_y' in transform:
            c, s = math.cos(angle_of(transform['rotate_y'])), math.sin(angle_of(transform['rotate_y']))
            M[0, 0], M[0, 2] = c, s
            M[2, 0], M[2, 2] = -s, c

        elif 'rotate_z' in transform:
            c, s = math.cos(angle_of(transform['rotate_z'])), math.sin(angle_of(transform['rotate_z']))
            M[0, 0], M[0, 1] = c, -s
            M[1, 0], M[1, 1] = s, c

        elif 'lookat' in transform:
            p = np.array(transform['lookat']['position'], dtype=np.float32)
            target = np.array(transform['lookat']['target'], dtype=np.float32)
            up = np.array(transform['lookat']['up'], dtype=np.float32)
            # Build an orthonormal frame: d = forward, r = right, u = true up
            d = (target - p) / np.linalg.norm(target - p)
            r = np.cross(d, up)
            r = r / np.linalg.norm(r)
            u = np.cross(r, d)
            # Columns map x -> r, y -> u, z -> -d, and translate to p
            M[:3, 0] = r
            M[:3, 1] = u
            M[:3, 2] = -d
            M[:3, 3] = p

        else:
            raise ValueError(f"Unknown transformation: {transform}")

        # Transforms apply in listed order, so each new one goes on the LEFT
        F = M @ F
    # ===== HW 3.4 END (compose_transformations) =====

    return F

def interpolate_transformation(transform_keyframes, t):
    # HW 3.4: copy your HW2.4 code here.

    # create a new transform from the two keyframes
    interp_transforms = []

    # ===== HW 3.4 BEGIN (interpolate_transformation) =====
    # Find the pair of keyframes [k0, k1] that surrounds time t
    i = 0
    while i < len(transform_keyframes) - 2 and t > transform_keyframes[i + 1]['time']:
        i += 1
    k0 = transform_keyframes[i]
    k1 = transform_keyframes[min(i + 1, len(transform_keyframes) - 1)]

    # How far t is between k0 and k1 (0 = at k0, 1 = at k1), clamped to [0, 1]
    dt = k1['time'] - k0['time']
    s = 0.0 if dt <= 0 else min(max((t - k0['time']) / dt, 0.0), 1.0)

    def lerp(a, b):
        return ((1 - s) * np.array(a, dtype=np.float64) +
                s * np.array(b, dtype=np.float64)).tolist()

    # Both keyframes have the same list of transforms in the same order,
    # so interpolate each transform's parameters pair by pair
    for tr0, tr1 in zip(k0['transform'], k1['transform']):
        name = next(iter(tr0))  # e.g. 'translate', 'rotate_y', 'lookat'
        if name == 'lookat':
            interp_transforms.append({'lookat': {
                key: lerp(tr0['lookat'][key], tr1['lookat'][key])
                for key in ('position', 'target', 'up')
            }})
        else:
            # Angles are lerped directly, so 0 -> 360 gives a full turn
            interp_transforms.append({name: lerp(tr0[name], tr1[name])})
    # ===== HW 3.4 END (interpolate_transformation) =====

    return interp_transforms

def get_transformation(obj, t):
    transforms = obj.get('transform', [])
    if 'transform_keyframes' in obj:
        transforms = interpolate_transformation(
            obj['transform_keyframes'], t)
    return compose_transformations(transforms)

def build_bvh(vertices, faces, triangle_offset, bvh_nodes):
    # A recursive BVH builder that split the longest axis
    # using the median of the centroids. The leaf nodes
    # have maximum of 4 triangles.

    assert(len(faces) > 0)
    # Needs a copy because we will sort them during the recursion
    faces = faces.copy()
    centers = vertices[faces].mean(axis=1)

    def build(begin, end):
        node_id = len(bvh_nodes)
        # Important: append a node first
        # before the recursion, so that the order
        # is "preorder" 
        bvh_nodes.append(None)

        # Bounding box of all triangles in this node
        points = vertices[faces[begin:end]].reshape(-1, 3)
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)

        # Stop recursion if the node has fewer than 4 triangles
        is_leaf = end - begin <= 4
        if not is_leaf:
            # Find the longest axis
            c = centers[begin:end]
            axis = np.argmax(c.max(axis=0) - c.min(axis=0))

            # Sort triangles by their centroids
            order = np.argsort(c[:, axis])
            faces[begin:end] = faces[begin:end][order]
            centers[begin:end] = c[order]

            # Split
            mid = (begin + end) // 2
            build(begin, mid)
            build(mid, end)

        bvh_nodes[node_id] = {
            'bbox_min': bbox_min,
            'bbox_max': bbox_max,
            'next_index': len(bvh_nodes),
            'triangle_begin': triangle_offset + begin,
            'triangle_end': triangle_offset + end,
            'is_leaf': is_leaf
        }

        return node_id

    root = build(0, len(faces))
    return root, faces

def upload_scene(filename, module, slang_device, use_bvh = False):
    with open(filename) as f:
        scene = json.load(f)

    background = scene.get('background', (0.5, 0.5, 0.5))

    camera = {'cam_to_world': np.eye(4, dtype=np.float32),
              'fov_x': 90.0}

    if 'camera' in scene:
        camera['cam_to_world'] = \
            get_transformation(scene['camera'], 0.0)
        camera['fov_x'] = scene['camera'].get('fov_x', 90.0)

    # Triangle meshes loading + BVH building
    mesh_descs = []
    vertices_list = []
    normals_list = []
    triangles_list = []
    bvh_nodes = []

    vertex_offset = 0
    triangle_offset = 0

    scene_dir = os.path.dirname(filename)

    for mesh_data in scene.get('meshes', []):
        mesh_filename = os.path.join(
            scene_dir, mesh_data['filename'])
        vertices, faces, vertex_normals = \
            parse_triangle_mesh(mesh_filename)

        # Construct the BVH and reorder triangles
        # The returned faces is a copy of our
        # original faces.
        if use_bvh:
            bvh_root_index, faces = build_bvh(
                vertices, faces, triangle_offset, bvh_nodes)
        else:
            bvh_root_index = 0

        mesh_descs.append({
            'vertex_offset': vertex_offset,
            'triangle_begin': triangle_offset,
            'triangle_end': triangle_offset + len(faces),
            'bvh_root_index': bvh_root_index
        })

        vertices_list.append(vertices)
        normals_list.append(vertex_normals)
        triangles_list.append(faces)

        vertex_offset += len(vertices)
        triangle_offset += len(faces)

    # Merge into global arrays
    if len(mesh_descs) > 0:
        vertices = np.concatenate(vertices_list).astype(np.float32)
        normals = np.concatenate(normals_list).astype(np.float32)
        triangles = np.concatenate(triangles_list).astype(np.uint32)
    else:
        # Use dummy entries if the scene has no meshes
        vertices = np.zeros((1, 3), dtype=np.float32)
        normals = np.zeros((1, 3), dtype=np.float32)
        triangles = np.zeros((1, 3), dtype=np.uint32)

    # Parse object instances.
    spheres = []
    shapes = []

    for obj in scene['objects']:
        if obj['type'] == 'sphere':
            shape_type = SPHERE
            shape_index = len(spheres)
            spheres.append({
                'center': obj['center'],
                'radius': obj['radius']
            })
        elif obj['type'] == 'mesh':
            shape_type = TRIANGLE_MESH
            shape_index = obj['index']
        else:
            raise ValueError(f"Unknown shape type: {obj['type']}")

        obj_to_world = get_transformation(obj, 0.0)
        world_to_obj = np.linalg.inv(obj_to_world)

        shapes.append({
            'type': shape_type,
            'index': shape_index,
            'Kd': obj['Kd'],
            'Kr': obj['Kr'],
            'world_to_obj': world_to_obj,
            'obj_to_world': obj_to_world
        })

    lights = scene['lights']

    max_depth = scene.get('max_depth', 20)

    bindings = {
        'background': spy.float3(background),
        'camera': camera,
        'shapes': buffer.create_structured_buffer(
            slang_device, module.Shape, shapes),
        'num_shapes': len(shapes),
        'spheres': buffer.create_structured_buffer(
            slang_device, module.Sphere, spheres),
        'meshes': buffer.create_structured_buffer(
            slang_device, module.TriangleMesh, mesh_descs),
        'vertices': buffer.structured_typed_buffer_from_numpy(
            slang_device, vertices, 'float3'),
        'vertex_normals': buffer.structured_typed_buffer_from_numpy(
            slang_device, normals, 'float3'),
        'triangles': buffer.structured_typed_buffer_from_numpy(
            slang_device, triangles, 'uint3'),
        'bvh_nodes': buffer.create_structured_buffer(
            slang_device, module.BVHNode, bvh_nodes),
        'lights': buffer.create_structured_buffer(
            slang_device, module.PointLight, lights),
        'num_lights': len(lights),
        'max_depth': max_depth
    }

    return bindings, scene, shapes
