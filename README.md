# CSE 167 Homework 3 Submission

## My scene ("Make your own scene")

- **Scene:** `scenes/hw3/my_scene/flyby.json`
- **Rendering:** `outputs/hw_3_7.png`

A low-poly spaceship flies across an alien planet, skimming low over a field of teal crystals. A warm "sun" casts the ship's shadow onto the ground and crystals, and a faint purple light adds a second color to the shading. A mirror moon hangs in the sky next to a plain purple planet. The ship is animated with keyframes: it enters from the left banked, levels out mid-screen, and climbs away to the right, looping every 6 seconds. The scene uses spheres, triangle meshes (rendered with the BVH from 3.6), shadows, mirror reflection, transformations, a look-at camera and animation.

Reproduce with:
```bash
uv run python main.py 3_6 scenes/hw3/my_scene/flyby.json                                          # animated
uv run python main.py -output outputs/hw_3_7.png 3_6 -time 3.0 scenes/hw3/my_scene/flyby.json   # still image
```

Files in `scenes/hw3/my_scene/`:
- `flyby.json`: the scene (camera, lights, objects, materials, keyframes)
- `ship_metal.ply`, `ship_metalDark.ply`, `ship_dark.ply`, `ship_metalRed.ply`: the spaceship, one mesh per material color. All four share the same keyframes, so they move as one ship.
- `CREDITS.txt`: model attribution

The crystals use `scenes/hw3/meshes/crystal_mesh.json`, provided with the homework.

**Model credit:** the spaceship is `craft_speederD` from [Space Kit by Kenney](https://kenney.nl/assets/space-kit) (www.kenney.nl), licensed CC0. Same files as my HW2 scene: converted from OBJ to PLY with trimesh, split into one PLY per material, centered and scaled, with flat-shaded normals.

## Bonus

None implemented.

## Notes

- **cube_sphere_sphere (3.5):** the blue "sphere" is a triangle mesh. Its interpolated vertex normals make the shading look smooth, but its outline and shadow are faceted, since the actual surface is made of flat triangles. The yellow sphere is a true sphere, so its outline and shadow are perfectly round.
- `hw3/hw3.py` picks the Metal GPU backend on macOS and Vulkan elsewhere, so it runs on Apple Silicon Macs.

## Required outputs

All graded renderings are in `outputs/` (`hw_3_1_*.png` through `hw_3_6_*.png`, plus `hw_3_7.png`).

---

# balboa
UCSD CSE 167 codebase
https://cseweb.ucsd.edu/~tzli/cse167/

This is a revised version of balboa that builds on [SlangPy](https://github.com/shader-slang/slangpy). Currently only Homework 1 and 2 are completed. Future homeworks will be added soon.

# Setup
Clone the repository:
```bash
git clone https://github.com/BachiLi/balboa_slang_public.git
cd balboa_slang_public
```

We use [uv](https://docs.astral.sh/uv/) to manage Python packages and environments. Go to the linked website to install it if you haven't. Once installed, run
```bash
uv sync
```

Then try
```bash
uv run python main.py 1_1
```

You should see a window with white background.

# Platforms

Balboa depends on [SlangPy](https://github.com/shader-slang/slangpy) as a GPU programming framework, and should work on Windows, Linux, and Apple Silicon Macs. 

**Intel Macs:** SlangPy does not currently publish macOS x86_64 wheels, so setup may require building SlangPy from source. If you are using an Intel Mac and `uv sync` fails because no compatible SlangPy package is available, you may need to build SlangPy from source. Contact the staff/instructor if you run into issues.
