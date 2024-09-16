#modified from example files for procedural terrain generation and legged robots bundled with IsaacLab
"""
FLAGS: 

# Generate terrain with height color scheme
    ./isaaclab.sh -p source/standalone/demos/procedural_terrain.py --color_scheme height

    # Generate terrain with random color scheme
    ./isaaclab.sh -p source/standalone/demos/procedural_terrain.py --color_scheme random

    # Generate terrain with no color scheme
    ./isaaclab.sh -p source/standalone/demos/procedural_terrain.py --color_scheme none

    # Generate terrain with curriculum
    ./isaaclab.sh -p source/standalone/demos/procedural_terrain.py --use_curriculum

    # Generate terrain with curriculum along with flat patches
    ./isaaclab.sh -p source/standalone/demos/procedural_terrain.py --use_curriculum --show_flat_patches
"""
import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script tests legged robots on varying terrain.")
#add terrain config
parser.add_argument(
    "--color_scheme",
    type=str,
    default="none",
    choices=["height", "random", "none"],
    help="Color scheme to use for the terrain generation.",
)
parser.add_argument(
    "--use_curriculum",
    action="store_true",
    default=False,
    help="Whether to use the curriculum for the terrain generation.",
)

parser.add_argument(
    "--show_flat_patches",
    action="store_true",
    default=False,
    help="Whether to show the flat patches computed during the terrain generation.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

#========================================================================================================================

#general
import numpy as np
import random
import torch

#sim and robot control
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation

#terrain generation
from omni.isaac.lab.assets import AssetBase
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.terrains import FlatPatchSamplingCfg, TerrainImporter, TerrainImporterCfg

##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort:skip
from omni.isaac.lab_assets.spot import SPOT_CFG # isort:skip

#=======================================================User Params======================================================
num_terrain_entities = 1
num_robots = 10
#========================================================================================================================
def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_cols = np.floor(np.sqrt(num_origins))
    num_rows = np.ceil(num_origins / num_cols)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0
    # return the origins
    return env_origins.tolist()

def design_scene() -> tuple[dict, torch.Tensor]:
    """Designs the scene."""
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
#======================================================Terrain Handling======================================================#
    # Parse terrain generation
    terrain_gen_cfg = ROUGH_TERRAINS_CFG.replace(curriculum=args_cli.use_curriculum, color_scheme=args_cli.color_scheme)

    # Add flat patch configuration
    # Note: To have separate colors for each sub-terrain type, we set the flat patch sampling configuration name
    #   to the sub-terrain name. However, this is not how it should be used in practice. The key name should be
    #   the intention of the flat patch. For instance, "source" or "target" for spawn and command related flat patches.
    if args_cli.show_flat_patches:
        for sub_terrain_name, sub_terrain_cfg in terrain_gen_cfg.sub_terrains.items():
            sub_terrain_cfg.flat_patch_sampling = {
                sub_terrain_name: FlatPatchSamplingCfg(num_patches=10, patch_radius=0.5, max_height_diff=0.05)
            }

    # Handler for terrains importing
    terrain_importer_cfg = TerrainImporterCfg(
        num_envs=2048,
        env_spacing=3.0,
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type="generator",
        terrain_generator=terrain_gen_cfg,
        debug_vis=True,
    )
    # Remove visual material for height and random color schemes to use the default material
    if args_cli.color_scheme in ["height", "random"]:
        terrain_importer_cfg.visual_material = None
    # Create terrain importer
    terrain_importer = TerrainImporter(terrain_importer_cfg)

    # Show the flat patches computed
    if args_cli.show_flat_patches:
        # Configure the flat patches
        vis_cfg = VisualizationMarkersCfg(prim_path="/Visuals/TerrainFlatPatches", markers={})
        for name in terrain_importer.flat_patches:
            vis_cfg.markers[name] = sim_utils.CylinderCfg(
                radius=0.5,  # note: manually set to the patch radius for visualization
                height=0.1,
                visual_material=sim_utils.GlassMdlCfg(glass_color=(random.random(), random.random(), random.random())),
            )
        flat_patches_visualizer = VisualizationMarkers(vis_cfg)

        # Visualize the flat patches
        all_patch_locations = []
        all_patch_indices = []
        for i, patch_locations in enumerate(terrain_importer.flat_patches.values()):
            num_patch_locations = patch_locations.view(-1, 3).shape[0]
            # store the patch locations and indices
            all_patch_locations.append(patch_locations.view(-1, 3))
            all_patch_indices += [i] * num_patch_locations
        # combine the patch locations and indices
        flat_patches_visualizer.visualize(torch.cat(all_patch_locations), marker_indices=all_patch_indices)

#======================================================Robot Handling======================================================#
    scene_entities = {"terrain": terrain_importer}

    for id in range(0, num_robots):

        # Origin 1 with Boston Dynamics Spot
        prim_utils.create_prim(f"/World/Origin{id+1}", "Xform", translation=terrain_importer.env_origins[id])
        # -- Robot
        path = f"/World/Origin{id+1}/Robot"
        robot = Articulation(SPOT_CFG.replace(prim_path=path))
        name = f"spot{id+1}"
        scene_entities.update({name: robot})

    print(scene_entities)
    # return the scene information   
    return scene_entities, terrain_importer.env_origins

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset robots
            for index, robot in enumerate(entities.values()):
                #only assigning for robots, not terrain entities
                if index > num_terrain_entities:
                    # root state
                    print(index)
                    root_state = robot.data.default_root_state.clone()
                    root_state[:, :3] += origins[index]
                    robot.write_root_state_to_sim(root_state)
                    # joint state
                    joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                    robot.write_joint_state_to_sim(joint_pos, joint_vel)
                    # reset the internal state
                    robot.reset()
            print("[INFO]: Resetting robots state...")
        # apply default actions to the quadrupedal robots
        for index, robot in enumerate(entities.values()):
            #only assigning for robots, not terrain entities
            if index > num_terrain_entities:
                # generate random joint positions
                joint_pos_target = robot.data.default_joint_pos + torch.randn_like(robot.data.joint_pos) * 0.1
                # apply action to the robot
                robot.set_joint_position_target(joint_pos_target)
                # write data to sim
                robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for index, robot in enumerate(entities.values()):
            if index > num_terrain_entities:
                robot.update(sim_dt)

def main():
    """Main function."""

    # Initialize the simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    # Set main camera
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()