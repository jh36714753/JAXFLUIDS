import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids_postprocess import load_data, create_2D_animation

input_manager = InputManager("inputfiles/case_setup_Re200.json",
                             "../numerical_setup_files/numerical_setup_solids.json")
initialization_manager  = InitializationManager(input_manager)
sim_manager = SimulationManager(input_manager)
jxf_buffers = initialization_manager.initialization()
sim_manager.simulate(jxf_buffers)

path = sim_manager.output_writer.save_path_case
quantities = ["pressure", "velocity", "vorticity", "volume_fraction"]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities, step=10)

volume_fraction = data_dict["volume_fraction"]
solid_mask = np.where(volume_fraction == 0.0, 1, 0) 
vorticity = data_dict["vorticity"][:,0]
velocityY = data_dict["velocity"][:,1]

vorticity = np.ma.masked_where(solid_mask, vorticity)
velocityY = np.ma.masked_where(solid_mask, velocityY)

plot_dict = {
    "vorticity": vorticity,
    "velocityY": velocityY,
}

minmax_list = [
    [np.min(vorticity),-np.min(vorticity)],
    [np.min(velocityY),-np.min(velocityY)]
]

nrows_ncols = (1,2)
cmap = plt.get_cmap("seismic")
cmap.set_bad("white")

save_path = os.path.join(path,"images")
os.makedirs(save_path, exist_ok=True)
create_2D_animation(plot_dict, cell_centers, times, nrows_ncols=nrows_ncols,
                    save_png=save_path, fig_args={"figsize": (20,10)}, cmap=cmap,
                    dpi=300, minmax_list=minmax_list)
