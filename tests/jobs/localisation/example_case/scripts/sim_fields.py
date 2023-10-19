#!/usr/bin/env python
"""
Script used as forward model in ERT to test localisation.
"""
import math
import os
import random
import sys

import xtgeo
import gaussianfft as sim
import numpy as np


# NOTE:  xtgeo MUST be imported BEFORE gaussianfft
# The reason is that xtgeo has functions importing roxar API and
# even though this is not used, the code will crash with core dump
# since Roxar API and gaussianfft both use Boost to wrap C++ code
# into python functions but Roxar API and gaussianfft uses two
# slightly different versions of Boost.
# The gaussianfft module uses version 1.76 which is newer than
# version 1.74 from Roxar API (and indirectly xtgeo)
# which may explain why it works importing gaussianfft after
# xtgeo (and Roxar API module roxar). We don't know exactly the reason
# for the core dump with wrong sequence of the import's, but probably
# it is due to some initialization related to the Boost library and
# Boost version 1.74.0 is not compatible with any initialization done by
# version 1.76.
#
# The message related to Boost and RMS (and then indirectly also xtgeo)
# from Aspentech Support is this:
# "The current version of boost is 1.74.0 and this version has been used
#  since RMS 12.1, and is still used in RMS V14.0.1 and V14.1.
#  Boost version 1.81.0 will be available in version 14.2 or version 15.
#  Thomas also write "Last time I checked, boost did not provide any
#  compatibility guarantees, so it's not expected to work if you
#  mix two different boost versions in the same process
#  by loading python modules into RMS that uses other versions."

# The current combination of gaussianfft and xtgeo (or RMS Roxar API)
# will work if xtgeo is imported first. But this may change later.
# The plan for gaussianfft is to ensure correct sequence of import by
# importing xtgeo (and when running from RMS, also roxar) in
# gaussianfft before calling any functions from the gaussianfft module.
# In this case it should work for the end user regardless of which sequence
# it is imported. The most robust solution would be to generate gaussianfft
# to use exactly the same.



# pylint: disable=missing-function-docstring, too-many-locals, invalid-name
# pylint: disable=bare-except, raise-missing-from
# pylint: disable= redefined-outer-name, too-many-nested-blocks


def specify_settings():
    """
    grid_size - length, width, thickness of a box containing the field
                Same size is used for both fine scale grid with the simulated field
                and the coarse scale grid containing upscaled values
                of the simulated field.

    field_settings - Define the dimension (number of grid cells) for fine scale grid,
                    name of output files and specification of model parameters for
                    simulation of gaussian field with option to use linear trend.
                    Relative standard deviation specify standard deviation of
                    gaussian residual field relative to the trends span of value
                    (max trend value - min trend value)

    response_settings - Specify the coarse grid dimensions, name of file and type
                        of average operation to calculated upscaled values that
                        are predictions of observations of the same grid cells.
                        Which cell indices are observed are specified in
                        observation settings.

    observation_settings - Specify name of files for generated observations
                           and also which grid cells from coarse grid is used
                           as observables.
                           (Cells that have values that are used as observations)
    """
    grid_size = {
        "xsize": 7500.0,
        "ysize": 12500.0,
        "zsize": 50.0,
        "use_eclipse_grid_index_origin": True,
    }
    field_settings = {
        "grid_dimension": [150, 250, 1],
        "grid_file_name": "GRID.EGRID",
        "field_correlation_range": [3000.0, 1000.0, 2.0],
        "field_correlation_azimuth": 45.0,
        "field_correlation_dip": 0.0,
        "field_variogram": "spherical",
        "field_trend_params": [1.0, -1.0],
        "field_trend_relstd": 0.05,
        "field_trend_use": 0,
        "field_name": "FIELDPARAM",
        "field_updated_file_name": "FieldParam.roff",
        "field_initial_file_name": "init_files/FieldParam.roff",
        "field_seed_file": "randomseeds.txt",
    }
    response_settings = {
        "grid_dimension": [15, 25, 1],
        "upscaled_file_name": "Upscaled.roff",
        "grid_file_name": "UpscaleGrid.EGRID",
        "response_function": "average",
        "gen_data_file_name": "UpscaledField_0.txt",
        "all": True,
    }

    observation_settings = {
        "observation_dir": "observations",
        "observation_file": "observations.obs",
        "observation_data_dir": "obs_data",
        "3D_param_file_name": "init_files/UpscaledObsField.roff",
        "rel_error": 0.10,
        "min_abs_error": 0.01,
        "selected_grid_cells": [
            [5, 10, 1],
        ],
        # "selected_grid_cells":[
        #         [15,  1, 1],
        #         [15, 25, 1],
        #         [ 1, 25, 1],
        #         [ 1,  1, 1],
        #         [ 3,  3, 1],
        #         [ 5, 23, 1],
        #         [11,  4, 1],
        #         [ 3, 12, 1],
        #         [13, 18, 1],
        #     ],
    }
    return grid_size, field_settings, response_settings, observation_settings


def generate_seed_file(
    seed_file_name: str = "randomseeds.txt",
    start_seed: int = 9828862224,
    number_of_seeds: int = 1000,
):
    # pylint: disable=unused-variable
    print(f"Generate random seed file: {seed_file_name}")
    random.seed(start_seed)
    with open(seed_file_name, "w", encoding="utf8") as file:
        for i in range(number_of_seeds):
            file.write(f"{random.randint(1, 999999999)}\n")


def get_seed(seed_file_name, r_number):
    with open(seed_file_name, "r", encoding="utf8") as file:
        lines = file.readlines()
        try:
            seed_value = int(lines[r_number - 1])
        except:  # noqa: E722
            raise IOError("Seed value not found for realization {r_number}  ")
    return seed_value


def obs_positions(grid_size, response_settings, observation_settings):
    NX, NY, _ = response_settings["grid_dimension"]
    use_eclipse_origin = grid_size["use_eclipse_grid_index_origin"]

    xsize = grid_size["xsize"]
    ysize = grid_size["ysize"]
    dx = xsize / NX
    dy = ysize / NY
    cell_indx_list = observation_settings["selected_grid_cells"]
    if use_eclipse_origin:
        print("Grid index origin: Eclipse standard")
    else:
        print("Grid index origin: RMS standard")
    print(
        "Observation reference point coordinates is always "
        "from origin at lower left corner"
    )

    pos_list = []
    for indices in cell_indx_list:
        Iindx = indices[0] - 1
        Jindx = indices[1] - 1
        x = (Iindx + 0.5) * dx
        if use_eclipse_origin:
            y = ysize - (Jindx + 0.5) * dy
        else:
            y = (Jindx + 0.5) * dy

        pos_list.append((x, y))

    return pos_list


def write_localisation_config(
    observation_settings,
    field_settings,
    positions,
    config_file_name="local_config.yml",
    write_scaling=True,
):
    obs_index_list = observation_settings["selected_grid_cells"]
    field_name = field_settings["field_name"]
    corr_ranges = field_settings["field_correlation_range"]
    azimuth = field_settings["field_correlation_azimuth"]
    space = " " * 2
    space2 = " " * 4
    space3 = " " * 6
    print(f"Write localisation config file: {config_file_name}")
    with open(config_file_name, "w", encoding="utf8") as file:
        file.write("log_level: 3\n")
        file.write(f"write_scaling_factors: {write_scaling}\n")
        file.write("correlations:\n")
        for i, indx in enumerate(obs_index_list):
            I, J, K = indx
            obs_name = f"OBS_{I}_{J}_{K}"
            pos = positions[i]
            file.write(f"{space}- name: CORR_{i}\n")
            file.write(f"{space2}obs_group:\n")
            file.write(f'{space3}add: ["{obs_name}"]\n')
            file.write(f"{space2}param_group:\n")
            file.write(f'{space3}add: ["{field_name}"]\n')
            file.write(f"{space2}field_scale:\n")
            file.write(f"{space3}method: gaussian_decay\n")
            file.write(f"{space3}main_range: {corr_ranges[0]}\n")
            file.write(f"{space3}perp_range: {corr_ranges[1]}\n")
            file.write(f"{space3}azimuth: {azimuth}\n")
            file.write(f"{space3}ref_point: [ {pos[0]}, {pos[1]} ]\n")


def upscaling(
    field_values, response_settings, observation_settings, write_field=True, iteration=0
):
    response_function_name = response_settings["response_function"]
    upscaled_file_name = response_settings["upscaled_file_name"]
    NX, NY, NZ = response_settings["grid_dimension"]
    calculate_all = response_settings["all"]

    coarse_cell_index_list = observation_settings["selected_grid_cells"]
    upscaled_values = np.zeros((NX, NY, NZ), dtype=np.float32, order="F")
    upscaled_values[:, :, :] = -999

    if response_function_name == "average":
        upscaled_values = upscale_average(
            field_values, coarse_cell_index_list, upscaled_values, use_all=calculate_all
        )

    if iteration == 0:
        upscaled_file_name = "init_files/" + upscaled_file_name

    if write_field:
        write_upscaled_field(upscaled_values, upscaled_file_name)
    return upscaled_values


def write_upscaled_field(
    upscaled_values, upscaled_file_name, selected_cell_index_list=None
):
    nx, ny, nz = upscaled_values.shape
    field_name = "Upscaled"

    field_object = xtgeo.grid3d.GridProperty(
        ncol=nx,
        nrow=ny,
        nlay=nz,
        values=upscaled_values,
        discrete=False,
        name=field_name,
    )

    print(f"Write upscaled field file: {upscaled_file_name}  ")
    field_object.to_file(upscaled_file_name, fformat="roff")
    if selected_cell_index_list is not None:
        selected_upscaled_values = np.zeros((nx, ny, nz), dtype=np.float32, order="F")
        selected_upscaled_values[:, :, :] = -1
        for indices in selected_cell_index_list:
            Iindx = indices[0] - 1
            Jindx = indices[1] - 1
            Kindx = indices[2] - 1
            selected_upscaled_values[Iindx, Jindx, Kindx] = upscaled_values[
                Iindx, Jindx, Kindx
            ]

        field_name_selected = field_name + "_conditioned_cells"
        file_name_selected = "init_files/" + field_name_selected + ".roff"
        cond_field_object = xtgeo.grid3d.GridProperty(
            ncol=nx,
            nrow=ny,
            nlay=nz,
            values=selected_upscaled_values,
            discrete=False,
            name=field_name_selected,
        )
        print(f"Write conditioned cell values as field: {file_name_selected}")
        cond_field_object.to_file(file_name_selected, fformat="roff")

    return field_object


def upscale_average(
    field_values, coarse_cell_index_list, upscaled_values, use_all=False
):
    """
    Input: field_values (numpy 3D)
           coarse_cell_index_list (list of tuples (I,J,K))
    Output: upscaled_values  (numpy 3D) initialized outside
    but filled in specified (I,J,K) cells.
    """
    nx, ny, nz = field_values.shape
    NX, NY, NZ = upscaled_values.shape

    print(f"Number of fine scale grid cells:   (nx,ny,nz): ({nx},{ny},{nz})")
    print(f"Number of coarse scale grid cells: (NX,NY,NZ): ({NX},{NY},{NZ})  ")
    mx = int(nx / NX)
    my = int(ny / NY)
    mz = int(nz / NZ)
    print(
        "Number of fine scale grid cells per coarse grid cell: "
        f"(mx,my,mz): ({mx},{my},{mz})    "
    )
    if use_all:
        print("Calculate upscaled values for all grid cells")
        for Kindx in range(NZ):
            for Jindx in range(NY):
                for Iindx in range(NX):
                    istart = mx * Iindx
                    iend = istart + mx
                    jstart = my * Jindx
                    jend = jstart + my
                    kstart = mz * Kindx
                    kend = kstart + mz
                    sum_val = 0.0
                    for k in range(kstart, kend):
                        for j in range(jstart, jend):
                            for i in range(istart, iend):
                                sum_val += field_values[i, j, k]
                    upscaled_values[Iindx, Jindx, Kindx] = sum_val / (mx * my * mz)

    else:
        print("Calculate upscaled values for selected grid cells")
        for indices in coarse_cell_index_list:
            Iindx = indices[0] - 1
            Jindx = indices[1] - 1
            Kindx = indices[2] - 1
            istart = mx * Iindx
            iend = istart + mx
            jstart = my * Jindx
            jend = jstart + my
            kstart = mz * Kindx
            kend = kstart + mz
            sum_val = 0.0
            for k in range(kstart, kend):
                for j in range(jstart, jend):
                    for i in range(istart, iend):
                        sum_val += field_values[i, j, k]
            upscaled_values[Iindx, Jindx, Kindx] = sum_val / (mx * my * mz)

    return upscaled_values


def write_gen_obs(upscaled_values, observation_settings):
    observation_dir = observation_settings["observation_dir"]
    obs_file_name = observation_settings["observation_file"]
    obs_data_dir = observation_settings["observation_data_dir"]
    cell_indx_list = observation_settings["selected_grid_cells"]
    rel_err = observation_settings["rel_error"]
    min_err = observation_settings["min_abs_error"]
    print(f"Write observation file: {obs_file_name} ")
    if not os.path.exists(obs_data_dir):
        print(f"Create directory: {obs_data_dir} ")
        os.makedirs(obs_data_dir)
    filename = observation_dir + "/" + obs_file_name
    with open(filename, "w", encoding="utf8") as obs_file:
        number = 0
        for indices in cell_indx_list:
            Iindx = indices[0] - 1
            Jindx = indices[1] - 1
            Kindx = indices[2] - 1

            value = upscaled_values[Iindx, Jindx, Kindx]
            value_err = math.fabs(value) * rel_err
            value_err = max(value_err, min_err)

            obs_data_relative_file_name = (
                obs_data_dir
                + "/obs_"
                + str(Iindx + 1)
                + "_"
                + str(Jindx + 1)
                + "_"
                + str(Kindx + 1)
                + ".txt"
            )

            obs_file.write(f"GENERAL_OBSERVATION   OBS_{Iindx+1}_{Jindx+1}_{Kindx+1}  ")
            obs_file.write("{ ")
            obs_file.write(
                f"DATA = RESULT_UPSCALED_FIELD ; INDEX_LIST = {number} ; RESTART = 0;  "
            )
            obs_file.write(f"OBS_FILE = ./{obs_data_relative_file_name} ; ")
            obs_file.write(" };\n")
            number += 1
            data_file_name = observation_dir + "/" + obs_data_relative_file_name
            print(f"Write file: {data_file_name} ")
            with open(data_file_name, "w", encoding="utf8") as data_file:
                data_file.write(f"{value}  {value_err}\n")


def write_prediction_gen_data(upscaled_values, observation_settings, response_settings):
    cell_indx_list = observation_settings["selected_grid_cells"]
    response_file_name = response_settings["gen_data_file_name"]
    print(f"Write GEN_DATA file with prediction of observations: {response_file_name}")
    with open(response_file_name, "w", encoding="utf8") as file:
        # NOTE: The sequence of values must be the same as for the observations
        for indices in cell_indx_list:
            Iindx = indices[0] - 1
            Jindx = indices[1] - 1
            Kindx = indices[2] - 1
            value = upscaled_values[Iindx, Jindx, Kindx]
            print(f"Prediction of obs for {Iindx+1},{Jindx+1},{Kindx+1}: {value}")
            file.write(f"{value}\n")


def trend(grid_size, field_settings):
    """
    Return 3D numpy array with values following a linear trend
    scaled to take values between 0 and 1.
    """
    nx, ny, nz = field_settings["grid_dimension"]
    xsize = grid_size["xsize"]
    ysize = grid_size["ysize"]
    a, b = field_settings["field_trend_params"]

    x0 = 0.0
    y0 = 0.0
    dx = xsize / nx
    dy = ysize / ny

    maxsize = ysize
    if xsize > ysize:
        maxsize = xsize

    val = np.zeros((nx, ny, nz), dtype=np.float32, order="F")
    for i in range(nx):
        x = x0 + i * dx
        for j in range(ny):
            y = y0 + j * dy
            for k in range(nz):
                val[i, j, k] = a * (x - x0) / maxsize + b * (y - y0) / maxsize

    minval = np.min(val)
    maxval = np.max(val)
    val_normalized = (val - minval) / (maxval - minval)
    return val_normalized


def simulate_field(grid_size, field_settings, start_seed):
    # pylint: disable=no-member,
    variogram_name = field_settings["field_variogram"]
    corr_ranges = field_settings["field_correlation_range"]
    xrange = corr_ranges[0]
    yrange = corr_ranges[1]
    zrange = corr_ranges[2]

    azimuth = field_settings["field_correlation_azimuth"]
    dip = field_settings["field_correlation_dip"]

    nx, ny, nz = field_settings["grid_dimension"]
    xsize = grid_size["xsize"]
    ysize = grid_size["ysize"]
    zsize = grid_size["zsize"]

    dx = xsize / nx
    dy = ysize / ny
    dz = zsize / nz

    print(f"Start seed: {start_seed}")
    sim.seed(start_seed)

    variogram = sim.variogram(
        variogram_name,
        xrange,
        perp_range=yrange,
        depth_range=zrange,
        azimuth=azimuth - 90,
        dip=dip,
    )

    print(f"Simulate field with size: nx={nx},ny={ny} ")
    field1D = sim.simulate(variogram, nx, dx, ny, dy, nz, dz)
    field = field1D.reshape((nx, ny, nz), order="F")
    return field


def create_grid(grid_size, field_settings):
    grid_file_name = field_settings["grid_file_name"]
    nx, ny, nz = field_settings["grid_dimension"]
    xsize = grid_size["xsize"]
    ysize = grid_size["ysize"]
    zsize = grid_size["zsize"]
    if grid_size["use_eclipse_grid_index_origin"]:
        flip = -1
        x0 = 0.0
        y0 = ysize
        z0 = 0.0
    else:
        flip = 1
        x0 = 0.0
        y0 = 0.0
        z0 = 0.0

    dx = xsize / nx
    dy = ysize / ny
    dz = zsize / nz

    grid_object = xtgeo.create_box_grid(
        dimension=(nx, ny, nz),
        origin=(x0, y0, z0),
        increment=(dx, dy, dz),
        rotation=0.0,
        flip=flip,
    )

    print(f"Write grid file: {grid_file_name} ")
    grid_object.to_file(grid_file_name, fformat="egrid")
    return grid_object


def create_upscaled_grid(grid_size, response_settings):
    grid_file_name = response_settings["grid_file_name"]
    nx, ny, nz = response_settings["grid_dimension"]
    xsize = grid_size["xsize"]
    ysize = grid_size["ysize"]
    zsize = grid_size["zsize"]
    if grid_size["use_eclipse_grid_index_origin"]:
        flip = -1
        x0 = 0.0
        y0 = ysize
        z0 = 0.0
    else:
        flip = 1
        x0 = 0.0
        y0 = 0.0
        z0 = 0.0

    dx = xsize / nx
    dy = ysize / ny
    dz = zsize / nz

    grid_object = xtgeo.create_box_grid(
        dimension=(nx, ny, nz),
        origin=(x0, y0, z0),
        increment=(dx, dy, dz),
        rotation=0.0,
        flip=flip,
    )

    print(f"Write grid file: {grid_file_name} ")
    grid_object.to_file(grid_file_name, fformat="egrid")
    return grid_object


def export_field(field_settings, field3D):
    # Export initial ensemble field
    nx, ny, nz = field_settings["grid_dimension"]
    field_name = field_settings["field_name"]
    field_file_name = field_settings["field_initial_file_name"]

    field_object = xtgeo.grid3d.GridProperty(
        ncol=nx, nrow=ny, nlay=nz, values=field3D, discrete=False, name=field_name
    )

    print(f"Write field file: {field_file_name}  ")
    field_object.to_file(field_file_name, fformat="roff")
    return field_object


def read_field_from_file(field_settings):
    input_file_name = field_settings["field_updated_file_name"]
    name = field_settings["field_name"]
    field_object = xtgeo.gridproperty_from_file(
        input_file_name, fformat="roff", name=name
    )
    return field_object


def read_obs_field_from_file(observation_settings):
    input_file_name = observation_settings["3D_param_file_name"]
    obs_field_object = xtgeo.gridproperty_from_file(input_file_name, fformat="roff")
    return obs_field_object


def read_upscaled_field_from_file(response_settings, iteration):
    input_file_name = response_settings["upscaled_file_name"]
    if iteration == 0:
        filename = "init_files/" + input_file_name
    else:
        filename = input_file_name
    field_object = xtgeo.gridproperty_from_file(filename, fformat="roff")
    return field_object


def write_obs_pred_diff_field(upscaled_field_object, observation_field_object):
    nx, ny, nz = upscaled_field_object.dimensions
    values_diff = upscaled_field_object.values - observation_field_object.values

    diff_object = xtgeo.grid3d.GridProperty(
        ncol=nx,
        nrow=ny,
        nlay=nz,
        values=values_diff,
        discrete=False,
        name="DiffObsPred",
    )

    filename = "DiffObsPred.roff"
    print(
        f"Write field with difference between observation and prediction: {filename}  "
    )
    diff_object.to_file(filename, fformat="roff")


def main(
    iteration,
    real_number,
    write_fine_grid=False,
    write_coarse_grid=False,
    write_upscaled_to_file=False,
    extract_and_write_obs=False,
    write_obs_pred_diff_field_file=False,
):
    # pylint: disable=too-many-arguments
    """
    Specify settings for fine grid and model parameters for simulating a
    field on fine grid.
    Specify settings for coarse grid.
    Specify settings for synthetic observations extracted from upscaled
    field values from coarse grid.
    Simulate a field on fine grid.
    Export the fine grid and the field for the fine grid to files to be used by ERT.
    Upscale the fine grid field to a coarse grid field which is used as
    response variables here.
    Option to extract synthetic observations for upscaled field parameters
    and generate ERT observation files.
    Options to generate a sequence of random seeds to make the simulations repeatable.

    """

    # NOTE: Both the fine scale grid with simulated field values
    #  and the coarse grid with upscaled values must have Eclipse grid index origin

    # Settings are specified here
    (
        grid_size,
        field_settings,
        response_settings,
        observation_settings,
    ) = specify_settings()

    # Create and write grid file for fine scale grid
    if write_fine_grid:
        create_grid(grid_size, field_settings)

    if write_coarse_grid:
        create_upscaled_grid(grid_size, response_settings)

    if iteration == 0:
        print(f"Generate new field parameter realization:{real_number} ")
        # Simulate field (with trend)
        seed_file_name = field_settings["field_seed_file"]
        relative_std = field_settings["field_trend_relstd"]
        start_seed = get_seed(seed_file_name, real_number)
        residual_field = simulate_field(grid_size, field_settings, start_seed)
        trend_field = trend(grid_size, field_settings)
        use_trend = field_settings["field_trend_use"]

        if use_trend == 1:
            field3D = trend_field + relative_std * residual_field
        else:
            field3D = residual_field

        # Write field parameter for fine scale grid
        field_object = export_field(field_settings, field3D)
        field_values = field_object.values

        # Calculate upscaled values for selected coarse grid cells
        upscaled_values = upscaling(
            field_values,
            response_settings,
            observation_settings,
            write_field=write_upscaled_to_file,
            iteration=iteration,
        )

    else:
        print(f"Import updated field parameter realization: {real_number} ")
        field_object = read_field_from_file(field_settings)
        field_values = field_object.values

        # Calculate upscaled values for selected coarse grid cells
        upscaled_values = upscaling(
            field_values,
            response_settings,
            observation_settings,
            write_field=write_upscaled_to_file,
            iteration=iteration,
        )

    if extract_and_write_obs:
        # ERT obs files
        write_gen_obs(upscaled_values, observation_settings)

        # Write upscaled field used as truth realisation
        write_upscaled_field(
            upscaled_values,
            observation_settings["3D_param_file_name"],
            selected_cell_index_list=observation_settings["selected_grid_cells"],
        )

        # Write file for non-adaptive localisation using distance based localisation
        positions = obs_positions(grid_size, response_settings, observation_settings)
        write_localisation_config(
            observation_settings,
            field_settings,
            positions,
            config_file_name="local_config.yml",
        )

    if write_obs_pred_diff_field_file:
        obs_field_object = read_obs_field_from_file(observation_settings)
        upscaled_field_object = read_upscaled_field_from_file(
            response_settings, iteration
        )
        write_obs_pred_diff_field(upscaled_field_object, obs_field_object)

    write_prediction_gen_data(upscaled_values, observation_settings, response_settings)


if __name__ == "__main__":
    # Create file with one seed pr realization
    make_seed_file = False

    # Create and write fine scale grid file to be used in ERT GRID keyword
    write_fine_grid = False

    # Create upscaled grid file to be used in visualization of upscaled values in RMS
    write_coarse_grid = False

    # Write 3D parameter of upscaled values for the selected
    # grid cells having observations
    write_upscaled_to_file = True
    write_obs_pred_diff_field_file = True

    # Extract and write observation files for ERT.
    # Observations extracted for a 3D parameter from the coarse grid.
    extract_and_write_obs = False

    try:
        iteration = int(os.environ.get("_ERT_ITERATION_NUMBER"))
        extract_and_write_obs = False
    except:  # noqa:  E722
        iteration = int(sys.argv[1])
    print(f"ERT iteration: {iteration}")

    try:
        real_number = int(os.environ.get("_ERT_REALIZATION_NUMBER"))
    except:  # noqa: E722
        real_number = int(sys.argv[2])
    print(f"ERT realization: {real_number}")

    if make_seed_file:
        generate_seed_file()

    main(
        iteration,
        real_number,
        write_fine_grid=write_fine_grid,
        write_coarse_grid=write_coarse_grid,
        write_upscaled_to_file=write_upscaled_to_file,
        extract_and_write_obs=extract_and_write_obs,
        write_obs_pred_diff_field_file=write_obs_pred_diff_field_file,
    )
