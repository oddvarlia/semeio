#!/usr/bin/env python
"""
Script used as forward model in ERT to test localisation.
"""
import os
import sys

# pylint: disable=import-error, redefined-outer-name
# pylint: disable=missing-function-docstring,invalid-name
from common_functions import (
    generate_field_and_upscale,
    get_cell_indices,
    get_nobs_from_cell_index_list,
    read_config_file,
    read_field_from_file,
    read_obs_field_from_file,
    read_observations,
    read_upscaled_field_from_file,
    upscaling,
    write_obs_pred_diff_field,
)


def write_prediction_gen_data(
    upscaled_values, cell_indx_list: list, response_file_name_prefix: str
):
    """
    Write GEN_DATA file with predicted values of observables (selected upscaled values)
    """
    print(response_file_name_prefix)
    response_file_name = response_file_name_prefix + "_0.txt"
    print(f"Write GEN_DATA file with prediction of observations: {response_file_name}")
    with open(response_file_name, "w", encoding="utf8") as file:
        # NOTE: The sequence of values must be the same as for the observations
        nobs = get_nobs_from_cell_index_list(cell_indx_list)
        for obs_number in range(nobs):
            (Iindx, Jindx, Kindx) = get_cell_indices(obs_number, cell_indx_list)
            value = upscaled_values[Iindx, Jindx, Kindx]
            file.write(f"{value}\n")


# pylint: disable=too-many-arguments,too-many-locals
def write_obs_and_predictions(
    config_path: str,
    upscaled_values,
    cell_index_list: list,
    observation_dir: str,
    obs_data_dir: str,
    predicted_obs_file_name: str = "obs_and_prediction.txt",
):
    obs_values = read_observations(
        config_path, observation_dir, obs_data_dir, cell_index_list
    )
    nobs = get_nobs_from_cell_index_list(cell_index_list)
    filename = config_path + "/" + predicted_obs_file_name
    print(f"Write file:  {filename}")
    with open(predicted_obs_file_name, "w", encoding="utf-8") as file:
        file.write(" Cell_index     Obs_value   Predicted_obs_value   Difference\n")
        for obs_number in range(nobs):
            (Iindx, Jindx, Kindx) = get_cell_indices(obs_number, cell_index_list)
            predicted_obs_values = upscaled_values[Iindx, Jindx, Kindx]
            obs_value = obs_values[obs_number]
            diff_obs_pred = obs_value - predicted_obs_values
            file.write(
                f"({Iindx+1}, {Jindx+1}, {Kindx + 1})      "
                f"{obs_value:10.5f}   {predicted_obs_values:10.5f}          "
                f"{diff_obs_pred:10.5f} \n"
            )


def get_iteration_real_number_config_file(argv):
    if len(argv) < 4:
        raise IOError(
            "Missing command line arguments <iteration> <real_number> <config_file>"
        )
    arg1 = argv[1]
    if arg1 is None:
        raise IOError(
            "Missing iteration number (argv[1]) when running this script manually"
        )
    iteration = int(arg1)
    print(f"ERT iteration: {iteration}")

    arg2 = argv[2]
    if arg2 is None:
        raise IOError("Missing real_number (argv[2]) when running this script manually")
    real_number = int(arg2)
    print(f"ERT realization: {real_number}")

    config_file_name = argv[3]

    config_path = argv[4]
    return iteration, real_number, config_file_name, config_path


def main(args):
    """
    For iteration = 0:
    - simulate field, export to file as initial ensemble realization
    - upscale and extract predicted values for observables
      (selected coarse grid cell values)
    For iteration > 0:
    - Import updated field from ERT.
    - upscale and extract predicted values for observables
      (selected coarse grid cell values)
    """

    # NOTE: Both the fine scale grid with simulated field values
    #  and the coarse grid with upscaled values must have Eclipse grid index origin

    # Read config_file if it exists. Use default settings for everything not specified.
    (
        iteration,
        real_number,
        config_file_name,
        config_path,
    ) = get_iteration_real_number_config_file(args)
    settings = read_config_file(config_file_name)

    print(f"Config path: {config_path}")
    if iteration == 0:
        print(f"Generate new field parameter realization:{real_number} ")
        # Simulate field (with trend)
        upscaled_values = generate_field_and_upscale(
            real_number,
            iteration,
            os.path.join(config_path, settings.field.seed_file),
            settings.field.algorithm,
            settings.field.name,
            settings.field.initial_file_name_prefix,
            settings.field.file_format,
            settings.field.grid_dimension,
            settings.model_size.size,
            settings.field.variogram,
            settings.field.correlation_range,
            settings.field.correlation_azimuth,
            settings.field.correlation_dip,
            settings.field.correlation_exponent,
            settings.field.trend_use,
            settings.field.trend_params,
            settings.field.trend_relstd,
            settings.response.name,
            settings.response.response_function,
            settings.response.upscaled_file_name,
            settings.response.grid_dimension,
            settings.response.write_upscaled_field,
            settings.model_size.use_eclipse_grid_index_origo,
        )

    else:
        print(f"Import updated field parameter realization: {real_number} ")
        field_object = read_field_from_file(
            settings.field.updated_file_name_prefix,
            settings.field.name,
            settings.field.file_format,
            settings.field.grid_file_name,
        )
        field_values = field_object.values

        # Calculate upscaled values for selected coarse grid cells
        upscaled_values = upscaling(
            field_values,
            settings.response.response_function,
            settings.response.file_format,
            settings.response.name,
            settings.response.write_upscaled_field,
            settings.response.upscaled_file_name,
            settings.response.grid_dimension,
            iteration=iteration,
        )
    # Write GEN_DATA file
    write_prediction_gen_data(
        upscaled_values,
        settings.observation.selected_grid_cells,
        settings.response.gen_data_file_prefix,
    )

    # Optional output calculate difference between upscaled field and
    # and reference upscaled field
    if settings.optional.write_obs_pred_diff_field_file:
        obs_field_object = read_obs_field_from_file(
            settings.response.file_format,
            os.path.join(config_path, settings.observation.reference_param_file),
            settings.response.grid_file_name,
            os.path.join(config_path, settings.observation.reference_field_name),
        )
        upscaled_field_object = read_upscaled_field_from_file(
            iteration,
            settings.response.upscaled_file_name,
            settings.response.file_format,
            settings.response.name,
            settings.response.grid_file_name,
        )
        write_obs_pred_diff_field(
            upscaled_field_object, obs_field_object, settings.field.file_format
        )

    write_obs_and_predictions(
        config_path,
        upscaled_values,
        settings.observation.selected_grid_cells,
        settings.observation.directory,
        settings.observation.data_dir,
    )


if __name__ == "__main__":
    # Command line arguments are iteration real_number  test_case_config_file
    main(sys.argv)
