"""
Import field parameters into RMS (Must be included as python job
in RMS workflow and edited to fit your scratch directory)
"""
from pathlib import Path

import xtgeo

SCRATCH = "/scratch/fmu/olia/sim_field/"
# SCRATCH = "/scratch/fmu/olia/sim_field_local/"
CASE_NAME = "original"
# CASE_NAME = "local"


# pylint: disable=undefined-variable, bare-except
PRJ = project  # noqa: 821

GRID_MODEL_NAME = "GRID"
FIELD_NAMES = [
    "FieldParam",
]


def main():
    """
    Import files with initial ensemble and updated fields into RMS project
    """
    xtgeo.grid_from_roxar(PRJ, GRID_MODEL_NAME, PRJ.current_realisation)

    path = Path(SCRATCH)
    if not path.exists():
        raise IOError(f"File path: {SCRATCH} does not exist. ")

    real = PRJ.current_realisation
    print("\n")
    print(f"Realization: {real} ")
    for name in FIELD_NAMES:
        for iteration in [0, 3]:
            print(f"Iteration: {iteration}")
            if iteration == 0:
                name_with_iter = name + "_" + CASE_NAME + "_" + str(iteration)
                path = (
                    SCRATCH
                    + "realization-"
                    + str(real)
                    + "/iter-"
                    + str(iter)
                    + "/init_files/"
                )
                file_name = path + name + ".roff"
                print(f"File name: {file_name}  ")

                try:
                    property0 = xtgeo.gridproperty_from_file(file_name, "roff")
                    print(
                        f"Import property {property0.name} from file"
                        f" {file_name} into {name_with_iter}  "
                    )
                    property0.to_roxar(
                        PRJ, GRID_MODEL_NAME, name_with_iter, realisation=real
                    )
                except:  # noqa: E722
                    print(f"Skip realization: {real} for iteration: {iteration}  ")
            elif iteration == 3:
                name_with_iter = name + "_" + CASE_NAME + "_" + str(iteration)
                path = (
                    SCRATCH
                    + "realization-"
                    + str(real)
                    + "/iter-"
                    + str(iteration)
                    + "/"
                )
                file_name = path + name + ".roff"
                print(f"File name: {file_name}  ")

                try:
                    property3 = xtgeo.gridproperty_from_file(file_name, "roff")
                    print(
                        f"Import property {property3.name} for iteration {iteration} "
                        f"from file {file_name} into {name_with_iter}  "
                    )
                    property3.to_roxar(
                        PRJ, GRID_MODEL_NAME, name_with_iter, realisation=real
                    )
                except:  # noqa: E722
                    print(f"Skip realization: {real} for iteration: {iteration}  ")
                try:
                    diff_property = property0
                    diff_property.values = property3.values - property0.values
                    name_diff = name + "_" + CASE_NAME + "_diff"
                    print(
                        f"Calculate difference between iteration 3 and 0:  {name_diff}"
                    )
                    diff_property.to_roxar(
                        PRJ, GRID_MODEL_NAME, name_diff, realisation=real
                    )
                except:  # noqa: E722
                    print(f"Skip difference for realisation: {real} ")


if __name__ == "__main__":
    main()
