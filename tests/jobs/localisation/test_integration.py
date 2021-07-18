import yaml
import pytest
from res.enkf import EnKFMain, ResConfig
from semeio.workflows.localisation.local_config_script import LocalisationConfigJob
from xtgeo.surface.regular_surface import RegularSurface
import xtgeo
import numpy as np


@pytest.mark.parametrize(
    "obs_group_add, param_group_add, expected, expected_obs1, expected_obs2",
    [
        (
            ["FOPR", "WOPR_OP1_190"],
            ["SNAKE_OIL_PARAM:OP1_DIVERGENCE_SCALE", "SNAKE_OIL_PARAM:OP1_OFFSET"],
            ["SNAKE_OIL_PARAM"],
            [
                "WOPR_OP1_108",
                "WOPR_OP1_144",
                "WOPR_OP1_36",
                "WOPR_OP1_72",
                "WOPR_OP1_9",
                "WPR_DIFF_1",
            ],
            ["FOPR", "WOPR_OP1_190"],
        ),
    ],
)
def test_localisation(
    setup_ert, obs_group_add, param_group_add, expected, expected_obs1, expected_obs2
):
    ert = EnKFMain(setup_ert)
    config = {
        "correlations": [
            {
                "name": "CORR1",
                "obs_group": {"add": "*", "remove": obs_group_add},
                "param_group": {
                    "add": "SNAKE_OIL_PARAM:OP1_PERSISTENCE",
                },
            },
            {
                "name": "CORR2",
                "obs_group": {"add": "*", "remove": obs_group_add},
                "param_group": {
                    "add": "SNAKE_OIL_PARAM:*",
                    "remove": "SNAKE_OIL_PARAM:OP1_PERSISTENCE",
                },
            },
            {
                "name": "CORR3",
                "obs_group": {"add": obs_group_add},
                "param_group": {
                    "add": param_group_add,
                },
            },
        ],
    }
    with open("local_config.yaml", "w") as fout:
        yaml.dump(config, fout)
    LocalisationConfigJob(ert).run("local_config.yaml")
    assert ert.getLocalConfig().getMinistep("CORR1").name() == "CORR1"
    assert (
        ert.getLocalConfig().getObsdata("CORR1_obs_group").name() == "CORR1_obs_group"
    )
    assert len(ert.getLocalConfig().getUpdatestep()) == 3
    ministep_names = ["CORR1", "CORR2", "CORR3"]
    for index, ministep in enumerate(ert.getLocalConfig().getUpdatestep()):
        assert ministep.name() == ministep_names[index]
        obs_list = []
        for count, obsnode in enumerate(ministep.getLocalObsData()):
            obs_list.append(obsnode.key())
        obs_list.sort()

        if index in [0, 1]:
            assert obs_list == expected_obs1
        else:
            assert obs_list == expected_obs2
        key = ministep_names[index] + "_param_group"
        assert ministep[key].keys() == expected


def test_localisation_gen_param(
    setup_poly_ert,
):
    with open("poly.ert", "a") as fout:
        fout.write(
            "GEN_PARAM PARAMS_A parameter_file_A INPUT_FORMAT:ASCII "
            "OUTPUT_FORMAT:ASCII INIT_FILES:initial_param_file_A_%d"
        )
    nreal = 5
    nparam = 10
    for n in range(nreal):
        filename = "initial_param_file_A_" + str(n)
        with open(filename, "w") as fout:
            for i in range(nparam):
                fout.write(f"{i}\n")

    res_config = ResConfig("poly.ert")
    ert = EnKFMain(res_config)
    config = {
        "correlations": [
            {
                "name": "CORR1",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": "*",
                },
            },
        ],
    }

    with open("local_config.yaml", "w") as fout:
        yaml.dump(config, fout)
    LocalisationConfigJob(ert).run("local_config.yaml")


def test_localisation_surf(
    setup_poly_ert,
):
    with open("poly.ert", "a") as fout:
        fout.write(
            "SURFACE   PARAM_SURF_A     OUTPUT_FILE:surf.txt    "
            "INIT_FILES:surf%d.txt   BASE_SURFACE:surf0.txt"
        )
    nreal = 20
    ncol = 10
    nrow = 10
    rotation = 0.0
    xinc = 50.0
    yinc = 50.0
    xori = 0.0
    yori = 0.0
    values = np.zeros(nrow * ncol)
    for n in range(nreal):
        filename = "surf" + str(n) + ".txt"
        delta = 0.1
        for j in range(nrow):
            for i in range(ncol):
                index = i + j * ncol
                values[index] = float(j) + n * delta
        surface = RegularSurface(
            ncol=ncol,
            nrow=nrow,
            xinc=xinc,
            yinc=yinc,
            xori=xori,
            yori=yori,
            rotation=rotation,
            values=values,
        )
        surface.to_file(filename, fformat="irap_ascii")

    res_config = ResConfig("poly.ert")
    ert = EnKFMain(res_config)
    config = {
        "correlations": [
            {
                "name": "CORR1",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": "*",
                },
                "ref_point": [250, 250],
                "surface_scale": {
                    "method": "gaussian_decay",
                    "main_range": 1700,
                    "perp_range": 850,
                    "angle": 200,
                    "filename": "surf0.txt",
                },
            },
        ],
    }

    with open("local_config.yaml", "w") as fout:
        yaml.dump(config, fout)
    LocalisationConfigJob(ert).run("local_config.yaml")


def test_localisation_field(
    setup_poly_ert,
):
    nreal = 20
    nx = 20
    ny = 10
    nz = 3
    xinc = 50.0
    yinc = 50.0
    zinc = 10.0
    xori = 0.0
    yori = 0.0
    grid = xtgeo.Grid()
    grid.create_box(
        dimension=(nx, ny, nz),
        origin=(xori, yori, 0.0),
        increment=(xinc, yinc, zinc),
        rotation=30.0,
        flip=-1,
    )
    grid_file_name = "grid3D.EGRID"
    print(f" Writel file: {grid_file_name}")
    grid.to_file(grid_file_name, fformat="egrid")
    property_name = "G"
    filename_output = property_name + ".roff"
    filename_input = property_name + "_%d.roff"
    values = np.zeros((nx, ny, nz), dtype=np.float32)
    property_field = xtgeo.GridProperty(grid, values=0.0, name=property_name)
    for n in range(nreal):
        values = np.zeros((nx, ny, nz), dtype=np.float32)
        property_field.values = values + 0.1 * n
        filename = property_name + str(n) + ".roff"
        print(f"Write file: {filename}")
        property_field.to_file(filename, fformat="roff", name=property_name)

    with open("poly.ert", "a") as fout:
        fout.write(
            f"GRID   {grid_file_name}\n"
            f"FIELD  {property_name}  PARAMETER  {filename_output}  "
            f"INIT_FILES:{filename_input}  MIN:-5.5   MAX:5.5     FORWARD_INIT:False\n"
        )

    res_config = ResConfig("poly.ert")
    ert = EnKFMain(res_config)
    config = {
        "correlations": [
            {
                "name": "CORR1",
                "obs_group": {
                    "add": "*",
                },
                "param_group": {
                    "add": "*",
                },
                "ref_point": [700, 370],
                "field_scale": {
                    "method": "gaussian_decay",
                    "main_range": 1700,
                    "perp_range": 850,
                    "angle": 200,
                },
            },
        ],
    }

    with open("local_config.yaml", "w") as fout:
        yaml.dump(config, fout)
    LocalisationConfigJob(ert).run("local_config.yaml")
