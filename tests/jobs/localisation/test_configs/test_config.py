from unittest.mock import MagicMock

import pytest
import pydantic

from semeio.workflows.localisation.localisation_config import (
    LocalisationConfig,
    expand_wildcards,
    check_for_duplicated_correlation_specifications,
)
from res.enkf.enums.ert_impl_type_enum import ErtImplType


ERT_OBS = ["OBS1", "OBS2", "OBS11", "OBS22", "OBS12", "OBS13", "OBS14", "OBS3"]
ERT_GEN_OBS_EXPANDED = [
    "GENOBSA:0",
    "GENOBSA:1",
    "GENOBSA:2",
    "GENOBSB:0",
    "GENOBSB:1",
    "GENOBSC:0",
]
ERT_GEN_OBS = [
    "GENOBSA",
    "GENOBSB",
    "GENOBSC",
]
ERT_SUMMARY_OBS = [
    "WBP9:OP_1",
    "WBP9:OP_2",
    "WBP9:OP_3",
]

ERT_PARAM = [
    "PARAM_NODE1:PARAM1",
    "PARAM_NODE1:PARAM2",
    "PARAM_NODE2:PARAM1",
    "PARAM_NODE2:PARAM2",
    "PARAM_NODE2:PARAM3",
    "PARAM_NODE3:PARAM4",
    "PARAM_NODE22:P1",
    "PARAM_NODE22:P2",
    "PARAM_NODE22:P22",
    "PARAM_NODE1X:X1",
    "PARAM_NODE1X:X2",
    "PARAM_NODE1X:X3",
    "PARAM_NODE2Y:Y1",
    "PARAM_NODE2Y:Y2",
    "PARAM_NODE2Y:Y3",
    "PARAM_FIELD1",
    "PARAM_FIELD2",
    "PARAM_FIELD3",
    "PARAM_GEN:0",
    "PARAM_GEN:1",
    "PARAM_GEN:2",
    "PARAM_GEN:3",
    "PARAM_GEN:4",
    "PARAM_SURFACE1",
    "PARAM_SURFACE2",
]


ERT_NODE_TYPE = {
    "PARAM_NODE1": ErtImplType.GEN_KW,
    "PARAM_NODE2": ErtImplType.GEN_KW,
    "PARAM_NODE3": ErtImplType.GEN_KW,
    "PARAM_NODE22": ErtImplType.GEN_KW,
    "PARAM_NODE1X": ErtImplType.GEN_KW,
    "PARAM_NODE2Y": ErtImplType.GEN_KW,
    "PARAM_FIELD1": ErtImplType.FIELD,
    "PARAM_FIELD2": ErtImplType.FIELD,
    "PARAM_FIELD3": ErtImplType.FIELD,
    "PARAM_GEN": ErtImplType.GEN_DATA,
    "PARAM_SURFACE1": ErtImplType.SURFACE,
    "PARAM_SURFACE2": ErtImplType.SURFACE,
}


@pytest.mark.parametrize(
    "param_group_add,  expected",
    [
        (
            "PARAM_NODE1:*",
            ["PARAM_NODE1:PARAM1", "PARAM_NODE1:PARAM2"],
        ),
        (
            "PARAM_N*1:*",
            ["PARAM_NODE1:PARAM1", "PARAM_NODE1:PARAM2"],
        ),
        (
            ["P*2*", "PARAM_NODE3*"],
            [
                "PARAM_FIELD2",
                "PARAM_GEN:2",
                "PARAM_NODE1:PARAM2",
                "PARAM_NODE1X:X2",
                "PARAM_NODE22:P1",
                "PARAM_NODE22:P2",
                "PARAM_NODE22:P22",
                "PARAM_NODE2:PARAM1",
                "PARAM_NODE2:PARAM2",
                "PARAM_NODE2:PARAM3",
                "PARAM_NODE2Y:Y1",
                "PARAM_NODE2Y:Y2",
                "PARAM_NODE2Y:Y3",
                "PARAM_NODE3:PARAM4",
                "PARAM_SURFACE2",
            ],
        ),
        (
            ["P*2:*", "PARAM_NODE3*"],
            [
                "PARAM_NODE2:PARAM1",
                "PARAM_NODE2:PARAM2",
                "PARAM_NODE2:PARAM3",
                "PARAM_NODE22:P1",
                "PARAM_NODE22:P2",
                "PARAM_NODE22:P22",
                "PARAM_NODE3:PARAM4",
            ],
        ),
        (
            ["PARAM_NODE3:P*", "PARAM_NODE2*:*2*"],
            [
                "PARAM_NODE2:PARAM2",
                "PARAM_NODE22:P2",
                "PARAM_NODE22:P22",
                "PARAM_NODE2Y:Y2",
                "PARAM_NODE3:PARAM4",
            ],
        ),
        (
            "PARAM_FIELD*",
            ["PARAM_FIELD1", "PARAM_FIELD2", "PARAM_FIELD3"],
        ),
        (
            "PARAM_FIELD1",
            ["PARAM_FIELD1"],
        ),
        (
            "PARAM_GEN:*",
            ["PARAM_GEN:0", "PARAM_GEN:1", "PARAM_GEN:2", "PARAM_GEN:3", "PARAM_GEN:4"],
        ),
    ],
)
def test_simple_config(param_group_add, expected):
    data = {
        "log_level": 2,
        "correlations": [
            {
                "name": "some_name",
                "obs_group": {"add": ["OBS1"]},
                "param_group": {
                    "add": param_group_add,
                },
            }
        ],
    }
    conf = LocalisationConfig(observations=ERT_OBS, parameters=ERT_PARAM, **data)
    assert sorted(conf.correlations[0].param_group.result_items) == sorted(expected)


# Test GEN_OBS expanded into list of individual observations and not expanded.
# Expansion occurs as long as max_gen_obs_size is larger than the number of obs in
# the GEN_OBS node. Error will occur if user has specified individual observations
# and the max_gen_obs_size is less than the number of obs for the GEN_OBS node.
# Error will also occur if user specify node name only and the number of obs in the
# node is less than max_gen_obs_size.
@pytest.mark.parametrize(
    "obs_group_add, obs_group_remove, ert_gen_obs, max_size_to_expand, expected",
    [
        (
            "GENOBS*",
            [],
            ERT_GEN_OBS_EXPANDED,
            10,
            [
                "GENOBSA:0",
                "GENOBSA:1",
                "GENOBSA:2",
                "GENOBSB:0",
                "GENOBSB:1",
                "GENOBSC:0",
            ],
        ),
        (
            ["GENOBSB:*"],
            ["GENOBSB:0"],
            ERT_GEN_OBS_EXPANDED,
            10,
            ["GENOBSB:1"],
        ),
        (
            ["*"],
            ["*B:0"],
            ERT_GEN_OBS_EXPANDED,
            10,
            ["GENOBSA:0", "GENOBSA:1", "GENOBSA:2", "GENOBSB:1", "GENOBSC:0"],
        ),
        (
            ["*"],
            ["*B"],
            ERT_GEN_OBS,
            0,
            ["GENOBSA", "GENOBSC"],
        ),
    ],
)
def test_gen_obs_config(
    obs_group_add, obs_group_remove, ert_gen_obs, max_size_to_expand, expected
):
    data = {
        "log_level": 2,
        "max_gen_obs_size": max_size_to_expand,
        "correlations": [
            {
                "name": "some_name",
                "obs_group": {
                    "add": obs_group_add,
                    "remove": obs_group_remove,
                },
                "param_group": {"add": ["PARAM_NODE1:*"]},
            }
        ],
    }
    conf = LocalisationConfig(observations=ert_gen_obs, parameters=ERT_PARAM, **data)
    assert sorted(conf.correlations[0].obs_group.result_items) == sorted(expected)


@pytest.mark.parametrize(
    "obs_group_add, obs_group_remove, expected",
    [
        (
            ["WBP9:*"],
            [],
            ["WBP9:OP_1", "WBP9:OP_2", "WBP9:OP_3"],
        ),
        (
            ["WBP9:*"],
            ["WBP9:OP_1", "WBP9:OP_3"],
            ["WBP9:OP_2"],
        ),
    ],
)
def test_summary_obs_config(obs_group_add, obs_group_remove, expected):
    data = {
        "log_level": 2,
        "max_gen_obs_size": 10,
        "correlations": [
            {
                "name": "some_name",
                "obs_group": {
                    "add": obs_group_add,
                    "remove": obs_group_remove,
                },
                "param_group": {"add": ["PARAM_NODE1:*"]},
            }
        ],
    }
    conf = LocalisationConfig(
        observations=ERT_SUMMARY_OBS, parameters=ERT_PARAM, **data
    )
    assert sorted(conf.correlations[0].obs_group.result_items) == sorted(expected)


@pytest.mark.parametrize(
    "obs_group_add, param_group_add,  param_group_remove, expected_error",
    [
        (
            ["OBS*"],
            "PARAM_NODE1:PARAM1",
            "P*:*:*",
            "No match for: P*:*:*",
        ),
        (
            [],
            "PARAM_NODE1:PARAM1",
            [],
            "",
        ),
    ],
)
def test_simple_config_error(
    obs_group_add, param_group_add, param_group_remove, expected_error
):
    data = {
        "log_level": 2,
        "correlations": [
            {
                "name": "some_name",
                "obs_group": {
                    "add": obs_group_add,
                },
                "param_group": {
                    "add": param_group_add,
                    "remove": param_group_remove,
                },
            }
        ],
    }
    with pytest.raises(pydantic.error_wrappers.ValidationError, match=expected_error):
        LocalisationConfig(observations=ERT_OBS, parameters=ERT_PARAM, **data)


@pytest.mark.parametrize(
    "obsgroup1, paramgroup1,  obsgroup2, paramgroup2, expected_error",
    [
        (
            "OBS1",
            "PARAM_NODE1:*",
            "OBS*",
            "PARAM_NODE*:*",
            "Found 2 duplicated correlations",
        ),
        (
            ["OBS1", "OBS2"],
            ["PARAM_NODE2:*"],
            ["OBS1*", "OBS2*"],
            ["P*:*"],
            "Found 6 duplicated correlations",
        ),
        (
            ["OBS1*"],
            ["PARAM_FIELD2"],
            ["OBS14"],
            ["PARAM_F*"],
            "Found 1 duplicated correlations",
        ),
        (
            "*",
            "*",
            "O*",
            "P*:*",
            "Found 160 duplicated correlations",
        ),
    ],
)
def test_simple_config_duplicate_error(
    obsgroup1, paramgroup1, obsgroup2, paramgroup2, expected_error
):
    data = {
        "log_level": 2,
        "correlations": [
            {
                "name": "some_name1",
                "obs_group": {
                    "add": obsgroup1,
                },
                "param_group": {
                    "add": paramgroup1,
                },
            },
            {
                "name": "some_name2",
                "obs_group": {
                    "add": obsgroup2,
                },
                "param_group": {
                    "add": paramgroup2,
                },
            },
        ],
    }
    with pytest.raises(ValueError, match=expected_error):
        LocalisationConfig(observations=ERT_OBS, parameters=ERT_PARAM, **data)


@pytest.mark.parametrize(
    "ref_point, expected_error",
    [
        (
            [],
            "least 2 items",
        ),
        (
            [100],
            "least 2 items ",
        ),
        (
            ["not_float", 200],
            "value is not a valid float",
        ),
        (
            [100, 200, 300],
            "at most 2 items",
        ),
    ],
)
def test_simple_config_ref_point_error(ref_point, expected_error):
    data = {
        "correlations": [
            {
                "name": "some_name",
                "obs_group": {
                    "add": "OBS",
                },
                "param_group": {
                    "add": "PARAM_NODE1",
                },
                "field_scale": {
                    "method": "gaussian_decay",
                    "main_range": 1000,
                    "perp_range": 1000,
                    "azimuth": 200,
                    "ref_point": ref_point,
                },
            }
        ],
    }
    with pytest.raises(ValueError, match=expected_error):
        LocalisationConfig(observations=["OBS"], parameters=["PARAM_NODE1"], **data)


@pytest.mark.parametrize(
    "param_group_add,  param_group_remove, expected",
    [
        (
            "PARAM_NODE1:*",
            "PARAM_NODE2:*",
            ["PARAM_NODE1:PARAM1", "PARAM_NODE1:PARAM2"],
        ),
        (
            "PARAM_N*1*",
            "PARAM_NODE1:PARAM1",
            [
                "PARAM_NODE1:PARAM2",
                "PARAM_NODE1X:X1",
                "PARAM_NODE1X:X2",
                "PARAM_NODE1X:X3",
                "PARAM_NODE2:PARAM1",
                "PARAM_NODE22:P1",
                "PARAM_NODE2Y:Y1",
            ],
        ),
        (
            ["P*2*:*", "PARAM_NODE3*", "P*_GEN*"],
            ["PARAM_NODE2:*", "PARAM_NODE22:P2*", "P*_G*:1", "P*_G*:3", "P*_G*:4"],
            [
                "PARAM_GEN:0",
                "PARAM_GEN:2",
                "PARAM_NODE22:P1",
                "PARAM_NODE2Y:Y1",
                "PARAM_NODE2Y:Y2",
                "PARAM_NODE2Y:Y3",
                "PARAM_NODE3:PARAM4",
            ],
        ),
        (["*FIELD*"], ["*NODE*"], ["PARAM_FIELD1", "PARAM_FIELD2", "PARAM_FIELD3"]),
        (
            ["*"],
            ["PARAM_NODE*"],
            [
                "PARAM_FIELD1",
                "PARAM_FIELD2",
                "PARAM_FIELD3",
                "PARAM_GEN:0",
                "PARAM_GEN:1",
                "PARAM_GEN:2",
                "PARAM_GEN:3",
                "PARAM_GEN:4",
                "PARAM_SURFACE1",
                "PARAM_SURFACE2",
            ],
        ),
    ],
)
def test_add_remove_param_config(param_group_add, param_group_remove, expected):
    data = {
        "log_level": 2,
        "correlations": [
            {
                "name": "some_name",
                "obs_group": {"add": ["OBS1"]},
                "param_group": {
                    "add": param_group_add,
                    "remove": param_group_remove,
                },
            }
        ],
    }
    conf = LocalisationConfig(observations=ERT_OBS, parameters=ERT_PARAM, **data)
    assert sorted(conf.correlations[0].param_group.result_items) == sorted(expected)


@pytest.mark.parametrize(
    "config, expected",
    [
        (
            {
                "add": ["PARAM_NODE1:PARAM1"],
                "remove": "PARAM_NODE*:PARAM1",
            },
            r"Adding: \['PARAM_NODE1:PARAM1'\] and removing: \['PARAM_NODE\*:PARAM1'\]",
        ),
        (
            {
                "add": ["*"],
                "remove": ["*"],
            },
            r"Adding: \['\*'\] and removing: \['\*'\]",
        ),
        (
            {
                "add": ["*FIELD*"],
                "remove": ["*"],
            },
            r"Adding: \['\*FIELD\*'\] and removing: \['\*'\]",
        ),
    ],
)
def test_add_remove_param_config_no_param(config, expected):
    data = {
        "correlations": [
            {"name": "some_name", "obs_group": {"add": ["OBS1"]}, "param_group": config}
        ],
    }
    with pytest.raises(ValueError, match=expected):
        LocalisationConfig(observations=ERT_OBS, parameters=ERT_PARAM, **data)


@pytest.mark.parametrize(
    "obs_group_add, obs_group_remove, expected",
    [
        (
            "OBS*",
            "OBS*2",
            ["OBS1", "OBS11", "OBS13", "OBS14", "OBS3"],
        ),
        (
            "*",
            "OBS*2",
            ["OBS1", "OBS11", "OBS13", "OBS14", "OBS3"],
        ),
        (
            "*2*",
            "*1*",
            ["OBS2", "OBS22"],
        ),
        (
            "*3",
            "*1*",
            ["OBS3"],
        ),
    ],
)
def test_add_remove_obs_config(obs_group_add, obs_group_remove, expected):
    data = {
        "log_level": 2,
        "correlations": [
            {
                "name": "some_name",
                "obs_group": {"add": [obs_group_add], "remove": [obs_group_remove]},
                "param_group": {
                    "add": ["PARAM_NODE1:PARAM1"],
                },
            }
        ],
    }
    conf = LocalisationConfig(observations=ERT_OBS, parameters=ERT_PARAM, **data)
    assert len(conf.correlations) == 1
    assert conf.correlations[0].obs_group.result_items == expected


@pytest.mark.parametrize(
    "pattern, list_of_words, expected_result",
    [(["*"], ["OBS_1", "2"], {"OBS_1", "2"}), ((["OBS*"], ["OBS_1", "2"], {"OBS_1"}))],
)
def test_wildcard_expansion(pattern, list_of_words, expected_result):
    result = expand_wildcards(pattern, list_of_words)
    assert result == expected_result


@pytest.mark.parametrize(
    "pattern, list_of_words, expected_error",
    [
        (["NOT:"], ["OBS_1", "2"], "No match for: NOT"),
        (["OBS", "OBS_1"], ["OBS_1", "2"], "No match for: OBS"),
        (["NOT", "OBS"], ["OBS_1", "2"], "No match for: NOT"),
    ],
)
def test_wildcard_expansion_mismatch(pattern, list_of_words, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        expand_wildcards(pattern, list_of_words)


@pytest.mark.parametrize(
    "obs_1, obs_2, param_1, param_2, expected",
    (
        [["a"], ["a"], ["b"], ["b"], ["Observation: a, parameter: b"]],
        [["a", "c"], ["a"], ["b"], ["b"], ["Observation: a, parameter: b"]],
        [
            ["a", "c"],
            ["a", "c"],
            ["b"],
            ["b"],
            ["Observation: a, parameter: b", "Observation: c, parameter: b"],
        ],
        [
            ["a", "c"],
            ["a", "c"],
            ["b", "d"],
            ["b"],
            ["Observation: a, parameter: b", "Observation: c, parameter: b"],
        ],
    ),
)
def test_check_for_duplicates(obs_1, obs_2, param_1, param_2, expected):
    correlation_1 = MagicMock()
    correlation_2 = MagicMock()
    correlation_1.obs_group.result_items = obs_1
    correlation_1.param_group.result_items = param_1
    correlation_2.obs_group.result_items = obs_2
    correlation_2.param_group.result_items = param_2
    correlations = [correlation_1, correlation_2]
    result = check_for_duplicated_correlation_specifications(correlations)
    assert result == expected


@pytest.mark.parametrize(
    "active_segment_list, scaling_factor_list, smooth_ranges",
    [
        (
            [1, 2, 3],
            [1.0, 1.0e-5, 0.1],
            [0, 0],
        ),
        (
            [4, 1, 3],
            [1.0, 0.5, 0.0],
            [1, 1],
        ),
    ],
)
def test_active_region_list(active_segment_list, scaling_factor_list, smooth_ranges):
    data = {
        "log_level": 2,
        "correlations": [
            {
                "name": "CORR1_SEGMENT",
                "obs_group": {
                    "add": ["OBS1"],
                },
                "param_group": {
                    "add": ["*"],
                },
                "field_scale": {
                    "method": "segment",
                    "segment_filename": "Region.GRDECL",
                    "param_name": "Region",
                    "active_segments": active_segment_list,
                    "scalingfactors": scaling_factor_list,
                    "smooth_ranges": smooth_ranges,
                },
            },
        ],
    }
    conf = LocalisationConfig(observations=["OBS1"], parameters=["PARAM_NODE1"], **data)
    assert conf.correlations[0].field_scale.active_segments == active_segment_list
    assert conf.correlations[0].field_scale.scalingfactors == scaling_factor_list
    assert conf.correlations[0].field_scale.smooth_ranges == smooth_ranges
    assert conf.correlations[0].field_scale.segment_filename == "Region.GRDECL"
    assert conf.correlations[0].field_scale.param_name == "Region"


@pytest.mark.parametrize(
    "active_segment_list, scaling_factor_list, smooth_ranges, expected_error",
    [
        (
            [1, -2, 3],
            [1.0, 0.5, 0.1],
            [0, 0],
            "2 validation errors for LocalisationConfig",
        ),
        (
            [1, 2, 3],
            [-1.0, -0.5, 0.1],
            [1, 1],
            "3 validation errors for LocalisationConfig",
        ),
        (
            [1, 2, 3],
            [1.0, 0.5, 0.1],
            [1, -1],
            "1 validation error for LocalisationConfig",
        ),
        (
            [1, 2, 3, 4],
            [1.0, 0.5, 0.1],
            [1, 1],
            "The specified length of 'active_segments' list",
        ),
        (
            [1, 2, 4],
            [1.0, 0.5, 0.1, 0.0],
            [1, 1],
            "The specified length of 'active_segments' list",
        ),
        (
            [1, 2, 4],
            [1.0, 0.5, 0.1],
            [1, 1, 2],
            "1 validation error for LocalisationConfig",
        ),
    ],
)
def test_active_region_list_mismatch(
    active_segment_list, scaling_factor_list, smooth_ranges, expected_error
):
    data = {
        "log_level": 2,
        "correlations": [
            {
                "name": "CORR1_SEGMENT",
                "obs_group": {
                    "add": ["OBS1"],
                },
                "param_group": {
                    "add": ["*"],
                },
                "field_scale": {
                    "method": "segment",
                    "segment_filename": "Region.GRDECL",
                    "param_name": "Region",
                    "active_segments": active_segment_list,
                    "scalingfactors": scaling_factor_list,
                    "smooth_ranges": smooth_ranges,
                },
            },
        ],
    }
    with pytest.raises(ValueError, match=expected_error):
        LocalisationConfig(observations=["OBS1"], parameters=["PARAM_NODE1"], **data)


def test_invalid_keyword_errors_method_segment():
    expected_error = "extra fields not permitted"
    data = {
        "log_level": 2,
        "correlations": [
            {
                "name": "CORR1_SEGMENT",
                "obs_group": {
                    "add": ["OBS1"],
                },
                "param_group": {
                    "add": ["*"],
                },
                "field_scale": {
                    "method": "segment",
                    "segment_filename": "Region.GRDECL",
                    "param_name": "Region",
                    "active_segments": [1, 2, 3],
                    "scalingfactors": [1.0, 0.5, 0.05],
                    "smooth_ranges": [0, 0],
                    "dummy1": "unused1",
                    "dummy2": "unused2",
                },
            },
        ],
    }
    with pytest.raises(ValueError, match=expected_error):
        LocalisationConfig(observations=["OBS1"], parameters=["PARAM_NODE1"], **data)


def test_invalid_keyword_errors_method_from_file():
    expected_error = "extra fields not permitted"
    data = {
        "log_level": 2,
        "correlations": [
            {
                "name": "CORR1_FROM_FILE",
                "obs_group": {
                    "add": ["OBS1"],
                },
                "param_group": {
                    "add": ["*"],
                },
                "field_scale": {
                    "method": "from_file",
                    "segment_filename": "dummy.GRDECL",
                    "param_name": "Scaling",
                    "active_segments": [1, 2, 3],
                },
            },
        ],
    }
    with pytest.raises(ValueError, match=expected_error):
        LocalisationConfig(observations=["OBS1"], parameters=["PARAM_NODE1"], **data)


def test_invalid_keyword_errors_in_obs_group_or_param_group():
    expected_error = "extra fields not permitted"
    data = {
        "log_level": 2,
        "correlations": [
            {
                "name": "CORR1",
                "obs_group": {
                    "add": ["OBS1"],
                    "unknown_obs_keyword": "dummy",
                },
                "param_group": {
                    "add": ["*"],
                    "unknown_param_keyword": "dummy",
                },
                "field_scale": {
                    "method": "from_file",
                    "segment_filename": "dummy.GRDECL",
                    "param_name": "Scaling",
                    "active_segments": [1, 2, 3],
                },
            },
        ],
    }
    with pytest.raises(ValueError, match=expected_error):
        LocalisationConfig(observations=["OBS1"], parameters=["PARAM_NODE1"], **data)


def test_missing_keyword_errors_method_gaussian_decay():
    expected_error = "correlations -> 0 -> field_scale -> perp_range\n  field required"
    data = {
        "log_level": 2,
        "correlations": [
            {
                "name": "CORR1_SEGMENT",
                "obs_group": {
                    "add": ["OBS1"],
                },
                "param_group": {
                    "add": ["*"],
                },
                "field_scale": {
                    "method": "gaussian_decay",
                    "main_range": 1000,
                },
            },
        ],
    }
    with pytest.raises(ValueError, match=expected_error):
        LocalisationConfig(observations=["OBS1"], parameters=["PARAM_NODE1"], **data)


def test_missing_param():
    expected_error = "correlations -> 0 -> param_group -> result_items"
    data = {
        "log_level": 2,
        "correlations": [
            {
                "name": "CORR",
                "obs_group": {
                    "add": ["OBS1"],
                },
                "param_group": {
                    "add": ["*"],
                    "remove": ["*"],
                },
                "ref_point": [250, 250],
                "field_scale": {
                    "method": "gaussian_decay",
                    "main_range": 1000,
                },
            },
        ],
    }
    with pytest.raises(ValueError, match=expected_error):
        LocalisationConfig(observations=["OBS1"], parameters=["PARAM_NODE1"], **data)


def test_missing_obs():
    expected_error = "correlations -> 0 -> obs_group -> result_items"
    data = {
        "log_level": 2,
        "correlations": [
            {
                "name": "CORR",
                "obs_group": {
                    "add": ["OBS1"],
                    "remove": ["OBS1"],
                },
                "param_group": {
                    "add": ["*"],
                },
                "ref_point": [250, 250],
                "field_scale": {
                    "method": "gaussian_decay",
                    "main_range": 1000,
                },
            },
        ],
    }
    with pytest.raises(ValueError, match=expected_error):
        LocalisationConfig(observations=["OBS1"], parameters=["PARAM_NODE1"], **data)


# User define obs in expanded form
# The nodes are not expanded since the number of obs
# are 3,2 and 1 for GENOBSA,GENOBSB, GENOBSC
# and parameter max_gen_obs_size is 0
def test_gen_obs_expansion_mismatch1():
    obs_group_add = [
        "GENOBSA:0",
        "GENOBSA:1",
        "GENOBSA:2",
        "GENOBSB:0",
        "GENOBSB:1",
        "GENOBSC:0",
    ]
    param_group_add = ["PARAM_NODE1:*"]
    max_size_to_expand = 0
    expected_error = (
        " ['No match for: GENOBSA:0', 'No match for: GENOBSA:1', "
        " 'No match for: GENOBSA:2', 'No match for: GENOBSB:0', "
        " 'No match for: GENOBSB:1', 'No match for: GENOBSC:0'] "
    )
    data = {
        "log_level": 2,
        "max_gen_obs_size": max_size_to_expand,
        "correlations": [
            {
                "name": "some_name",
                "obs_group": {
                    "add": obs_group_add,
                },
                "param_group": {
                    "add": param_group_add,
                },
            }
        ],
    }
    with pytest.raises(ValueError, match=expected_error):
        LocalisationConfig(observations=ERT_GEN_OBS, parameters=ERT_PARAM, **data)


# User define obs are not in expanded form.
# The nodes are expanded since the number of obs
# are 3,2 and 1 for GENOBSA,GENOBSB, GENOBSC
# and parameter max_gen_obs_size is larger than equal to 3.
def test_gen_obs_expansion_mismatch2():
    obs_group_add = [
        "GENOBSA",
        "GENOBSB",
        "GENOBSC",
    ]
    param_group_add = ["PARAM_NODE1:*"]
    max_size_to_expand = 2
    expected_error = (
        "['No match for: GENOBSA', 'No match for: GENOBSB', 'No match for: GENOBSC']"
    )
    data = {
        "log_level": 2,
        "max_gen_obs_size": max_size_to_expand,
        "correlations": [
            {
                "name": "some_name",
                "obs_group": {
                    "add": obs_group_add,
                },
                "param_group": {
                    "add": param_group_add,
                },
            }
        ],
    }
    with pytest.raises(ValueError, match=expected_error):
        LocalisationConfig(
            observations=ERT_GEN_OBS_EXPANDED, parameters=ERT_PARAM, **data
        )
