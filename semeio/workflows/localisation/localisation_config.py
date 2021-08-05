# pylint: disable=E0213
import itertools
import pathlib
from typing import List, Optional, Union, Dict
from typing_extensions import Literal

from pydantic import BaseModel, validator, confloat, conint, conlist, root_validator

from semeio.workflows.localisation.localisation_debug_settings import (
    LocalDebugLog,
    LogLevel,
)


def expand_wildcards(patterns, list_of_words):
    all_matches = []
    errors = []
    for pattern in patterns:
        matches = [
            words for words in list_of_words if pathlib.Path(words).match(pattern)
        ]
        if len(matches) > 0:
            all_matches.extend(matches)
        else:
            errors.append(f"No match for: {pattern}")
    all_matches = set(all_matches)
    if len(errors) > 0:
        raise ValueError(
            " These specifications does not match anything defined in ERT model\n"
            f"     {errors}, available: {list_of_words}"
        )
    return all_matches


def check_for_duplicated_correlation_specifications(correlations):
    # All observations and model parameters used in correlations
    all_combinations = []

    for corr in correlations:
        all_combinations.extend(
            list(
                itertools.product(
                    corr.obs_group.result_items, corr.param_group.result_items
                )
            )
        )
    errors = []
    seen = set()
    for combination in all_combinations:
        if combination in seen:
            errors.append(f"Observation: {combination[0]}, parameter: {combination[1]}")
        else:
            seen.add(combination)
    return errors


def _check_method_param_consistency(method_config):
    if isinstance(method_config, ScalingForSegments):
        # Check consistency between active_segments list and scalingfactors list
        active_segments_list = method_config.active_segments
        scalingfactor_list = method_config.scalingfactors
        print(f"active_segment: {active_segments_list}")
        print(f"scalingfactor: {scalingfactor_list}")
        if len(active_segments_list) != len(scalingfactor_list):
            raise IndexError(
                "The specified length of 'active_segments' list"
                f"{active_segments_list }\n"
                f"  and 'scalingfactors' list {scalingfactor_list} are different."
            )


class ObsConfig(BaseModel):
    add: Union[str, List[str]]
    remove: Optional[Union[str, List[str]]]
    context: List[str]
    result_items: Optional[List[str]]

    @validator("add")
    def validate_add(cls, add):
        if isinstance(add, str):
            add = [add]
        return add

    @validator("remove")
    def validate_remove(cls, remove):
        if isinstance(remove, str):
            remove = [remove]
        return remove

    @validator("result_items", always=True)
    def expanded_items(cls, _, values):
        add, remove = values["add"], values.get("remove", None)
        result = _check_specification(add, remove, values["context"])
        if len(result) == 0:
            raise ValueError(
                f"Adding: {add} and removing: {remove} resulted in no items"
            )
        return result


class ParamConfig(ObsConfig):
    pass


class GaussianConfig(BaseModel):
    method: Literal["gaussian_decay"]
    main_range: confloat(gt=0)
    perp_range: confloat(gt=0)
    azimuth: confloat(ge=0.0, le=360)
    surface_directory: Optional[str] = "."


class ExponentialConfig(GaussianConfig):
    method: Literal["exponential_decay"]


class ScalingFromFile(BaseModel):
    method: Literal["from_file"]
    filename: str
    param_name: str


class ScalingForSegments(BaseModel):
    method: Literal["segment"]
    segment_filename: str
    param_name: str
    active_segments: Union[int, List[int]]
    scalingfactors: Optional[Union[float, List[float]]]

    @validator("active_segments", pre=True)
    def validate_active_segments(cls, values):

        if isinstance(values, int):
            active_segments = [values]
        elif isinstance(values, list):
            active_segments = values
        return active_segments

    @validator("scalingfactors", pre=True)
    def validate_scalingfactors(cls, values):
        if isinstance(values, float):
            if not 0 <= values <= 1:
                raise ValueError(
                    "scalingfactors must be in interval [0.0 ,1.0] in keyword 'segment'"
                )
            scalingfactors = [values]
        elif isinstance(values, list):
            scalingfactors = values
            for v in scalingfactors:
                if not 0 <= v <= 1:
                    raise ValueError(
                        "scalingfactors must be in interval [0.0 ,1.0] "
                        "in keyword 'segment.'"
                    )
        else:
            scalingfactors = None
        return scalingfactors


class CorrelationConfig(BaseModel):
    name: str
    obs_group: ObsConfig
    param_group: ParamConfig
    ref_point: Optional[conlist(float, min_items=2, max_items=2)]
    field_scale: Optional[
        Union[
            GaussianConfig,
            ExponentialConfig,
            ScalingFromFile,
            ScalingForSegments,
        ]
    ]
    surface_scale: Optional[Union[GaussianConfig, ExponentialConfig]]
    obs_context: list
    params_context: list

    @root_validator(pre=True)
    def inject_context(cls, values: Dict) -> Dict:
        values["obs_group"]["context"] = values["obs_context"]
        values["param_group"]["context"] = values["params_context"]
        return values

    @validator("field_scale", pre=True)
    def validate_field_scale(cls, value):
        """
        To improve the user feedback we explicitly check
        which method is configured and bypass the Union
        """
        if isinstance(value, BaseModel):
            return value
        if not isinstance(value, dict):
            raise ValueError("value must be dict")
        method = value.get("method")
        _valid_methods = {
            "gaussian_decay": GaussianConfig,
            "exponential_decay": ExponentialConfig,
            "from_file": ScalingFromFile,
            "segment": ScalingForSegments,
        }
        if method in _valid_methods.keys():
            method_config = _valid_methods[method]
            _check_method_param_consistency(method_config(**value))
            return _valid_methods[method](**value)
        else:
            raise ValueError(
                f"Unknown method: {method}, valid methods are: {_valid_methods.keys()}"
            )

    @root_validator()
    def valid_ref_point_and_scale(cls, values: Dict) -> Dict:
        field_scale = values.get("field_scale", None)
        surface_scale = values.get("surface_scale", None)
        ref_point = values.get("ref_point")
        if field_scale is not None:
            # ref_point is required for method:
            # - gaussian_decay
            # - exponential_decay
            if field_scale.method in ["gaussian_decay", "exponential_decay"]:
                if ref_point is None:
                    raise KeyError(
                        "When using FIELD with scaling of correlation with "
                        f"method {field_scale.method}, "
                        "the reference point must be specified."
                    )
        if surface_scale is not None:
            # ref_point is required for method:
            # - gaussian_decay
            # - exponential_decay
            if surface_scale.method in ["gaussian_decay", "exponential_decay"]:
                if ref_point is None:
                    raise KeyError(
                        "When using SURFACE with scaling of correlation with "
                        f"method {surface_scale.method}, "
                        "the reference point must be specified."
                    )

        return values

    @validator("surface_scale", pre=True)
    def validate_surface_scale(cls, value):
        """
        To improve the user feedback we explicitly check
        which method is configured and bypass the Union
        """
        if isinstance(value, BaseModel):
            return value
        if not isinstance(value, dict):
            raise ValueError("value must be dict")

        # String with relative path to surface files relative to config path
        key = "surface_directory"
        if key not in value.keys():
            # Set default relative directory
            surface_directory = "."
            value[key] = surface_directory

        method = value.get("method")
        _valid_methods = {
            "gaussian_decay": GaussianConfig,
            "exponential_decay": ExponentialConfig,
        }
        if method in _valid_methods.keys():
            return _valid_methods[method](**value)
        else:
            raise ValueError(
                f"Unknown method: {method}, valid methods are: {_valid_methods.keys()}"
            )


class LocalisationConfig(BaseModel):
    """
    observations:  A list of observations from ERT in format nodename
    parameters:    A dict of  parameters from ERT in format nodename:paramname.
                            Key is node name. Values are lists of parameter names
                            for the node.
    correlations:   A list of CorrelationConfig objects keeping name of
                            one correlation set which defines the input to
                            create a ministep object.
    log_level:       Integer defining how much log output to write to screen
    write_scaling_factors: Turn on writing calculated scaling parameters to file.
                            Possible values: True/False. Default: False
    """

    observations: List[str]
    parameters: List[str]
    correlations: List[CorrelationConfig]
    log_level: Optional[conint(ge=0, le=5)] = 1
    write_scaling_factors: Optional[bool] = False

    @validator("log_level")
    def validate_log_level(cls, level):
        LocalDebugLog.level = LogLevel.LEVEL1
        if isinstance(level, int):
            # Change the log level from default to user defined
            LocalDebugLog.level = level
        return level

    @root_validator(pre=True)
    def inject_context(cls, values: Dict) -> Dict:
        for correlation in values["correlations"]:
            correlation["obs_context"] = values["observations"]
            correlation["params_context"] = values["parameters"]
        return values

    @validator("correlations")
    def validate_correlations(cls, correlations):
        duplicates = check_for_duplicated_correlation_specifications(correlations)
        if len(duplicates) > 0:
            error_msgs = "\n".join(duplicates)
            raise ValueError(
                f"Found {len(duplicates)} duplicated correlations: \n{error_msgs}"
            )
        return correlations


def _check_specification(items_to_add, items_to_remove, valid_items):
    added_items = expand_wildcards(items_to_add, valid_items)
    if items_to_remove is not None:
        removed_items = expand_wildcards(items_to_remove, valid_items)
        added_items = added_items.difference(removed_items)
    added_items = list(added_items)
    return sorted(added_items)
