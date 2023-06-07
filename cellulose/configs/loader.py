# Standard imports
import logging
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Third party imports
import tomlkit
from pydantic.dataclasses import dataclass

# Cellulose imports
from cellulose.configs.config import (
    AllConfig,
    BenchmarkConfig,
    DecoratorConfig,
    ExportConfig,
    OutputConfig,
)

DEFAULT_CELLULOSE_CONFIG_FILENAME = ".cellulose_config.toml"
DEFAULT_CELLULOSE_CONFIG_PATH = Path.home()

logger = logging.getLogger(__name__)


def get_cellulose_config_path() -> Path:
    """
    Returns the current CELLULOSE_CONFIG_PATH env variable, if defined.
    Will return DEFAULT_CELLULOSE_CONFIG_PATH otherwise.

    Returns
    -------
    CELLULOSE_CONFIG_PATH: pathlib.Path
    """

    if os.getenv("CELLULOSE_CONFIG_PATH") is None:
        warning_msg = "$CELLULOSE_CONFIG_PATH not set, defaulting to {default_cellulose_config_path} instead".format(  # noqa: E501
            default_cellulose_config_path=DEFAULT_CELLULOSE_CONFIG_PATH,
        )
        logger.warning(warning_msg)
        return DEFAULT_CELLULOSE_CONFIG_PATH
    else:
        return Path(str(os.getenv("CELLULOSE_CONFIG_PATH")))


@dataclass
class ConfigLoader:
    absolute_config_path: Path = get_cellulose_config_path()
    cellulose_config: AllConfig | None = None

    def parse_user_config(self) -> dict[str, Any]:
        """
        This method parses the user specified Cellulose configs and returns
        them as a dictionary.
        """
        cellulose_config_toml = None
        full_path = (
            "{absolute_config_path}/{cellulose_config_filename}".format(
                absolute_config_path=self.absolute_config_path,
                cellulose_config_filename=DEFAULT_CELLULOSE_CONFIG_FILENAME,
            )
        )
        try:
            cellulose_config_toml = tomlkit.loads(Path(full_path).read_text())
        except FileNotFoundError as e:
            warning_msg = "Could not find Cellulose config file in {full_path}, defaulting to Cellulose's default configurations".format(  # noqa: E501
                full_path=full_path
            )
            logger.warning(warning_msg)
            logger.debug(e)

        if cellulose_config_toml is None:
            return dict()
        return cellulose_config_toml

    def parse_cellulose_internal_config(self) -> dict[str, Any]:
        """
        This method parses the internal Cellulose configs and returns
        them as a dictionary.
        """
        cellulose_config_toml = None

        full_path = os.path.join(
            os.path.dirname(__file__),
            "{cellulose_config_filename}".format(
                cellulose_config_filename=DEFAULT_CELLULOSE_CONFIG_FILENAME,
            ),
        )
        try:
            cellulose_config_toml = tomlkit.loads(Path(full_path).read_text())
        except FileNotFoundError as e:
            error_msg = "Could not find internal Cellulose config file!"
            logger.error(error_msg)
            raise FileNotFoundError(e)

        return cellulose_config_toml

    def parse_decorator_config(
        self, decorator_dict: dict[str, Any]
    ) -> AllConfig:
        if self.cellulose_config is None:
            error_msg = (
                "Internal: Expected type(cellulose_config) attribute to be"
            )
            error_msg += "AllConfig, got None instead"
            logger.error(error_msg)
            raise Exception(error_msg)

        result_dict = deepcopy(self.cellulose_config)

        # Note that decorator_dict is "flat", ie. no TOML config file
        # "section" hierarchy.
        # This is why we can just update the internal dicts this way.

        # Update export_config
        result_dict.export_config.update(decorator_dict)

        # Update benchmark_config
        result_dict.benchmark_config.update(decorator_dict)

        # Update output_config
        result_dict.output_config.update(decorator_dict)

        # Create new instance of DecoratorConfig
        result_dict.decorator_config = DecoratorConfig(**decorator_dict)

        return result_dict

    def override_with_provided_config_dict(
        self, all_config: AllConfig, config_dict: dict[str, Any]
    ):
        """
        Internal helper method to override AllConfig instance with a provided
        dict[str, Any]
        """
        # Update export_config
        if "exports" in config_dict:
            user_export_config = config_dict["exports"]
            all_config.export_config.update(user_export_config)

        # Update benchmark_config
        if "benchmarks" in config_dict:
            user_benchmark_config = config_dict["benchmarks"]
            all_config.benchmark_config.update(user_benchmark_config)

        # Update output_config
        if "outputs" in config_dict:
            user_output_config = config_dict["outputs"]
            all_config.output_config.update(user_output_config)

        return all_config

    def override_internal_with_user_config(
        self, user_config_dict: dict[str, Any]
    ):
        """
        This method overrides the internal, in-memory config values
        with anything explicitly defined by the user_configs.
        """
        if self.cellulose_config is None:
            error_msg = "Internal: Expected type(cellulose_config) attribute"
            error_msg += "to be AllConfig, got None instead"
            logger.error(error_msg)
            raise Exception(error_msg)

        self.cellulose_config = self.override_with_provided_config_dict(
            all_config=self.cellulose_config, config_dict=user_config_dict
        )

    def parse(self):
        """
        This method parses the input configs based on the config path
        and maintains an in memory representation of it.
        """
        cellulose_internal_config_dict = self.parse_cellulose_internal_config()

        # We first initialize based on Cellulose internal default
        # configurations.
        self.cellulose_config = AllConfig(
            export_config=ExportConfig(
                **cellulose_internal_config_dict["exports"]
            ),
            benchmark_config=BenchmarkConfig(
                **cellulose_internal_config_dict["benchmarks"]
            ),
            output_config=OutputConfig(
                **cellulose_internal_config_dict["outputs"]
            ),
        )
        logger.debug("Cellulose internal configs:")
        logger.debug(asdict(self.cellulose_config))

        # Then now we parse user configs
        # then override internal configs with them.
        user_config_dict = self.parse_user_config()
        self.override_internal_with_user_config(
            user_config_dict=user_config_dict
        )

        logger.debug("Cellulose configs after user config updates:")
        logger.debug(asdict(self.cellulose_config))
