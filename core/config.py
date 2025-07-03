import logging
import logging.config
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class GPUConfig(BaseSettings):
    auto_detect: bool = True
    memory_buffer: int = 300  # MB to keep free
    max_utilization: float = 1.0


class SchedulerConfig(BaseSettings):
    queue_size: int = 1000
    job_timeout: int = 1800  # 30 minutes
    cleanup_interval: int = 300  # 5 minutes
    priority_weights: Dict[str, float] = {
        "mesh_generation": 1.0,
        "texture_generation": 0.8,
        "segmentation": 0.6,
        "auto_rigging": 0.4,
    }


class StorageConfig(BaseSettings):
    input_dir: str = "/tmp/inputs"
    output_dir: str = "/tmp/outputs"
    log_dir: str = "logs"
    cleanup_after: int = 3600  # 1 hour
    max_file_size: str = "500MB"


class ServerConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 7842
    workers: int = 4
    max_request_size: str = "100MB"
    timeout: int = 300


class LoggingConfig(BaseSettings):
    """Enhanced logging configuration that supports both simple and dictConfig formats"""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    # For dictConfig support
    use_dict_config: bool = False
    dict_config_path: Optional[str] = None
    dict_config: Optional[Dict] = None


class SecurityConfig(BaseSettings):
    api_key_required: bool = False
    rate_limit_per_minute: int = 60
    cors_origins: List[str] = ["*"]
    jwt_secret: Optional[str] = None


class ModelConfig(BaseSettings):
    """Configuration for a single model"""

    vram_requirement: int
    supported_inputs: List[str]
    supported_outputs: List[str]
    max_workers: int = 1
    model_path: Optional[str] = None
    enabled: bool = True

    model_config = SettingsConfigDict(protected_namespaces=("settings_",))


class Settings(BaseSettings):
    """Main settings class"""

    # Core configurations
    server: ServerConfig = ServerConfig()
    gpu: GPUConfig = GPUConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    storage: StorageConfig = StorageConfig()
    logging: LoggingConfig = LoggingConfig()
    security: SecurityConfig = SecurityConfig()

    # Model configurations
    models: Dict[str, Dict[str, ModelConfig]] = {}

    # Environment
    environment: str = "development"
    debug: bool = False

    model_config = SettingsConfigDict(env_prefix="P3D_", case_sensitive=False)

    @field_validator("models")
    def parse_models(cls, v):
        """Parse model configurations from nested dict"""
        if isinstance(v, dict):
            parsed = {}
            for feature, models in v.items():
                parsed[feature] = {}
                for model_id, config in models.items():
                    if isinstance(config, dict):
                        parsed[feature][model_id] = ModelConfig(**config)
                    else:
                        parsed[feature][model_id] = config
            return parsed
        return v

    def get_model_config(self, feature: str, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration by feature and model ID"""
        if feature in self.models and model_id in self.models[feature]:
            return self.models[feature][model_id]
        return None

    def get_feature_models(self, feature: str) -> Dict[str, ModelConfig]:
        """Get all models for a specific feature"""
        return self.models.get(feature, {})

    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models by feature"""
        result = {}
        for feature, models in self.models.items():
            result[feature] = [
                model_id for model_id, config in models.items() if config.enabled
            ]
        return result


def load_config_from_file(config_path: str) -> Settings:
    """Load configuration from YAML file"""
    config_file = Path(config_path)

    if not config_file.exists():
        logger.warning(f"Config file {config_path} not found, using defaults")
        return Settings()

    try:
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)

        # Load system config
        settings = Settings()

        # Update configurations if they exist in the file
        if "server" in config_data:
            settings.server = ServerConfig(**config_data["server"])
        if "gpu" in config_data:
            settings.gpu = GPUConfig(**config_data["gpu"])
        if "scheduler" in config_data:
            settings.scheduler = SchedulerConfig(**config_data["scheduler"])
        if "storage" in config_data:
            settings.storage = StorageConfig(**config_data["storage"])
        if "logging" in config_data:
            settings.logging = LoggingConfig(**config_data["logging"])
        if "security" in config_data:
            settings.security = SecurityConfig(**config_data["security"])

        # Load models config
        if "models" in config_data:
            settings.models = config_data["models"]
            settings = Settings(**settings.dict())  # Re-validate

        # Update other settings
        if "environment" in config_data:
            settings.environment = config_data["environment"]
        if "debug" in config_data:
            settings.debug = config_data["debug"]

        logger.info(f"Successfully loaded configuration from {config_path}")
        return settings

    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {str(e)}")
        logger.info("Using default configuration")
        return Settings()


def load_models_config(models_config_path: str) -> Dict[str, Dict[str, ModelConfig]]:
    """Load models configuration from separate YAML file"""
    config_file = Path(models_config_path)

    if not config_file.exists():
        logger.warning(f"Models config file {models_config_path} not found")
        return {}

    try:
        with open(config_file, "r") as f:
            models_data = yaml.safe_load(f)

        parsed_models = {}
        for feature, models in models_data.items():
            parsed_models[feature] = {}
            for model_id, config in models.items():
                parsed_models[feature][model_id] = ModelConfig(**config)

        logger.info(
            f"Successfully loaded models configuration from {models_config_path}"
        )
        return parsed_models

    except Exception as e:
        logger.error(f"Error loading models config from {models_config_path}: {str(e)}")
        return {}


def load_logging_dict_config(config_path: str) -> Optional[Dict]:
    """Load logging configuration from YAML file in dictConfig format"""
    config_file = Path(config_path)

    if not config_file.exists():
        logger.warning(f"Logging config file {config_path} not found")
        return None

    try:
        with open(config_file, "r") as f:
            logging_config = yaml.safe_load(f)

        # Ensure logs directory exists
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        return logging_config
    except Exception as e:
        logger.error(f"Error loading logging config from {config_path}: {str(e)}")
        return None


# Global settings instance
settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance"""
    global settings
    if settings is None:
        # Try to load from config files
        config_dir = Path(__file__).parent.parent / "config"
        system_config = config_dir / "system.yaml"
        models_config = config_dir / "models.yaml"

        settings = load_config_from_file(str(system_config))

        # Load models separately if exists
        if models_config.exists():
            models = load_models_config(str(models_config))
            settings.models = models

    return settings


def setup_logging(config: LoggingConfig):
    """Setup logging configuration with support for both simple and dictConfig formats"""

    # Try to use dictConfig first if available
    config_dir = Path(__file__).parent.parent / "config"
    logging_yaml_path = config_dir / "logging.yaml"

    if logging_yaml_path.exists():
        # Use the YAML logging configuration file
        dict_config = load_logging_dict_config(str(logging_yaml_path))
        if dict_config:
            try:
                logging.config.dictConfig(dict_config)
                logger.info(f"Logging configured from YAML: {logging_yaml_path}")
                return
            except Exception as e:
                logger.error(f"Failed to configure logging from YAML: {str(e)}")
                logger.info("Falling back to simple logging configuration")

    # Fallback to simple configuration
    level = getattr(logging, config.level.upper())

    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    handlers: List[logging.Handler] = [logging.StreamHandler()]

    # Add file handler if specified
    if config.file:
        file_path = Path(config.file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(config.file))
    else:
        # Default log file
        default_log_file = logs_dir / "app.log"
        handlers.append(logging.FileHandler(str(default_log_file)))

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=config.format,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Configure uvicorn logger
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(level)

    logger.info(
        f"Logging configured: level={config.level}, file={config.file or 'logs/app.log'}"
    )


def create_directories(storage_config: StorageConfig):
    """Create necessary directories"""
    directories = [
        storage_config.input_dir,
        storage_config.output_dir,
        storage_config.log_dir,
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created/verified directory: {directory}")
