import os
from pathlib import Path
from typing import Optional, Literal, Type, Tuple
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, TomlConfigSettingsSource
from pydantic import Field, validator
import tomli_w  # for writing TOML files


class BarkSettings(BaseSettings):
    """Configuration for Bark notification service."""
    key: str = ""
    server: str = "https://api.day.app"
    enabled: bool = True
    service_name: str = "bark"


class TelegramSettings(BaseSettings):
    """Configuration for Telegram notification service."""
    bot_token: str = ""
    chat_id: str = ""
    enabled: bool = True
    service_name: str = "telegram"


class SlackSettings(BaseSettings):
    """Configuration for Slack notification service."""
    webhook_url: str = ""
    enabled: bool = True
    service_name: str = "slack"


class NotificationSettings(BaseSettings):
    """Configuration for notification system."""
    service: Literal["none", "bark", "telegram", "slack"] = "none"
    notify_on_gpu_found: bool = True
    notify_on_task_complete: bool = True
    bark: Optional[BarkSettings] = Field(default_factory=BarkSettings)
    telegram: Optional[TelegramSettings] = Field(default_factory=TelegramSettings)
    slack: Optional[SlackSettings] = Field(default_factory=SlackSettings)

    @validator("service")
    def validate_service(cls, v, values):
        """Validate that the selected service is enabled."""
        if v == "bark" and not values.get("bark", {}).get("enabled", True):
            raise ValueError("Bark service is disabled")
        if v == "telegram" and not values.get("telegram", {}).get("enabled", True):
            raise ValueError("Telegram service is disabled")
        if v == "slack" and not values.get("slack", {}).get("enabled", True):
            raise ValueError("Slack service is disabled")
        return v


class Settings(BaseSettings):
    """Main application settings."""
    model_config = SettingsConfigDict(
        toml_file=Path(os.path.expanduser("~")) / ".config" / "grun" / "config.toml",
        toml_file_encoding="utf-8",
    )

    # Default check interval in seconds
    check_interval: float = 1.0
    
    # Notification settings
    notification: NotificationSettings = Field(default_factory=NotificationSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls),)

    @classmethod
    def get_settings(cls, config_path: Optional[Path] = None) -> "Settings":
        """Load settings from config file and environment variables.
        
        Args:
            config_path: Optional path to a specific config file to load.
                        If None, uses the default config location.
        """
        if config_path is None:
            config_dir = Path(os.path.expanduser("~")) / ".config" / "grun"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / "config.toml"
        
        if config_path.exists():
            return cls()
        else:
            # Create default config file if it doesn't exist
            settings = cls()
            settings.create_default_config(config_path)
            return settings

    def create_default_config(self, config_path: Path) -> None:
        """Create a default configuration file with current settings.
        
        Args:
            config_path: Path where to create the config file
        """
        # Convert settings to dict
        config_dict = self.model_dump()
        
        # Write to TOML file
        with open(config_path, "w") as f:
            tomli_w.dump(config_dict, f)

    @classmethod
    def from_file(cls, config_path: Path) -> "Settings":
        """Load settings from a specific config file.
        
        Args:
            config_path: Path to the config file to load
            
        Returns:
            Settings: Loaded settings instance
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return cls(_env_file=str(config_path))


# Global settings instance
settings = Settings.get_settings() 