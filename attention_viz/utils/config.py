"""Configuration management for attention visualization."""

import yaml
import os
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    default_colormap: str = "viridis"
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    interactive: bool = True
    save_format: str = "png"


@dataclass
class ExportConfig:
    """Configuration for data export settings."""
    default_format: str = "json"
    include_metadata: bool = True
    compression: bool = False


@dataclass 
class ModelConfig:
    """Configuration for model inference settings."""
    max_length: int = 512
    batch_size: int = 1
    device: str = "auto"
    cache_attention: bool = True


@dataclass
class Config:
    """Main configuration class for attention visualization."""
    
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    export: ExportConfig = field(default_factory=ExportConfig) 
    model: ModelConfig = field(default_factory=ModelConfig)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Config instance with loaded settings
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create Config instance from dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            Config instance
        """
        # Extract sub-configurations
        viz_config = VisualizationConfig(**config_dict.get("visualization", {}))
        export_config = ExportConfig(**config_dict.get("export", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        
        return cls(
            visualization=viz_config,
            export=export_config,
            model=model_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Config to dictionary.
        
        Returns:
            Dictionary representation of config
        """
        return {
            "visualization": {
                "default_colormap": self.visualization.default_colormap,
                "figure_size": list(self.visualization.figure_size),
                "dpi": self.visualization.dpi,
                "interactive": self.visualization.interactive,
                "save_format": self.visualization.save_format
            },
            "export": {
                "default_format": self.export.default_format,
                "include_metadata": self.export.include_metadata,
                "compression": self.export.compression
            },
            "model": {
                "max_length": self.model.max_length,
                "batch_size": self.model.batch_size,
                "device": self.model.device,
                "cache_attention": self.model.cache_attention
            }
        }
    
    def save_to_file(self, config_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config_path: Path to save configuration file
        """
        config_dict = self.to_dict()
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary.
        
        Args:
            updates: Dictionary with configuration updates
        """
        if "visualization" in updates:
            viz_updates = updates["visualization"]
            for key, value in viz_updates.items():
                if hasattr(self.visualization, key):
                    setattr(self.visualization, key, value)
        
        if "export" in updates:
            export_updates = updates["export"]
            for key, value in export_updates.items():
                if hasattr(self.export, key):
                    setattr(self.export, key, value)
        
        if "model" in updates:
            model_updates = updates["model"]
            for key, value in model_updates.items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
    
    def get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        return os.path.join(os.getcwd(), "attention_viz_config.yaml")
    
    def create_default_config(self, save_path: Optional[str] = None) -> str:
        """
        Create a default configuration file.
        
        Args:
            save_path: Path to save the config file (if None, uses default path)
            
        Returns:
            Path to the created configuration file
        """
        if save_path is None:
            save_path = self.get_default_config_path()
        
        self.save_to_file(save_path)
        return save_path


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or create default.
    
    Args:
        config_path: Path to configuration file (if None, looks for default)
        
    Returns:
        Config instance
    """
    if config_path is None:
        default_path = Config().get_default_config_path()
        if os.path.exists(default_path):
            config_path = default_path
        else:
            # Return default config if no file exists
            return Config()
    
    if os.path.exists(config_path):
        return Config.from_file(config_path)
    else:
        # Create default config file
        config = Config()
        config.create_default_config(config_path)
        return config 