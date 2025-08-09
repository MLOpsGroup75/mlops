"""
Unity Catalog utilities for Databricks MLflow model registration.
"""

import os
import logging
import mlflow
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class UnityCatalogModelRegistry:
    """Handles model registration with Unity Catalog compatibility."""

    def __init__(self, catalog_name: str = "main", schema_name: str = "mlops"):
        """
        Initialize Unity Catalog model registry.

        Args:
            catalog_name: Unity Catalog name (default: 'main')
            schema_name: Schema name within catalog (default: 'mlops')
        """
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.is_unity_catalog = self._check_unity_catalog_availability()

    def _check_unity_catalog_availability(self) -> bool:
        """Check if Unity Catalog is available and enabled."""
        try:
            # This is a simple check - in practice you might need more sophisticated detection
            tracking_uri = mlflow.get_tracking_uri()
            if "databricks" in tracking_uri:
                logger.info("Unity Catalog mode detected (Databricks workspace)")
                # Set registry URI to Unity Catalog
                try:
                    mlflow.set_registry_uri("databricks-uc")
                    logger.info("✅ MLflow registry URI set to Unity Catalog")
                except Exception as e:
                    logger.warning(f"Could not set Unity Catalog registry URI: {e}")
                return True
            else:
                logger.info("Non-Unity Catalog mode (local MLflow)")
                return False
        except Exception as e:
            logger.warning(f"Could not determine Unity Catalog status: {e}")
            return False

    def _sanitize_model_name(self, name: str) -> str:
        """
        Sanitize model name to comply with Unity Catalog naming rules.
        
        Unity Catalog model names must be non-empty UTF-8 strings and cannot contain
        forward slashes (/), periods (.), or colons (:).
        
        Args:
            name: Original model name
            
        Returns:
            Sanitized model name
        """
        # Replace invalid characters with underscores
        sanitized = name.replace("/", "_").replace(".", "_").replace(":", "_")
        
        # Remove any duplicate underscores and strip
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        
        sanitized = sanitized.strip("_")
        
        # Ensure name is not empty
        if not sanitized:
            sanitized = "model"
            
        return sanitized

    def get_model_name(self, base_name: str) -> str:
        """
        Get proper model name format for Unity Catalog.

        Args:
            base_name: Base model name

        Returns:
            Properly formatted model name
        """
        # Sanitize the base name first
        sanitized_name = self._sanitize_model_name(base_name)
        
        if self.is_unity_catalog:
            # Unity Catalog format: catalog.schema.model_name
            return f"{self.catalog_name}.{self.schema_name}.{sanitized_name}"
        else:
            # Legacy format: just the sanitized model name
            return sanitized_name

    def register_model_safely(
        self,
        model_uri: str,
        model_name: str,
        description: str = None,
        tags: Dict[str, str] = None,
    ) -> Optional[str]:
        """
        Register model with proper error handling for Unity Catalog.

        Args:
            model_uri: MLflow model URI
            model_name: Base model name
            description: Model description
            tags: Model tags

        Returns:
            Model version if successful, None if failed
        """
        try:
            # Validate original model name and log if sanitization is needed
            is_valid, validation_msg = validate_unity_catalog_model_name(model_name)
            if not is_valid:
                logger.warning(f"⚠️ Model name validation failed: {validation_msg}")
                logger.info(f"🔧 Sanitizing model name for Unity Catalog compatibility")
            
            # Get proper model name format
            full_model_name = self.get_model_name(model_name)
            
            if model_name != self._sanitize_model_name(model_name):
                logger.info(f"Model name sanitized: '{model_name}' -> '{self._sanitize_model_name(model_name)}'")

            logger.info(f"Attempting to register model: {full_model_name}")

            # Try to register the model
            model_version = mlflow.register_model(
                model_uri=model_uri, name=full_model_name
            )

            logger.info(
                f"✅ Successfully registered model: {full_model_name}, version: {model_version.version}"
            )

            # Add tags if provided and possible
            if tags and hasattr(model_version, "version"):
                try:
                    client = mlflow.MlflowClient()
                    for key, value in tags.items():
                        client.set_model_version_tag(
                            name=full_model_name,
                            version=model_version.version,
                            key=key,
                            value=value,
                        )
                    logger.info(
                        f"✅ Added tags to model version {model_version.version}"
                    )
                except Exception as tag_error:
                    logger.warning(f"Could not add tags: {tag_error}")

            return model_version.version

        except Exception as e:
            error_msg = str(e)

            if "INVALID_PARAMETER_VALUE" in error_msg and "Invalid name" in error_msg:
                logger.warning("⚠️ Invalid model name for Unity Catalog")
                logger.info(f"💡 Original name: {model_name}")
                logger.info(f"💡 Attempted name: {full_model_name}")
                logger.info("💡 Model names cannot contain periods (.), forward slashes (/), or colons (:)")
                logger.info("💡 Model artifacts are still available in MLflow run")
                return "not_registered_invalid_name"

            elif "legacy workspace model registry is disabled" in error_msg:
                logger.warning(
                    "⚠️ Legacy model registry disabled - model logged but not registered"
                )
                logger.info("💡 Model artifacts are still available in MLflow run")
                logger.info("💡 You can manually register in Unity Catalog later")
                return "not_registered_legacy_disabled"

            elif "PERMISSION_DENIED" in error_msg:
                logger.warning("⚠️ Permission denied for model registration")
                logger.info("💡 Model artifacts are still available in MLflow run")
                logger.info(
                    "💡 Check Unity Catalog permissions or ask admin to register"
                )
                return "not_registered_permission_denied"

            elif "does not exist" in error_msg and "catalog" in error_msg:
                logger.warning("⚠️ Unity Catalog or schema does not exist")
                logger.info(
                    f"💡 Try creating catalog '{self.catalog_name}' and schema '{self.schema_name}'"
                )
                logger.info("💡 Or model artifacts are available in MLflow run")
                return "not_registered_catalog_missing"

            else:
                logger.error(f"❌ Model registration failed: {e}")
                return None

    def suggest_unity_catalog_setup(self):
        """Provide suggestions for Unity Catalog setup."""
        logger.info("\n📋 Unity Catalog Setup Suggestions:")
        logger.info("1. Ask your Databricks admin to:")
        logger.info(f"   - Create catalog '{self.catalog_name}' (if not exists)")
        logger.info(
            f"   - Create schema '{self.catalog_name}.{self.schema_name}' (if not exists)"
        )
        logger.info("   - Grant you CREATE MODEL permissions")

        logger.info("\n2. Alternative options:")
        logger.info("   - Use different catalog/schema names you have access to")
        logger.info("   - Manually register models through Databricks UI")
        logger.info("   - Use model artifacts directly from MLflow runs")

        logger.info(f"\n3. Model artifacts location:")
        logger.info("   - Available in MLflow run artifacts")
        logger.info("   - Can be downloaded and used for inference")
        logger.info("   - No registration required for model usage")


def register_model_with_unity_catalog(
    model_uri: str,
    model_name: str,
    catalog: str = "main",
    schema: str = "mlops",
    description: str = None,
    tags: Dict[str, str] = None,
) -> Optional[str]:
    """
    Convenience function to register model with Unity Catalog handling.

    Args:
        model_uri: MLflow model URI
        model_name: Base model name
        catalog: Unity Catalog name
        schema: Schema name
        description: Model description
        tags: Model tags

    Returns:
        Model version if successful
    """
    registry = UnityCatalogModelRegistry(catalog, schema)
    return registry.register_model_safely(model_uri, model_name, description, tags)


def validate_unity_catalog_model_name(model_name: str) -> Tuple[bool, str]:
    """
    Validate if a model name complies with Unity Catalog naming rules.
    
    Args:
        model_name: Model name to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not model_name:
        return False, "Model name cannot be empty"
        
    if not isinstance(model_name, str):
        return False, "Model name must be a string"
        
    # Check for invalid characters
    invalid_chars = ["/", ".", ":"]
    found_invalid = []
    
    for char in invalid_chars:
        if char in model_name:
            found_invalid.append(char)
            
    if found_invalid:
        return False, f"Model name contains invalid characters: {', '.join(found_invalid)}. Unity Catalog model names cannot contain forward slashes (/), periods (.), or colons (:)"
        
    return True, "Valid model name"


def check_unity_catalog_permissions(
    catalog: str = "main", schema: str = "mlops"
) -> bool:
    """
    Check if Unity Catalog permissions are properly set up.

    Args:
        catalog: Catalog name to check
        schema: Schema name to check

    Returns:
        True if permissions are adequate
    """
    try:
        # Try to list models (read permission check)
        client = mlflow.MlflowClient()

        # Try with Unity Catalog format
        test_name = f"{catalog}.{schema}.permission_test"

        try:
            # This will fail if we don't have permissions, but that's expected
            models = client.search_registered_models(
                filter_string=f"name='{test_name}'"
            )
            logger.info("✅ Unity Catalog read permissions verified")
            return True
        except Exception as e:
            if "PERMISSION_DENIED" in str(e):
                logger.warning("❌ Insufficient Unity Catalog permissions")
                return False
            else:
                # Other errors might be due to catalog/schema not existing
                logger.info(
                    "⚠️ Unity Catalog permissions unclear (catalog/schema might not exist)"
                )
                return False

    except Exception as e:
        logger.error(f"❌ Error checking Unity Catalog permissions: {e}")
        return False
