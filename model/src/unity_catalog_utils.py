"""
Unity Catalog utilities for Databricks MLflow model registration.
"""

import os
import logging
import mlflow
from typing import Optional, Dict, Any

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
                return True
            else:
                logger.info("Non-Unity Catalog mode (local MLflow)")
                return False
        except Exception as e:
            logger.warning(f"Could not determine Unity Catalog status: {e}")
            return False

    def get_model_name(self, base_name: str) -> str:
        """
        Get proper model name format for Unity Catalog.

        Args:
            base_name: Base model name

        Returns:
            Properly formatted model name
        """
        if self.is_unity_catalog:
            # Unity Catalog format: catalog.schema.model_name
            return f"{self.catalog_name}.{self.schema_name}.{base_name}"
        else:
            # Legacy format: just the model name
            return base_name

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
            # Get proper model name format
            full_model_name = self.get_model_name(model_name)

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

            if "legacy workspace model registry is disabled" in error_msg:
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
