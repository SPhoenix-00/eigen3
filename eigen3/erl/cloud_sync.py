"""Cloud storage synchronization for HoF agent weights and ledger files.

Supports Google Cloud Storage (GCS) with local-only fallback.  Adapted from
Eigen2's ``utils/cloud_sync.py`` — trimmed to the subset required by the
Hall of Fame (upload, download, delete, exists, verified upload).

Environment variables (GCS):
    CLOUD_PROVIDER   = "gcs"
    CLOUD_BUCKET     = "<bucket-name>"
    GOOGLE_APPLICATION_CREDENTIALS — optional path to the service-account JSON.
    When unset with CLOUD_PROVIDER=gcs, from_env() uses repo-root/gcs-credentials.json
    if that file exists (repo root is the parent of the eigen3 package directory).
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)

CloudProvider = Literal["gcs", "local"]


class CloudSync:
    """Upload / download files to GCS (or stay local-only)."""

    def __init__(
        self,
        provider: CloudProvider = "local",
        bucket_name: Optional[str] = None,
        project_name: str = "eigen3",
        credentials_path: Optional[str] = None,
        max_workers: int = 4,
    ):
        self.provider = provider
        self.bucket_name = bucket_name
        self.project_name = project_name
        self.credentials_path = credentials_path
        self.client = None
        self.bucket = None

        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="CloudSync"
        )
        self._lock = threading.Lock()

        if provider != "local":
            self._init_client()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_client(self) -> None:
        if self.provider == "gcs":
            try:
                from google.cloud import storage  # type: ignore[import-untyped]

                if self.credentials_path:
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
                self.client = storage.Client()
                self.bucket = self.client.bucket(self.bucket_name)
                logger.info("Connected to GCS bucket: %s", self.bucket_name)
            except ImportError:
                logger.warning(
                    "google-cloud-storage not installed — falling back to local. "
                    "pip install google-cloud-storage"
                )
                self.provider = "local"
            except Exception as exc:
                logger.warning("GCS connection failed (%s) — falling back to local", exc)
                self.provider = "local"

    @classmethod
    def from_env(cls, project_name: str = "eigen3", **kwargs) -> "CloudSync":
        """Create a ``CloudSync`` from standard env vars."""
        provider = os.environ.get("CLOUD_PROVIDER", "local").lower()
        bucket = os.environ.get("CLOUD_BUCKET")
        creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds and provider == "gcs":
            default_key = Path(__file__).resolve().parents[2] / "gcs-credentials.json"
            if default_key.is_file():
                creds = str(default_key)
        if provider not in ("gcs", "local"):
            logger.warning("Unsupported CLOUD_PROVIDER=%s, using local", provider)
            provider = "local"
        return cls(
            provider=provider,  # type: ignore[arg-type]
            bucket_name=bucket,
            project_name=project_name,
            credentials_path=creds,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def upload_file(
        self, local_path: str, cloud_path: str, *, background: bool = True
    ) -> bool:
        """Upload *local_path* to *cloud_path*.  Returns True on success."""
        if self.provider == "local":
            return True
        if background:
            self._executor.submit(self._upload, local_path, cloud_path)
            return True
        return self._upload(local_path, cloud_path)

    def _upload(self, local_path: str, cloud_path: str) -> bool:
        try:
            blob = self.bucket.blob(cloud_path)
            blob.upload_from_filename(local_path)
            return True
        except Exception as exc:
            logger.warning("Upload failed %s -> %s: %s", local_path, cloud_path, exc)
            return False

    def upload_file_verified(self, local_path: str, cloud_path: str) -> bool:
        """Upload then verify by comparing MD5 hashes."""
        if self.provider == "local":
            return True
        if not self._upload(local_path, cloud_path):
            return False
        try:
            blob = self.bucket.blob(cloud_path)
            blob.reload()
            remote_md5 = blob.md5_hash  # base64-encoded MD5
            import base64

            with open(local_path, "rb") as f:
                local_md5 = base64.b64encode(
                    hashlib.md5(f.read()).digest()
                ).decode()
            if remote_md5 != local_md5:
                logger.warning(
                    "Verified upload MD5 mismatch for %s (local=%s remote=%s)",
                    cloud_path, local_md5, remote_md5,
                )
                return False
            return True
        except Exception as exc:
            logger.warning("Verification failed for %s: %s", cloud_path, exc)
            return False

    def download_file(
        self, cloud_path: str, local_path: str, *, silent: bool = False
    ) -> bool:
        """Download *cloud_path* to *local_path*.  Returns True on success."""
        if self.provider == "local":
            return False
        try:
            blob = self.bucket.blob(cloud_path)
            if not blob.exists():
                return False
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(local_path)
            return True
        except Exception as exc:
            if not silent:
                logger.warning("Download failed %s: %s", cloud_path, exc)
            return False

    def file_exists(self, cloud_path: str) -> bool:
        if self.provider == "local":
            return False
        try:
            return self.bucket.blob(cloud_path).exists()
        except Exception:
            return False

    def delete_file(self, cloud_path: str) -> bool:
        if self.provider == "local":
            return True
        try:
            blob = self.bucket.blob(cloud_path)
            if blob.exists():
                blob.delete()
            return True
        except Exception as exc:
            logger.warning("Delete failed %s: %s", cloud_path, exc)
            return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Wait for pending background uploads then shut down the pool."""
        self._executor.shutdown(wait=True)
