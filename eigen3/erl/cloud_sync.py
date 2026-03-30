"""Cloud storage synchronization for HoF agent weights and ledger files.

Supports Google Cloud Storage (GCS) with local-only fallback.  Adapted from
Eigen2's ``utils/cloud_sync.py`` — trimmed to the subset required by the
Hall of Fame (upload, download, delete, exists, verified upload).

Training never depends on cloud success: uploads are best-effort, failures are
logged, and nothing in the training loop waits on verification or completion.
Replay buffers and full ``training_state.pkl`` checkpoints are never uploaded.

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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)

CloudProvider = Literal["gcs", "local"]


def is_forbidden_cloud_upload_local_path(local_path: str) -> bool:
    """True for full training checkpoints (replay + env state); never upload these."""
    name = Path(local_path).name.lower()
    if name in ("training_state.pkl", "training_state.pkl.tmp"):
        return True
    if name.startswith("training_state") and name.endswith(".pkl"):
        return True
    return False


class CloudSync:
    """Best-effort GCS uploads. Callers must not gate training on return values or completion."""

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
                logger.error(
                    "CLOUD SYNC DISABLED: package missing. "
                    "Install google-cloud-storage to enable GCS uploads. "
                    "(Training and local HoF files are unaffected.)"
                )
                self.provider = "local"
            except Exception as exc:
                logger.error(
                    "CLOUD SYNC DISABLED: GCS client or bucket init failed (%s). "
                    "Falling back to local-only; training continues.",
                    exc,
                )
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
            logger.error(
                "CLOUD SYNC DISABLED: unsupported CLOUD_PROVIDER=%r (use gcs or local). "
                "Using local-only.",
                provider,
            )
            provider = "local"
        if provider == "gcs" and not bucket:
            logger.error(
                "CLOUD SYNC DISABLED: CLOUD_PROVIDER=gcs but CLOUD_BUCKET is unset. "
                "HoF uploads are off; set CLOUD_BUCKET to enable."
            )
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
        """Schedule or run an upload. Always returns True for local provider.

        On GCS, failures are logged only; return value is informational for
        foreground uploads. Background scheduling errors do not propagate.
        """
        if self.provider == "local":
            return True
        if is_forbidden_cloud_upload_local_path(local_path):
            logger.warning(
                "Skipping cloud upload (replay/training state stays local-only): %s",
                local_path,
            )
            return True
        if background:
            try:
                self._executor.submit(self._upload, local_path, cloud_path)
            except RuntimeError as exc:
                logger.error(
                    "CLOUD UPLOAD FAILED TO SCHEDULE (training continues). "
                    "local=%s gcs=%s error=%s",
                    local_path,
                    cloud_path,
                    exc,
                )
            return True
        return self._upload(local_path, cloud_path)

    def _upload(self, local_path: str, cloud_path: str) -> bool:
        if is_forbidden_cloud_upload_local_path(local_path):
            logger.warning(
                "Skipping cloud upload (replay/training state stays local-only): %s",
                local_path,
            )
            return True
        try:
            blob = self.bucket.blob(cloud_path)
            blob.upload_from_filename(local_path)
            return True
        except Exception as exc:
            logger.error(
                "CLOUD UPLOAD FAILED (training continues). local=%s gcs=%s error=%s",
                local_path,
                cloud_path,
                exc,
            )
            return False

    def upload_file_verified(self, local_path: str, cloud_path: str) -> bool:
        """Upload then verify by comparing MD5 hashes (synchronous; blocks caller).

        Prefer :meth:`upload_file` with ``background=True`` for HoF paths so
        training is not delayed on GCS or hash checks.
        """
        if self.provider == "local":
            return True
        if is_forbidden_cloud_upload_local_path(local_path):
            logger.warning(
                "Skipping cloud upload (replay/training state stays local-only): %s",
                local_path,
            )
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
                logger.error(
                    "CLOUD UPLOAD VERIFICATION FAILED (MD5 mismatch; training continues). "
                    "gcs=%s local_md5_b64=%s remote_md5_b64=%s",
                    cloud_path,
                    local_md5,
                    remote_md5,
                )
                return False
            return True
        except Exception as exc:
            logger.error(
                "CLOUD UPLOAD VERIFICATION FAILED (training continues). gcs=%s error=%s",
                cloud_path,
                exc,
            )
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
                logger.error(
                    "CLOUD DOWNLOAD FAILED (training continues). gcs=%s error=%s",
                    cloud_path,
                    exc,
                )
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
            logger.error(
                "CLOUD DELETE FAILED (training continues). gcs=%s error=%s",
                cloud_path,
                exc,
            )
            return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self, *, wait: bool = False) -> None:
        """Shut down the upload executor.

        By default does not wait for in-flight uploads so process exit and
        training teardown are never gated on GCS finishing.
        """
        self._executor.shutdown(wait=wait)
