"""Unit tests for services/s3_client.py (AIS-3634)."""
import json
from unittest.mock import MagicMock, patch

import services.s3_client as s3_client
from services.s3_client import download_json


def setup_function(_fn):
    # The module-level client is process-wide singleton state (code review
    # must-fix #6/B3) — reset it before every test so each test observes its
    # own patched boto3.client call instead of a previous test's cached client.
    s3_client._client = None


def _fake_client():
    fake_client = MagicMock()
    fake_client.get_object.return_value = {"Body": MagicMock(read=lambda: json.dumps({"docs": []}).encode())}
    return fake_client


def test_download_json_parses_body():
    with patch("boto3.client", return_value=_fake_client()) as boto_client:
        result = download_json("my-bucket", "my/key.json")
    boto_client.assert_called_once()
    assert result == {"docs": []}


def test_download_json_passes_bucket_and_key_to_get_object():
    fake_client = _fake_client()
    with patch("boto3.client", return_value=fake_client):
        download_json("my-bucket", "my/key.json")
    fake_client.get_object.assert_called_once_with(Bucket="my-bucket", Key="my/key.json")


def test_download_json_reuses_the_same_client_across_calls():
    """Code review must-fix #6 (B3): the client must be built once and reused,
    not re-constructed (fresh TCP/TLS handshake) on every single call."""
    with patch("boto3.client", return_value=_fake_client()) as boto_client:
        download_json("bucket-1", "key-1.json")
        download_json("bucket-2", "key-2.json")
    boto_client.assert_called_once()


def test_download_json_passes_explicit_timeouts_to_boto_config():
    with patch("boto3.client", return_value=_fake_client()) as boto_client:
        download_json("my-bucket", "my/key.json")
    _, kwargs = boto_client.call_args
    assert kwargs["config"].connect_timeout is not None
    assert kwargs["config"].read_timeout is not None
