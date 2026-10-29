"""_build_active_session: pick the credential source that actually authenticates.

The bug this guards against: with AWS_PROFILE set, boto3.Session(profile_name=...)
makes botocore drop the env-var provider (its disable_env_vars path), so static
AWS_ACCESS_KEY_ID/SECRET keys are silently ignored and an expired SSO profile is
used instead. We must prefer usable static env creds (via the DEFAULT session)
and only fall back to the profile.
"""
from unittest.mock import MagicMock, patch

import connectors.secrets_manager as sm


def _session(active: bool) -> MagicMock:
    s = MagicMock(name="session")
    s.get_credentials.return_value = MagicMock()  # resolvable
    if active:
        s.client.return_value.get_caller_identity.return_value = {"Account": "123"}
    else:
        s.client.return_value.get_caller_identity.side_effect = Exception("ExpiredTokenException")
    return s


def _capture_sessions(active_for):
    """Return (side_effect, calls) where a session is 'active' iff active_for(kwargs)."""
    calls = []

    def _factory(**kwargs):
        calls.append(kwargs)
        return _session(active_for(kwargs))

    return _factory, calls


def test_prefers_static_env_creds_and_does_not_pass_profile(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sec")
    monkeypatch.setenv("AWS_PROFILE", "dev")
    # static (default session, no profile_name) authenticates
    factory, calls = _capture_sessions(lambda kw: "profile_name" not in kw)
    with patch.object(sm.boto3, "Session", side_effect=factory):
        sm._build_active_session("us-west-2", "dev")
    # The FIRST candidate tried must be the default session (no profile_name) —
    # that's the only way env creds survive botocore's disable_env_vars path.
    assert "profile_name" not in calls[0]


def test_falls_back_to_profile_when_static_creds_are_expired(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sec")
    monkeypatch.setenv("AWS_PROFILE", "dev")
    # static session is INACTIVE (expired); only the profile session authenticates
    factory, calls = _capture_sessions(lambda kw: "profile_name" in kw)
    with patch.object(sm.boto3, "Session", side_effect=factory):
        sm._build_active_session("us-west-2", "dev")
    assert any("profile_name" in kw for kw in calls)


def test_uses_profile_when_no_static_creds(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setenv("AWS_PROFILE", "dev")
    factory, calls = _capture_sessions(lambda kw: True)
    with patch.object(sm.boto3, "Session", side_effect=factory):
        sm._build_active_session("us-west-2", "dev")
    # No static creds → the only candidate built is the profile session.
    assert calls == [{"profile_name": "dev", "region_name": "us-west-2"}]


def test_returns_first_candidate_when_nothing_authenticates(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sec")
    monkeypatch.setenv("AWS_PROFILE", "dev")
    factory, calls = _capture_sessions(lambda kw: False)  # everything expired
    with patch.object(sm.boto3, "Session", side_effect=factory):
        session = sm._build_active_session("us-west-2", "dev")
    # Doesn't raise here — returns the first candidate so its own error surfaces
    # later (downstream get_secret call), not a vaguer one.
    assert session is not None
