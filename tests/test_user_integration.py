import pytest
from unittest.mock import MagicMock
from jose import jwt
from botocore.exceptions import ClientError
import requests

from app.services.cognito_service import CognitoService, RoleChecker
from app.exceptions import ServiceException


@pytest.fixture
def mock_cognito(mocker):
    """Fixture to provide a CognitoService with mocked boto3 and requests."""
    # 1. Properly mock requests.get so the __init__ validation passes
    mock_requests = mocker.patch("app.services.cognito_service.requests.get")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"keys": [{"kid": "123", "alg": "RS256"}]}
    mock_requests.return_value = mock_response

    # 2. Mock Boto3 client
    mock_boto = mocker.patch("app.services.cognito_service.boto3.client")
    mock_client = mock_boto.return_value

    # 3. CRITICAL FIX: Boto3 dynamically generates exception classes.
    # We MUST mock them as REAL Python Exception classes so `except` blocks don't crash.
    class MockExceptions:
        UsernameExistsException = type("UsernameExistsException", (Exception,), {})
        NotAuthorizedException = type("NotAuthorizedException", (Exception,), {})
        UserNotConfirmedException = type("UserNotConfirmedException", (Exception,), {})
        CodeMismatchException = type("CodeMismatchException", (Exception,), {})
        ExpiredCodeException = type("ExpiredCodeException", (Exception,), {})
        UserNotFoundException = type("UserNotFoundException", (Exception,), {})
        InvalidPasswordException = type("InvalidPasswordException", (Exception,), {})
        LimitExceededException = type("LimitExceededException", (Exception,), {})

    mock_client.exceptions = MockExceptions

    # 4. Instantiate the service
    service = CognitoService()
    service.client_id = "test_client_id"
    service.client_secret = "test_client_secret"
    # 👇 关键修复：去掉了 service.client = mock_client，因为属性被设置为了只读
    return service


# --- Tests for calculate_secret_hash ---
def test_calculate_secret_hash(mock_cognito):
    """Test HMAC calculation logic."""
    hash_val = mock_cognito.calculate_secret_hash("test_user")
    assert isinstance(hash_val, str)
    assert len(hash_val) > 0


# --- Tests for register_user ---
def test_register_user_success(mock_cognito):
    """Test successful user registration."""
    mock_cognito.client.sign_up.return_value = {"UserSub": "123", "UserConfirmed": False}
    result = mock_cognito.register_user("test@example.com", "test@example.com", "Pass123!")
    assert result["UserSub"] == "123"


def test_register_user_exists(mock_cognito):
    """Test registration when user already exists."""
    # Raise the true Mock Exception class we created
    mock_cognito.client.sign_up.side_effect = mock_cognito.client.exceptions.UsernameExistsException("Exists")
    with pytest.raises(ServiceException) as exc:
        mock_cognito.register_user("test@example.com", "test@example.com", "Pass123!")
    assert exc.value.status_code == 400


def test_register_user_other_error(mock_cognito):
    """Test registration with generic AWS error."""
    mock_cognito.client.sign_up.side_effect = Exception("General Error")
    with pytest.raises(ServiceException) as exc:
        mock_cognito.register_user("test@example.com", "test@example.com", "Pass")
    assert exc.value.status_code == 500


# --- Tests for confirm_user ---
def test_confirm_user_success(mock_cognito):
    """Test successful user confirmation."""
    mock_cognito.client.confirm_sign_up.return_value = {}
    result = mock_cognito.confirm_user("test@example.com", "123456")
    assert result == "User confirmed successfully."


def test_confirm_user_failure(mock_cognito):
    """Test user confirmation failure."""
    mock_cognito.client.confirm_sign_up.side_effect = mock_cognito.client.exceptions.CodeMismatchException("Mismatch")
    with pytest.raises(ServiceException) as exc:
        mock_cognito.confirm_user("test@example.com", "wrong_code")
    assert exc.value.status_code == 400


def test_confirm_user_expired_code(mock_cognito):
    mock_cognito.client.confirm_sign_up.side_effect = mock_cognito.client.exceptions.ExpiredCodeException("Expired")
    with pytest.raises(ServiceException) as exc:
        mock_cognito.confirm_user("test@example.com", "wrong_code")
    assert exc.value.status_code == 400


def test_confirm_user_not_found(mock_cognito):
    mock_cognito.client.confirm_sign_up.side_effect = mock_cognito.client.exceptions.UserNotFoundException("Missing")
    with pytest.raises(ServiceException) as exc:
        mock_cognito.confirm_user("missing@example.com", "wrong_code")
    assert exc.value.status_code == 404


def test_confirm_user_generic_error(mock_cognito):
    mock_cognito.client.confirm_sign_up.side_effect = Exception("unexpected")
    with pytest.raises(ServiceException) as exc:
        mock_cognito.confirm_user("test@example.com", "wrong_code")
    assert exc.value.status_code == 500


# --- Tests for authenticate_user ---
def test_authenticate_user_success(mock_cognito):
    """Test successful login."""
    mock_cognito.client.initiate_auth.return_value = {
        "AuthenticationResult": {
            "AccessToken": "access",
            "IdToken": "id",
            "RefreshToken": "refresh"
        }
    }
    result = mock_cognito.authenticate_user("test@example.com", "Pass123!")
    assert result["access_token"] == "access"


def test_authenticate_user_not_authorized(mock_cognito):
    """Test login with wrong password."""
    mock_cognito.client.initiate_auth.side_effect = mock_cognito.client.exceptions.NotAuthorizedException("Not Auth")
    with pytest.raises(ServiceException) as exc:
        mock_cognito.authenticate_user("test@example.com", "WrongPass!")
    assert exc.value.status_code == 401


def test_authenticate_user_not_confirmed(mock_cognito):
    """Test login when user is not confirmed."""
    mock_cognito.client.initiate_auth.side_effect = mock_cognito.client.exceptions.UserNotConfirmedException(
        "Unconfirmed")
    with pytest.raises(ServiceException) as exc:
        mock_cognito.authenticate_user("test@example.com", "Pass123!")
    assert exc.value.status_code == 403


def test_authenticate_user_generic_error(mock_cognito):
    mock_cognito.client.initiate_auth.side_effect = Exception("unexpected")
    with pytest.raises(ServiceException) as exc:
        mock_cognito.authenticate_user("test@example.com", "Pass123!")
    assert exc.value.status_code == 500


# --- Tests for forgot_password ---
def test_initiate_forgot_password_success(mock_cognito):
    """Test requesting a password reset code."""
    mock_cognito.client.forgot_password.return_value = {}
    result = mock_cognito.initiate_forgot_password("test@example.com")
    assert result == {}


# --- Tests for confirm_forgot_password ---
def test_confirm_forgot_password_success(mock_cognito):
    """Test resetting password with code."""
    mock_cognito.client.confirm_forgot_password.return_value = {}
    result = mock_cognito.confirm_forgot_password("test@example.com", "123456", "NewPass123!")
    assert result == {}


def test_confirm_forgot_password_failure(mock_cognito):
    """Test resetting password failure."""
    mock_cognito.client.confirm_forgot_password.side_effect = mock_cognito.client.exceptions.CodeMismatchException(
        "Bad Code")
    with pytest.raises(ServiceException) as exc:
        mock_cognito.confirm_forgot_password("test@example.com", "000000", "NewPass123!")
    assert exc.value.status_code == 400


def test_confirm_forgot_password_expired_code(mock_cognito):
    mock_cognito.client.confirm_forgot_password.side_effect = mock_cognito.client.exceptions.ExpiredCodeException(
        "Expired"
    )
    with pytest.raises(ServiceException) as exc:
        mock_cognito.confirm_forgot_password("test@example.com", "000000", "NewPass123!")
    assert exc.value.status_code == 400


def test_confirm_forgot_password_invalid_password(mock_cognito):
    mock_cognito.client.confirm_forgot_password.side_effect = mock_cognito.client.exceptions.InvalidPasswordException(
        "Policy"
    )
    with pytest.raises(ServiceException) as exc:
        mock_cognito.confirm_forgot_password("test@example.com", "000000", "bad")
    assert exc.value.status_code == 400


def test_confirm_forgot_password_user_not_found(mock_cognito):
    mock_cognito.client.confirm_forgot_password.side_effect = mock_cognito.client.exceptions.UserNotFoundException(
        "Missing"
    )
    with pytest.raises(ServiceException) as exc:
        mock_cognito.confirm_forgot_password("missing@example.com", "000000", "NewPass123!")
    assert exc.value.status_code == 404


def test_confirm_forgot_password_client_error(mock_cognito):
    mock_cognito.client.confirm_forgot_password.side_effect = ClientError(
        {"Error": {"Code": "InvalidParameterException", "Message": "Bad request"}},
        "ConfirmForgotPassword",
    )
    with pytest.raises(ServiceException) as exc:
        mock_cognito.confirm_forgot_password("test@example.com", "000000", "NewPass123!")
    assert exc.value.status_code == 400


def test_confirm_forgot_password_generic_exception(mock_cognito):
    mock_cognito.client.confirm_forgot_password.side_effect = Exception("boom")
    with pytest.raises(ServiceException) as exc:
        mock_cognito.confirm_forgot_password("test@example.com", "000000", "NewPass123!")
    assert exc.value.status_code == 500


# --- Tests for validate_token ---
class DummyAuth:
    """Helper class to mock HTTPAuthorizationCredentials."""
    credentials = "fake.jwt.token"


def test_validate_token_success(mock_cognito, mocker):
    mocker.patch("app.services.cognito_service.jwt.get_unverified_header", return_value={"kid": "123"})
    mocker.patch("app.services.cognito_service.jwt.decode", return_value={"sub": "user-uuid"})
    result = mock_cognito.validate_token(DummyAuth())
    assert result["sub"] == "user-uuid"


def test_validate_token_invalid_kid(mock_cognito, mocker):
    mocker.patch("app.services.cognito_service.jwt.get_unverified_header", return_value={"kid": "wrong-kid"})
    with pytest.raises(ServiceException) as exc:
        mock_cognito.validate_token(DummyAuth())
    assert exc.value.status_code == 401


def test_validate_token_expired(mock_cognito, mocker):
    mocker.patch("app.services.cognito_service.jwt.get_unverified_header", return_value={"kid": "123"})
    mocker.patch("app.services.cognito_service.jwt.decode", side_effect=jwt.ExpiredSignatureError)
    with pytest.raises(ServiceException) as exc:
        mock_cognito.validate_token(DummyAuth())
    assert exc.value.status_code == 401


def test_validate_token_jwt_error(mock_cognito, mocker):
    mocker.patch("app.services.cognito_service.jwt.get_unverified_header", return_value={"kid": "123"})
    mocker.patch("app.services.cognito_service.jwt.decode", side_effect=jwt.JWTError("bad token"))
    with pytest.raises(ServiceException) as exc:
        mock_cognito.validate_token(DummyAuth())
    assert exc.value.status_code == 401


def test_jwks_keys_uses_cached_value_after_first_fetch(mock_cognito, mocker):
    mocked_fetch = mocker.patch.object(
        mock_cognito,
        "_get_cognito_jwks",
        return_value=[{"kid": "123", "alg": "RS256"}],
    )

    first = mock_cognito.jwks_keys
    second = mock_cognito.jwks_keys

    assert first == [{"kid": "123", "alg": "RS256"}]
    assert second == first
    assert mocked_fetch.call_count == 1


def test_get_cognito_jwks_non_200_raises_service_exception(mock_cognito, mocker):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mocker.patch("app.services.cognito_service.requests.get", return_value=mock_response)

    with pytest.raises(ServiceException) as exc:
        mock_cognito._get_cognito_jwks()

    assert exc.value.status_code == 500
    assert "Unable to fetch JWKS" in exc.value.detail


def test_get_cognito_jwks_request_exception_returns_empty_list(mock_cognito, mocker):
    mocker.patch("app.services.cognito_service.requests.get", side_effect=requests.RequestException("network"))
    assert mock_cognito._get_cognito_jwks() == []


# --- Tests for check_user_role & RoleChecker ---
def test_check_user_role_success(mock_cognito):
    claims = {"cognito:groups": ["Users", "Admins"]}
    assert mock_cognito.check_user_role(claims, "Admins") is True


def test_check_user_role_failure(mock_cognito):
    claims = {"cognito:groups": ["Users"]}
    with pytest.raises(ServiceException) as exc:
        mock_cognito.check_user_role(claims, "Admins")
    assert exc.value.status_code == 403


def test_role_checker_call(mock_cognito, mocker):
    checker = RoleChecker("Users")
    mocker.patch.object(mock_cognito, "validate_token", return_value={"cognito:groups": ["Users"]})
    mocker.patch.object(mock_cognito, "check_user_role", return_value=True)

    result = checker(auth=DummyAuth(), cognito_service=mock_cognito)
    assert "cognito:groups" in result


def test_role_checker_without_auth_raises_403(mock_cognito):
    checker = RoleChecker("Users")

    with pytest.raises(ServiceException) as exc:
        checker(auth=None, cognito_service=mock_cognito)

    assert exc.value.status_code == 403