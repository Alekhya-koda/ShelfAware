import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Internal project imports
from app.main import app
from app.db.database import Base, get_db
from app.models.user import User
from app.models.user_profile import UserProfile
from app.dependencies.auth import get_current_db_user

# --- Fixtures ---
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def db_session():
    """Create a fresh database for each test case."""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(db_session):
    """Create a test client with a database override."""

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ==========================================
# TEST ROUTES: /auth
# ==========================================

def test_auth_registration(client, mocker):
    """Test successful registration."""
    mocker.patch("app.routes.auth.cognito_service.register_user",
                 return_value={"UserSub": "sub", "UserConfirmed": False})
    payload = {"username": "test_user", "email": "test@test.com", "password": "Password123!"}
    response = client.post("/auth/registration", json=payload)
    assert response.status_code == 201


def test_auth_confirm(client, mocker):
    """Test account confirmation."""
    mocker.patch("app.routes.auth.cognito_service.confirm_user", return_value=True)
    payload = {"email": "test@test.com", "confirmation_code": "123456"}
    response = client.post("/auth/confirm", json=payload)
    assert response.status_code == 200


def test_auth_login_success(client, mocker, db_session):
    """Test successful login."""
    # Setup db user
    test_user = User(email="login@test.com", cognito_sub="sub-login")
    db_session.add(test_user)
    db_session.commit()

    mocker.patch("app.routes.auth.cognito_service.authenticate_user",
                 return_value={"access_token": "token1", "id_token": "token2", "refresh_token": "token3"})
    payload = {"email": "login@test.com", "password": "Password123!"}
    response = client.post("/auth/login", json=payload)

    assert response.status_code == 200
    # FIXED: Access token is inside the "tokens" dictionary
    assert "access_token" in response.json()["tokens"]


def test_auth_login_user_not_in_db(client, mocker):
    """Test login when user is in Cognito but missing in local DB."""
    mocker.patch("app.routes.auth.cognito_service.authenticate_user", return_value={"access_token": "token1"})
    payload = {"email": "ghost@test.com", "password": "Password123!"}
    response = client.post("/auth/login", json=payload)
    assert response.status_code == 404


def test_auth_forgot_password(client, mocker):
    """Test forgot password request."""
    # FIXED: Changed from forgot_password to initiate_forgot_password
    mocker.patch("app.routes.auth.cognito_service.initiate_forgot_password", return_value=True)
    payload = {"email": "test@test.com"}
    response = client.post("/auth/forgot-password", json=payload)
    assert response.status_code == 200


def test_auth_reset_password(client, mocker):
    """Test resetting password."""
    mocker.patch("app.routes.auth.cognito_service.confirm_forgot_password", return_value=True)
    # FIXED: The route expects 'token', not 'confirmation_code' based on the schema usage
    payload = {"email": "test@test.com", "token": "123456", "new_password": "NewPassword123!"}
    response = client.post("/auth/reset-password", json=payload)
    assert response.status_code == 200


# ==========================================
# TEST ROUTES: /user-profile
# ==========================================

def test_get_my_profile_auto_create(client, db_session):
    """Test GET profile auto-creates missing profile."""
    test_user = User(email="new_profile@example.com", cognito_sub="sub-new")
    db_session.add(test_user)
    db_session.commit()

    app.dependency_overrides[get_current_db_user] = lambda: test_user
    response = client.get("/user-profile/me")
    assert response.status_code == 200
    assert response.json()["display_name"] == "new_profile"


def test_patch_my_profile(client, db_session):
    """Test updating existing profile."""
    test_user = User(email="update@example.com", cognito_sub="sub-update")
    db_session.add(test_user)
    db_session.commit()

    profile = UserProfile(user_id=test_user.user_id, display_name="OldName")
    db_session.add(profile)
    db_session.commit()

    app.dependency_overrides[get_current_db_user] = lambda: test_user

    payload = {"display_name": "NewName", "bio": "Updated Bio"}
    response = client.patch("/user-profile/me", json=payload)
    assert response.status_code == 200
    assert response.json()["display_name"] == "NewName"
    assert response.json()["bio"] == "Updated Bio"


def test_patch_my_profile_auto_create(client, db_session):
    """Test updating profile when profile doesn't exist yet (auto-creates)."""
    # FIXED: The endpoint auto-creates the profile if it's missing, so it returns 200, not 404.
    test_user = User(email="noprofile@example.com", cognito_sub="sub-no")
    db_session.add(test_user)
    db_session.commit()

    app.dependency_overrides[get_current_db_user] = lambda: test_user

    payload = {"bio": "Trying to update"}
    response = client.patch("/user-profile/me", json=payload)

    assert response.status_code == 200
    assert response.json()["bio"] == "Trying to update"
    assert response.json()["display_name"] == "noprofile"


def test_public_profile_found(client, db_session):
    """Test viewing a public profile."""
    test_user = User(email="public@example.com", cognito_sub="sub-public")
    db_session.add(test_user)
    db_session.commit()

    profile = UserProfile(user_id=test_user.user_id, display_name="PublicStar", bio="Famous")
    db_session.add(profile)
    db_session.commit()

    response = client.get("/user-profile/public/PublicStar")
    assert response.status_code == 200
    assert response.json()["bio"] == "Famous"


def test_public_profile_not_found(client):
    """Test viewing a non-existent public profile."""
    response = client.get("/user-profile/public/GhostUser")
    assert response.status_code == 404