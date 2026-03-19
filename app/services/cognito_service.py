import os
from jose import jwt
import boto3
import hmac
import hashlib
import base64
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import requests
from app.exceptions import ServiceException
from botocore.exceptions import ClientError

from dotenv import load_dotenv

load_dotenv()


CognitoUserRole = os.getenv("COGNITO_USER_ROLE", "Users")
CognitoAdminRole = os.getenv("COGNITO_ADMIN_ROLE", "Admins")
bearer_scheme = HTTPBearer(auto_error=False)

class CognitoService:
    def __init__(self):
        self.region = os.getenv("COGNITO_REGION")
        self.user_pool_id = os.getenv("COGNITO_USER_POOL_ID")
        self.client_id = os.getenv("COGNITO_CLIENT_ID")
        self.client_secret = os.getenv("COGNITO_CLIENT_SECRET")

        # JSON Web Key Set (JWKS) is a collection of public cryptographic keys used to verify JSON Web Tokens
        self.jwks_url = f"https://cognito-idp.{self.region}.amazonaws.com/{self.user_pool_id}/.well-known/jwks.json"
        self._jwks_keys = None
        self._client = None
        self.bearer = bearer_scheme

    @property
    def client(self):
        if self._client is None:
            # Initialize Boto3 Cognito client
            self._client = boto3.client("cognito-idp", region_name=self.region)
        return self._client

    @property
    def jwks_keys(self):
        if self._jwks_keys is None:
            self._jwks_keys = self._get_cognito_jwks()
        return self._jwks_keys

    def _get_cognito_jwks(self):
        """
        Retrieve JWKS (JSON Web Key Set) for token validation from AWS Cognito.
        """
        response = requests.get(self.jwks_url)
        if response.status_code != 200:
            raise ServiceException(status_code=500, detail="Unable to fetch JWKS for token validation.")
        return response.json()["keys"]

    def validate_token(self, auth: HTTPAuthorizationCredentials):
        """
        Validate and decode a JWT token issued by AWS Cognito.

        :param credentials: HTTPAuthorizationCredentials (token from the Authorization header).
        :return: The decoded token payload.
        """
        try:
            # Decode token using Cognito's JWKS
            token = auth.credentials
            headers = jwt.get_unverified_header(token)
            
            # Finding a specific JSON Web Key (JWK) from a JWKS using the "kid" (Key ID) parameter
            kid = headers.get("kid")
            key = next((k for k in self.jwks_keys if k["kid"] == kid), None)
            if not key:
                raise ServiceException(status_code=401, detail="Invalid token signature.")
            
            payload = jwt.decode(
                token,
                key=key,
                algorithms=["RS256"],
                audience=self.client_id,
                issuer=f"https://cognito-idp.{self.region}.amazonaws.com/{self.user_pool_id}",
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise ServiceException(status_code=401, detail="Token has expired.")
        except jwt.JWTError as e:
            raise ServiceException(status_code=401, detail=f"Token validation error: {str(e)}")
        
    def initiate_forgot_password(self, username: str) -> dict:
        """
        Starts Cognito forgot-password flow (sends OTP/code to email/SMS).
        """
        return self.client.forgot_password(
            ClientId=self.client_id,
            Username=username,
            SecretHash=self.calculate_secret_hash(username) 
        )
    def confirm_forgot_password(self, username: str, confirmation_code: str, new_password: str) -> dict:

        try:
            return self.client.confirm_forgot_password(
                ClientId=self.client_id,
                Username=username,
                ConfirmationCode=confirmation_code,
                Password=new_password,
                SecretHash=self.calculate_secret_hash(username),
            )
        except self.client.exceptions.CodeMismatchException:
            raise ServiceException(status_code=400, detail="Invalid reset code.")
        except self.client.exceptions.ExpiredCodeException:
            raise ServiceException(status_code=400, detail="Reset code has expired.")
        except self.client.exceptions.InvalidPasswordException as e:
            raise ServiceException(status_code=400, detail=f"Password does not meet policy: {str(e)}")
        except self.client.exceptions.UserNotFoundException:
            raise ServiceException(status_code=404, detail="User not found.")
        except ClientError as e:
            raise ServiceException(status_code=400, detail=str(e))
        except Exception as e:
            raise ServiceException(status_code=500, detail=f"Reset password failed: {str(e)}")
    
    def calculate_secret_hash(self, username) -> str:
        """
        Calculate the Cognito SECRET_HASH for the given username.
        """
        message = (username + self.client_id).encode("utf-8")
        key = self.client_secret.encode("utf-8")
        dig = hmac.new(key, message,hashlib.sha256).digest()
        return base64.b64encode(dig).decode("utf-8")

    def authenticate_user(self, username: str, password: str):
        """
        Authenticate a user with Cognito using their username and password.

        :param username: Username of the user.
        :param password: Password of the user.
        :return: Dictionary containing tokens if authentication is successful.
        """
        try:
            # Calculate the SECRET_HASH
            secret_hash = self.calculate_secret_hash(username)

            # Initiate the authentication
            response = self.client.initiate_auth(
                AuthFlow="USER_PASSWORD_AUTH",
                AuthParameters={
                    "USERNAME": username,
                    "PASSWORD": password,
                    "SECRET_HASH": secret_hash
                },
                ClientId=self.client_id
            )

            return {
                "id_token": response["AuthenticationResult"]["IdToken"],
                "access_token": response["AuthenticationResult"]["AccessToken"],
                "refresh_token": response["AuthenticationResult"]["RefreshToken"]
            }

        except self.client.exceptions.NotAuthorizedException:
            raise ServiceException(status_code=401, detail="Invalid username or password.")
        except self.client.exceptions.UserNotConfirmedException:
            raise ServiceException(status_code=403, detail="User account not confirmed.")
        except Exception as e:
            raise ServiceException(status_code=500, detail=f"Authentication failed: {str(e)}")
        
    def check_user_role(self, claims, required_role: str):
        """
        Check if the token contains the required role.
        """
        try:
            groups = claims.get("cognito:groups", [])
            if required_role in groups:
                return True
            raise ServiceException(status_code=403, detail="Insufficient permissions")
        except Exception as e:
            raise ServiceException(status_code=403, detail=f"Invalid token or permissions: {str(e)}")

    def register_user(self, username: str, email: str, password: str):
        """
        Register a new user with a distinct username, and store the user's email in Cognito.
        """
        try:
            # Calculate the SECRET_HASH if your app client has a client secret
            secret_hash = self.calculate_secret_hash(username)

            response = self.client.sign_up(
                ClientId=self.client_id,
                SecretHash=secret_hash,
                Username=username,      # <--- Distinct username
                Password=password,
                UserAttributes=[
                    {
                        'Name': 'email',
                        'Value': email       # <--- Storing user's email as an attribute
                    }
                ]
            )

            return response

        except self.client.exceptions.UsernameExistsException:
            raise ServiceException(status_code=400, detail="User already exists.")
        except Exception as e:
            raise ServiceException(status_code=500, detail=f"Registration failed: {str(e)}")
    
    
    def confirm_user(self, username: str, confirmation_code: str):
        """
        Confirm the user's signup with the code they received by email
        """
        try:
            # First confirm the sign-up
            self.client.confirm_sign_up(
                ClientId=self.client_id,
                Username=username,
                ConfirmationCode=confirmation_code,
                SecretHash=self.calculate_secret_hash(username)
            )

            return "User confirmed successfully."
        
        except self.client.exceptions.CodeMismatchException:
            raise ServiceException(status_code=400, detail="Invalid confirmation code.")
        except self.client.exceptions.ExpiredCodeException:
            raise ServiceException(status_code=400, detail="Confirmation code has expired.")
        except self.client.exceptions.UserNotFoundException:
            raise ServiceException(status_code=404, detail="User not found.")
        except Exception as e:
            raise ServiceException(status_code=500, detail=f"Confirmation failed: {str(e)}")

class RoleChecker:
    def __init__(self, allowed_role: str):
        self.allowed_role = allowed_role

    def __call__(self, auth: HTTPAuthorizationCredentials = Depends(bearer_scheme), 
                 cognito_service: CognitoService = Depends(CognitoService)):
        # Validate the token and check the user's role
        if not auth:
            raise ServiceException(status_code=401, detail="Not authenticated")
        claims = cognito_service.validate_token(auth)
        cognito_service.check_user_role(claims, self.allowed_role)
        return claims