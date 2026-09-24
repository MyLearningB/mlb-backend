import os
from datetime import datetime, timedelta, timezone

from passlib.context import CryptContext
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session, select

from database import get_session
from models import User


# ============================================================
# 1. CONFIGURATION & SECRETS
# ============================================================

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError(
        "FATAL: SECRET_KEY environment variable is missing. "
        "Application cannot start securely."
    )

ALGORITHM = "HS256"

# Access tokens are short-lived on purpose. Combined with
# token_version bumping on logout / password change, a stolen
# access token has a small window of usefulness.
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7


# ============================================================
# 2. PASSWORD HASHING
# ============================================================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


# ============================================================
# 3. JWT TOKEN GENERATION
# ============================================================
#
# Every token carries:
#   sub      — the user's email
#   version  — the user's token_version at issue time; used
#              to revoke sessions server-side
#   type     — "access" | "refresh" | "password_reset"
#   exp      — expiry
#
# The "type" claim is what allows each verifier to reject
# tokens that aren't meant for it. Without it, an access token
# could be accepted by the refresh endpoint, and a refresh
# token could be accepted as an access token.
# ============================================================


def create_access_token(data: dict, version: int) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )
    to_encode.update({
        "exp": expire,
        "version": version,
        "type": "access",
    })
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict, version: int) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(
        days=REFRESH_TOKEN_EXPIRE_DAYS
    )
    to_encode.update({
        "exp": expire,
        "version": version,
        "type": "refresh",
    })
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_password_reset_token(email: str, version: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=1)
    to_encode = {
        "sub": email,
        "type": "password_reset",
        "version": version,
        "exp": expire,
    }
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# ============================================================
# 4. PROTECTED ROUTE DEPENDENCY
# ============================================================

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        # ----------------------------------------------------
        # TOKEN TYPE CHECK
        # ----------------------------------------------------
        #
        # Protected endpoints must only accept access tokens.
        # A refresh token (7-day lifetime) presented here would
        # otherwise be honoured as a valid credential and let a
        # stolen refresh token authenticate API calls directly.
        #
        # Tokens issued before this change have no "type" claim,
        # so we only reject when the claim is explicitly present
        # and wrong. Once all live tokens carry the claim, tighten
        # to require `token_type == "access"` unconditionally.
        #
        token_type = payload.get("type")
        if token_type is not None and token_type != "access":
            raise credentials_exception

        email: str = payload.get("sub")
        token_version: int = payload.get("version")

        if email is None or token_version is None:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    statement = select(User).where(User.email == email)
    user = session.exec(statement).first()

    if user is None:
        raise credentials_exception

    # --------------------------------------------------------
    # TRUE LOGOUT ENFORCEMENT
    # --------------------------------------------------------
    #
    # If the versions don't match, the token was revoked by a
    # logout, password change, or password reset.
    #
    if user.token_version != token_version:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user