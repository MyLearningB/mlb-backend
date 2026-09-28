import os
import secrets

import resend

from fastapi import APIRouter, HTTPException, status, Depends, Request
from pydantic import BaseModel
from jose import jwt, JWTError
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlmodel import Session, select

from database import get_session
from models import User

from security import (
    get_password_hash,
    verify_password,
    create_access_token,
    create_refresh_token,
    get_current_user,
    create_password_reset_token,
    SECRET_KEY,
    ALGORITHM,
)

from schemas import (
    UserRegister,
    UserLogin,
    TokenResponse,
    TokenRefreshRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
)

from google.oauth2 import id_token
from google.auth.transport import requests as google_requests


# ============================================================
# GOOGLE OAUTH CONFIG
# ============================================================

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

if not GOOGLE_CLIENT_ID:
    print("WARNING: GOOGLE_CLIENT_ID is not configured.")


class GoogleLoginRequest(BaseModel):
    id_token: str


# ============================================================
# RATE LIMITER
# ============================================================
#
# Render / Fly / etc. put the client IP in X-Forwarded-For and
# set request.client.host to the proxy address. Keying on the
# proxy IP means every user shares one bucket — one abusive
# client locks everyone out.

def _rate_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # X-Forwarded-For can be "client, proxy1, proxy2" — take the first.
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(
    key_func=_rate_key
)


# ============================================================
# EMAIL
# ============================================================

resend.api_key = os.getenv(
    "RESEND_API_KEY"
)

FROM_EMAIL = os.getenv(
    "RESEND_FROM_EMAIL",
    "onboarding@resend.dev",
)


def send_email(
    to_email: str,
    subject: str,
    html_content: str,
):
    try:
        resend.Emails.send(
            {
                "from": f"myLB <{FROM_EMAIL}>",
                "to": [to_email],
                "subject": subject,
                "html": html_content,
            }
        )

        return True

    except Exception as e:
        print(
            f"❌ Failed to send email to "
            f"{to_email}: {e}"
        )

        return False


# ============================================================
# REQUEST MODELS
# ============================================================

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


# Logout is identified by the access token (via get_current_user),
# and the resulting token_version bump revokes BOTH the access and
# refresh tokens. The refresh_token field was unused, so it's gone.
# Old clients that still send {"refresh_token": "..."} are fine —
# FastAPI silently ignores unknown body fields.
class LogoutRequest(BaseModel):
    pass


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


# ============================================================
# REGISTER
# ============================================================

@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
)
@limiter.limit("10/minute")
def register_user(
    request: Request,
    user_data: UserRegister,
    session: Session = Depends(get_session),
):
    """
    Create a new account.

    Newly registered users have not completed onboarding.

    Therefore:
        is_first_session = True
    """

    email = (
        user_data.email
        .strip()
        .lower()
    )

    # --------------------------------------------------------
    # CHECK EXISTING USER
    # --------------------------------------------------------

    existing_user = session.exec(
        select(User).where(
            User.email == email
        )
    ).first()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "An account with this "
                "email already exists."
            ),
        )

    # --------------------------------------------------------
    # HASH PASSWORD
    # --------------------------------------------------------

    hashed_password = get_password_hash(
        user_data.password
    )

    # --------------------------------------------------------
    # CREATE USER
    # --------------------------------------------------------

    new_user = User(
        name=user_data.name.strip(),
        email=email,
        hashed_password=hashed_password,
        is_verified=True,
        is_first_session=True,
        token_version=1,
    )

    session.add(new_user)

    try:
        session.commit()

    except Exception:
        session.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Could not create your account. "
                "Please try again."
            ),
        )

    session.refresh(new_user)

    # --------------------------------------------------------
    # CREATE TOKENS
    # --------------------------------------------------------

    access_token = create_access_token(
        data={
            "sub": new_user.email
        },
        version=new_user.token_version,
    )

    refresh_token = create_refresh_token(
        data={
            "sub": new_user.email
        },
        version=new_user.token_version,
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",

        "user": {
            "id": new_user.id,
            "email": new_user.email,
            "name": new_user.name,
            "is_first_session": (
                new_user.is_first_session
            ),
        },
    }


# ============================================================
# LOGIN
# ============================================================

@router.post(
    "/login",
    response_model=TokenResponse,
)
@limiter.limit("5/minute")
def login_user(
    request: Request,
    credentials: UserLogin,
    session: Session = Depends(get_session),
):
    """
    Authenticate an existing user.

    Backend is the source of truth for onboarding state.
    """

    email = (
        credentials.email
        .strip()
        .lower()
    )

    # --------------------------------------------------------
    # FIND USER
    # --------------------------------------------------------

    db_user = session.exec(
        select(User).where(
            User.email == email
        )
    ).first()

    # --------------------------------------------------------
    # VERIFY CREDENTIALS
    # --------------------------------------------------------

    if (
        not db_user
        or not verify_password(
            credentials.password,
            db_user.hashed_password,
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Incorrect email or password."
            ),
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    # --------------------------------------------------------
    # CREATE TOKENS
    # --------------------------------------------------------

    access_token = create_access_token(
        data={
            "sub": db_user.email
        },
        version=db_user.token_version,
    )

    refresh_token = create_refresh_token(
        data={
            "sub": db_user.email
        },
        version=db_user.token_version,
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",

        "user": {
            "id": db_user.id,
            "email": db_user.email,
            "name": db_user.name,
            "is_first_session": (
                db_user.is_first_session
            ),
        },
    }


# ============================================================
# REFRESH ACCESS TOKEN
# ============================================================
#
# NOTE: We deliberately do NOT bump token_version here.
#
# Bumping on every refresh made only one device able to stay
# logged in — the second device's next refresh would 401 and
# get force-logged-out. It also caused parallel refreshes
# (three 401s at once from the dashboard) to invalidate each
# other.
#
# Revocation still works: logout, change-password, and
# reset-password all bump token_version, which invalidates
# every existing access AND refresh token for that user.
#
# Trade-off: refresh tokens are no longer strictly one-time
# use. If you later want rotation with replay protection, add
# a `previous_token_version` column and accept N-1 for a
# short grace window.

@router.post(
    "/token/refresh",
    response_model=TokenResponse,
)
@limiter.limit("20/minute")
def refresh_access_token(
    request: Request,
    body: TokenRefreshRequest,
    session: Session = Depends(get_session),
):
    try:
        payload = jwt.decode(
            body.refresh_token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
        )

        # Strict: only refresh tokens allowed here.
        if payload.get("type") != "refresh":
            raise JWTError

        email = payload.get("sub")
        token_version = payload.get("version")

        if email is None or token_version is None:
            raise JWTError

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token.",
        )

    db_user = session.exec(
        select(User).where(User.email == email)
    ).first()

    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
        )

    if db_user.token_version != token_version:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired. Please log in again.",
        )

    # ========================================================
    # ISSUE NEW TOKENS WITH THE SAME VERSION
    # ========================================================

    access_token = create_access_token(
        data={"sub": db_user.email},
        version=db_user.token_version,
    )

    refresh_token = create_refresh_token(
        data={"sub": db_user.email},
        version=db_user.token_version,
    )

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": {
            "id": db_user.id,
            "email": db_user.email,
            "name": db_user.name,
            "is_first_session": db_user.is_first_session,
        },
    }


# ============================================================
# FORGOT PASSWORD
# ============================================================

@router.post(
    "/forgot-password",
    status_code=status.HTTP_200_OK,
)
@limiter.limit("3/minute")
def forgot_password(
    request: Request,
    body: ForgotPasswordRequest,
    session: Session = Depends(get_session),
):
    email = (
        body.email
        .strip()
        .lower()
    )

    db_user = session.exec(
        select(User).where(
            User.email == email
        )
    ).first()

    # --------------------------------------------------------
    # PREVENT EMAIL ENUMERATION
    # --------------------------------------------------------

    if db_user:
        reset_token = create_password_reset_token(
            db_user.email,
            db_user.token_version,
        )

        reset_link = (
            "mylb://reset-password"
            f"?token={reset_token}"
        )

        first_name = (
            db_user.name.split()[0]
            if db_user.name
            else "there"
        )

        html_content = f"""
        <div
            style="
                font-family: Arial, sans-serif;
                max-width: 600px;
                margin: auto;
                padding: 20px;
                text-align: center;
            "
        >
            <h2 style="color: #2196F3;">
                Password Reset Request
            </h2>

            <p>
                Hi {first_name},
            </p>

            <p>
                Click below to reset your password:
            </p>

            <a
                href="{reset_link}"
                style="
                    display: inline-block;
                    padding: 12px 24px;
                    margin: 20px 0;
                    background-color: #2196F3;
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                    font-weight: bold;
                "
            >
                Reset Password
            </a>
        </div>
        """

        send_email(
            db_user.email,
            "Reset your myLB Password",
            html_content,
        )

    return {
        "message": (
            "If an account with that email exists, "
            "you'll receive a reset link."
        )
    }


# ============================================================
# RESET PASSWORD
# ============================================================

@router.post(
    "/reset-password",
    status_code=status.HTTP_200_OK,
)
@limiter.limit("5/minute")
def reset_password(
    request: Request,
    body: ResetPasswordRequest,
    session: Session = Depends(get_session),
):
    try:
        payload = jwt.decode(
            body.token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
        )

        # Strict: only password-reset tokens.
        if payload.get("type") != "password_reset":
            raise JWTError

        email = payload.get("sub")
        token_version = payload.get("version")

        db_user = session.exec(
            select(User).where(
                User.email == email
            )
        ).first()

        if (
            not db_user
            or db_user.token_version
            != token_version
        ):
            raise JWTError

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired token.",
        )

    # --------------------------------------------------------
    # CHANGE PASSWORD
    # --------------------------------------------------------

    db_user.hashed_password = get_password_hash(
        body.password
    )

    # --------------------------------------------------------
    # REVOKE ALL EXISTING SESSIONS
    # --------------------------------------------------------

    db_user.token_version += 1

    session.add(db_user)

    try:
        session.commit()

    except Exception:
        session.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Could not update password. "
                "Please try again."
            ),
        )

    return {
        "message": "Password updated successfully."
    }


# ============================================================
# CHANGE PASSWORD
# ============================================================

@router.post(
    "/change-password",
    status_code=status.HTTP_200_OK,
)
def change_password(
    request: ChangePasswordRequest,
    current_user: User = Depends(
        get_current_user
    ),
    session: Session = Depends(
        get_session
    ),
):
    # --------------------------------------------------------
    # VERIFY CURRENT PASSWORD
    # --------------------------------------------------------

    if not verify_password(
        request.current_password,
        current_user.hashed_password,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect current password",
        )

    # --------------------------------------------------------
    # UPDATE PASSWORD
    # --------------------------------------------------------

    current_user.hashed_password = get_password_hash(
        request.new_password
    )

    # --------------------------------------------------------
    # REVOKE OTHER SESSIONS
    # --------------------------------------------------------

    current_user.token_version += 1

    session.add(current_user)

    try:
        session.commit()

    except Exception:
        session.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Could not update password. "
                "Please try again."
            ),
        )

    return {
        "success": True,
        "message": (
            "Password updated successfully. "
            "You will be logged out of other devices."
        ),
    }


# ============================================================
# LOGOUT
# ============================================================

@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
)
def logout(
    current_user: User = Depends(
        get_current_user
    ),
    session: Session = Depends(
        get_session
    ),
):
    """
    Secure server-side logout.

    Incrementing token_version invalidates existing access
    AND refresh tokens for this user.

    The client is expected to call this before clearing its
    local secure storage.
    """

    current_user.token_version += 1

    session.add(current_user)

    try:
        session.commit()

    except Exception:
        session.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Could not log out securely. "
                "Please try again."
            ),
        )

    return {
        "logged_out": True,
        "message": "Successfully logged out securely.",
    }


# ============================================================
# GOOGLE LOGIN
# ============================================================

@router.post(
    "/google",
    response_model=TokenResponse,
)
@limiter.limit("5/minute")
def google_login(
    request: Request,
    body: GoogleLoginRequest,
    session: Session = Depends(get_session),
):
    """
    Authenticate a user with a Google ID token.

    Flutter sends:

        {
            "id_token": "GOOGLE_ID_TOKEN"
        }

    Google verifies the token and returns the user's identity.

    The backend then:
        1. Finds the user by email.
        2. Creates the user if they don't exist.
        3. Creates this application's JWT access token.
        4. Creates this application's refresh token.
        5. Returns onboarding state.
    """

    # ========================================================
    # CHECK GOOGLE CONFIG
    # ========================================================

    if not GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Google authentication is not configured "
                "on the server."
            ),
        )

    # ========================================================
    # VERIFY GOOGLE ID TOKEN
    # ========================================================

    try:
        id_info = id_token.verify_oauth2_token(
            body.id_token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )

    except ValueError as e:
        print(
            f"❌ GOOGLE TOKEN VERIFICATION FAILED: {e}"
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Invalid or expired Google authentication token."
            ),
        )

    except Exception as e:
        print(
            f"❌ GOOGLE TOKEN VERIFICATION ERROR: {e}"
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Could not verify Google authentication."
            ),
        )

    # ========================================================
    # GET GOOGLE USER INFORMATION
    # ========================================================

    email = id_info.get("email")
    name = id_info.get("name")

    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Google did not provide an email address."
            ),
        )

    email = email.strip().lower()

    if not name:
        name = "Google User"

    name = name.strip()

    # ========================================================
    # OPTIONAL: CHECK VERIFIED EMAIL
    # ========================================================

    email_verified = id_info.get(
        "email_verified",
        False,
    )

    if not email_verified:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Your Google email address is not verified."
            ),
        )

    # ========================================================
    # FIND EXISTING USER
    # ========================================================

    db_user = session.exec(
        select(User).where(
            User.email == email
        )
    ).first()

    # ========================================================
    # CREATE USER IF IT DOESN'T EXIST
    # ========================================================

    if not db_user:

        random_password = secrets.token_urlsafe(32)

        hashed_password = get_password_hash(
            random_password
        )

        db_user = User(
            name=name,
            email=email,
            hashed_password=hashed_password,

            # Google already verified the identity.
            is_verified=True,

            # New Google users still need onboarding.
            is_first_session=True,

            token_version=1,
        )

        session.add(db_user)

        try:
            session.commit()
            session.refresh(db_user)

        except Exception as e:
            session.rollback()

            print(
                f"❌ GOOGLE USER CREATION ERROR: {e}"
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "Could not create your account. "
                    "Please try again."
                ),
            )

    else:

        # ====================================================
        # EXISTING USER
        # ====================================================

        # Ensure Google authentication marks the account
        # as verified.
        if not db_user.is_verified:
            db_user.is_verified = True

            session.add(db_user)

            try:
                session.commit()
                session.refresh(db_user)

            except Exception as e:
                session.rollback()

                print(
                    f"❌ GOOGLE USER UPDATE ERROR: {e}"
                )

                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=(
                        "Could not update your account. "
                        "Please try again."
                    ),
                )

    # ========================================================
    # CREATE APP ACCESS TOKEN
    # ========================================================

    access_token = create_access_token(
        data={
            "sub": db_user.email
        },
        version=db_user.token_version,
    )

    # ========================================================
    # CREATE APP REFRESH TOKEN
    # ========================================================

    refresh_token = create_refresh_token(
        data={
            "sub": db_user.email
        },
        version=db_user.token_version,
    )

    # ========================================================
    # RESPONSE
    # ========================================================

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",

        "user": {
            "id": db_user.id,
            "email": db_user.email,
            "name": db_user.name,

            # Backend remains source of truth.
            "is_first_session": (
                db_user.is_first_session
            ),
        },
    }