# routers/onboarding.py
"""
Onboarding + pet endpoints.

  GET   /pets/catalog          — pet list (emoji, trait, evolution line)
  GET   /users/me/onboarding   — where is this user in onboarding?
  POST  /users/me/pet          — adopt (moved here from main.py)
  PATCH /users/me/pet          — rename ("you can change this anytime")

The onboarding step is decided by the server, so splash / login /
quiz all route the same way:

  quiz  → goal_type not chosen yet
  pet   → no pet yet
  done  → finished (the goal step after the pet is optional, so it
          never blocks anyone)
"""

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlmodel import Session

from database import get_session
from gamification import (
    PET_CATALOG,
    STAGE_MIN_LEVELS,
    XP_PER_LEVEL,
    catalog_list,
    get_pet,
    pet_view,
)
from models import Pet, User
from schemas import PetAdoptionRequest
from security import get_current_user
from utils import get_local_now

router = APIRouter(tags=["Onboarding & Pet"])


class PetRenameRequest(BaseModel):
    pet_name: str = Field(
        ...,
        min_length=1,
        max_length=20,
        pattern=r"^[a-zA-Z0-9 \-']+$",
    )


def onboarding_step(db: Session, user: User) -> str:
    if not user.is_first_session:
        return "done"
    if not user.goal_type:
        return "quiz"
    if get_pet(db, user.id) is None:
        return "pet"
    return "done"


# ------------------------------------------------------------
# CATALOG
# ------------------------------------------------------------

@router.get("/pets/catalog", status_code=status.HTTP_200_OK)
def get_pet_catalog():
    return {
        "pets": catalog_list(),
        "stage_levels": STAGE_MIN_LEVELS,
        "xp_per_level": XP_PER_LEVEL,
    }


# ------------------------------------------------------------
# ONBOARDING STATE
# ------------------------------------------------------------

@router.get("/users/me/onboarding", status_code=status.HTTP_200_OK)
def get_onboarding_state(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
    x_timezone: str = Header("UTC"),
):
    pet = get_pet(db, current_user.id)
    today = get_local_now(x_timezone).date()
    return {
        "step": onboarding_step(db, current_user),
        "first_name": (current_user.name or "Student").split()[0],
        "goal_type": current_user.goal_type,
        "study_goal": current_user.study_goal,
        "target_date": current_user.target_date,
        "daily_goal_mins": current_user.daily_goal_mins,
        "has_pet": pet is not None,
        "pet": pet_view(pet, current_user, today) if pet else None,
    }


# ------------------------------------------------------------
# ADOPT
# ------------------------------------------------------------

@router.post("/users/me/pet", status_code=status.HTTP_200_OK)
def adopt_pet(
    pet_data: PetAdoptionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
    x_timezone: str = Header("UTC"),
):
    pet_name = pet_data.pet_name.strip()
    if not pet_name:
        raise HTTPException(status_code=400, detail="Pet name cannot be empty.")
    if pet_data.pet_type not in PET_CATALOG:
        raise HTTPException(status_code=400, detail="Invalid pet type.")

    if get_pet(db, current_user.id) is not None:
        raise HTTPException(status_code=409, detail="Already adopted!")

    new_pet = Pet(
        user_id=current_user.id,
        pet_type=pet_data.pet_type,
        pet_name=pet_name,
        level=1,
        xp=0,
    )
    db.add(new_pet)

    # Adopting finishes the required part of onboarding. The goal step
    # that follows is optional, so older app versions keep working.
    current_user.is_first_session = False
    db.add(current_user)

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Could not save your buddy. Please try again.",
        )

    db.refresh(new_pet)
    db.refresh(current_user)
    today = get_local_now(x_timezone).date()

    return {
        "pet_id": new_pet.id,
        "pet_type": new_pet.pet_type,
        "pet_name": new_pet.pet_name,
        "level": new_pet.level,
        "xp": new_pet.xp,
        "is_first_session": current_user.is_first_session,
        "next_step": "goal",
        "pet": pet_view(new_pet, current_user, today),
    }


# ------------------------------------------------------------
# RENAME
# ------------------------------------------------------------

@router.patch("/users/me/pet", status_code=status.HTTP_200_OK)
def rename_pet(
    payload: PetRenameRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    pet = get_pet(db, current_user.id)
    if pet is None:
        raise HTTPException(status_code=404, detail="You haven't adopted a buddy yet.")
    name = payload.pet_name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Pet name cannot be empty.")
    pet.pet_name = name
    db.add(pet)
    db.commit()
    return {"pet_name": pet.pet_name}