"""Version 2 recordings: pre-action states at 10 Hz, plus the terminal state."""
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

Pair = tuple[FiniteFloat, FiniteFloat]
Pose = tuple[FiniteFloat, FiniteFloat, FiniteFloat]


class State(BaseModel):
    agent: Pair
    block: Pose


class Frame(State):
    step: int = Field(ge=0)
    timestamp: FiniteFloat
    action: Pair


class InitialState(BaseModel):
    agent: Pair
    blockPos: Pair
    blockAngle: FiniteFloat
    seed: int = Field(ge=0, le=2**32 - 1)


class Goal(BaseModel):
    x: FiniteFloat
    y: FiniteFloat
    angle: FiniteFloat


class Episode(BaseModel):
    model_config = ConfigDict(extra="allow")
    schemaVersion: Literal[2]
    name: str = Field(min_length=1)
    engine: str = Field(min_length=1)
    simulatorSha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: Literal["product"]
    dt: Literal[0.01]
    controlHz: Literal[10]
    scale: FiniteFloat
    agentRadius: FiniteFloat
    goalPose: Goal
    successThreshold: Annotated[FiniteFloat, Field(gt=0, le=1)]
    init: InitialState
    success: Literal[True]
    nFrames: int = Field(ge=1)
    frames: list[Frame] = Field(min_length=1)
    finalState: State
    finalTimestamp: FiniteFloat

    @model_validator(mode="after")
    def check_timeline(self):
        if not self.name.strip():
            raise ValueError("name is required")
        if self.nFrames != len(self.frames):
            raise ValueError("nFrames does not match frames")
        for i, frame in enumerate(self.frames):
            if frame.step != i or not math.isclose(frame.timestamp, i / self.controlHz, abs_tol=1e-9):
                raise ValueError("frames must be consecutive control steps")
        if not math.isclose(self.finalTimestamp, self.nFrames / self.controlHz, abs_tol=1e-9):
            raise ValueError("incorrect final timestamp")
        first = self.frames[0]
        expected = (*self.init.agent, *self.init.blockPos, self.init.blockAngle)
        if any(abs(a - b) > 1e-4 for a, b in zip((*first.agent, *first.block), expected)):
            raise ValueError("first frame must match the initial state")
        return self
