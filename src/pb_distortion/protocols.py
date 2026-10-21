"""Shared interfaces that make ballot formats and rules easy to extend."""

from __future__ import annotations

from typing import Generic, Protocol, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray

from pb_distortion.models import Outcome, PBInstance


BallotT = TypeVar("BallotT", covariant=True)
BallotT_contra = TypeVar("BallotT_contra", contravariant=True)


class ElicitationFormat(Protocol, Generic[BallotT]):
    """Map one voter's latent values to a ballot."""

    name: str

    def elicit(
        self,
        utilities: NDArray[np.float64],
        instance: PBInstance,
    ) -> BallotT: ...


class AggregationRule(Protocol, Generic[BallotT_contra]):
    """Map ballots and public instance data to a feasible outcome."""

    name: str

    def aggregate(
        self,
        ballots: Sequence[BallotT_contra],
        instance: PBInstance,
    ) -> Outcome: ...
