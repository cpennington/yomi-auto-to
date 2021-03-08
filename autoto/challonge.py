import attr
import datetime
import challonge as pychal
import click
from lazy import lazy
import textwrap
from .db import get_template, tournaments
from collections import defaultdict
from typing import TypedDict


def on_day(date, day):
    return date + datetime.timedelta(days=(day - date.weekday()) % 7)


class TournamentData(TypedDict):
    tournament_type: str
    name: str
    id: int
    completed_at: datetime.datetime
    game_name: str


@attr.s
class Tournament:
    data: TournamentData = attr.ib()

    @classmethod
    def load(id):
        return Tournament(pychal.tournaments.show(id))

    @lazy
    def matches(self):
        return [
            Match(self, match_data)
            for match_data in pychal.matches.index(self.data["id"])
        ]

    @lazy
    def matches_by_id(self):
        return {match.data["id"]: match for match in self.matches}

    @property
    def pending_matches(self):
        if self.data["tournament_type"] in (
            "single elimination",
            "double elimination",
            "swiss",
            "round robin",
        ):
            for match in self.matches:
                if match.ready:
                    yield match

    def matches_for_round(self, round):
        for match in self.matches:
            if match.data["round"] == round:
                yield match

    @lazy
    def rounds(self):
        return {match.data["round"] for match in self.matches}

    @lazy
    def preceding_rounds(self):
        return defaultdict(
            set,
            {
                round: (
                    {
                        match.player1_prereq_match.data["round"]
                        for match in self.matches
                        if match.data["round"] == round
                        and match.player1_prereq_match is not None
                        and match.player1_prereq_match.round != round
                    }
                    | {
                        match.player2_prereq_match.data["round"]
                        for match in self.matches
                        if match.data["round"] == round
                        and match.player2_prereq_match is not None
                        and match.player2_prereq_match.round != round
                    }
                    | {round - 1 if round > 0 else round + 1}
                )
                & self.rounds
                for round in self.rounds
            },
        )

    @lazy
    def following_rounds(self):
        return defaultdict(
            set,
            {
                round: (
                    {
                        match.data["round"]
                        for match in self.matches
                        if match.player1_prereq_match is not None
                        and match.player1_prereq_match.data["round"] == round
                        and match.round != round
                    }
                    | {
                        match.data["round"]
                        for match in self.matches
                        if match.player2_prereq_match is not None
                        and match.player2_prereq_match.data["round"] == round
                        and match.round != round
                    }
                    | {round + 1 if round > 0 else round - 1}
                )
                & self.rounds
                for round in self.rounds
            },
        )

    @lazy
    def round_due_dates(self):
        first_round_due = on_day(
            datetime.datetime.combine(self.data["start_at"].date(), datetime.time())
            + datetime.timedelta(days=1),
            6,
        )
        dates = {
            round: first_round_due
            for round in self.rounds
            if not self.preceding_rounds[round]
        }
        rounds_to_process = []
        for round in dates:
            rounds_to_process.extend(self.following_rounds[round])

        while rounds_to_process:
            next_to_process = rounds_to_process.pop(0)
            if all(
                prev_round in dates
                for prev_round in self.preceding_rounds[next_to_process]
            ):
                dates[next_to_process] = max(
                    dates[prev_round]
                    for prev_round in self.preceding_rounds[next_to_process]
                ) + datetime.timedelta(days=7)

                for round in dates:
                    rounds_to_process.extend(
                        round
                        for round in self.following_rounds[next_to_process]
                        if round not in rounds_to_process
                    )

            else:
                rounds_to_process.append(next_to_process)

        for round in sorted(self.rounds, key=dates.get, reverse=True):
            if self.following_rounds[round]:
                dates[round] = min(
                    dates[next_round] for next_round in self.following_rounds[round]
                ) - datetime.timedelta(days=7)

        return dates

    @lazy
    def participants(self):
        return [
            Participant(participant_data)
            for participant_data in pychal.participants.index(self.data["id"])
        ]

    @lazy
    def template(self):
        return get_template(self.data["id"], self.data["name"])

    @lazy
    def slug(self):
        existing = tournaments.find_one(challonge_id=self.data["id"])
        if existing and existing["slug"]:
            return existing["slug"]
        else:
            slug = click.prompt(f'Slug for {self.data["name"]}?')
            tournaments.upsert(
                {"challonge_id": self.data["id"], "slug": slug}, keys=["challonge_id"]
            )
            return slug

    @lazy
    def co_tos(self):
        return [
            co_to
            for co_to in (
                (tournaments.find_one(challonge_id=self.data["id"]) or {})
                .get("tos", "")
                .split(",")
            )
            if co_to
        ]


@attr.s
class Match:
    tournament = attr.ib()
    data = attr.ib()

    @property
    def complete(self):
        return self.data["completed_at"] is not None

    @property
    def expired(self):
        return datetime.datetime.utcnow() > self.tournament.round_due_dates[self.round]

    @property
    def round(self):
        return self.data["round"]

    @lazy
    def ready(self):

        has_p1 = self.player1 is not None
        has_p2 = self.player2 is not None

        p1_finished_preceding = has_p1 and all(
            match.complete or match.expired
            for match in self.tournament.matches
            if match.round in self.tournament.preceding_rounds[self.round]
            and (match.player1 == self.player1 or match.player2 == self.player1)
        )

        p2_finished_preceding = has_p1 and all(
            match.complete or match.expired
            for match in self.tournament.matches
            if match.round in self.tournament.preceding_rounds[self.round]
            and (match.player1 == self.player2 or match.player2 == self.player2)
        )

        return (
            not self.complete
            and has_p1
            and has_p2
            and p1_finished_preceding
            and p2_finished_preceding
        )

    @lazy
    def player1(self):
        for participant in self.tournament.participants:
            if participant.data["id"] == self.data["player1_id"] or (
                self.data["player1_id"] in participant.data["group_player_ids"]
            ):
                return participant
        return None

    @lazy
    def player2(self):
        for participant in self.tournament.participants:
            if participant.data["id"] == self.data["player2_id"] or (
                self.data["player2_id"] in participant.data["group_player_ids"]
            ):
                return participant
        return None

    @lazy
    def player1_prereq_match(self):
        prereq_id = self.data.get("player1_prereq_match_id")
        if prereq_id:
            return self.tournament.matches_by_id[prereq_id]
        else:
            return None

    @lazy
    def player2_prereq_match(self):
        prereq_id = self.data.get("player2_prereq_match_id")
        if prereq_id:
            return self.tournament.matches_by_id[prereq_id]
        else:
            return None


@attr.s
class Participant:
    data = attr.ib()
