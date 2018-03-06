import attr
import challonge as pychal
import click
from lazy import lazy
import textwrap
from .db import get_template


@attr.s
class Tournament:
    data = attr.ib()

    @lazy
    def matches(self):
        return [
            Match(self, match_data)
            for match_data
            in pychal.matches.index(self.data['id'])
        ]

    @property
    def pending_matches(self):
        if self.data['tournament_type'] in ('single elimination', 'double elimination'):
            for match in self.matches:
                if not match.data['completed_at'] and match.data['player1_id'] and match.data['player2_id']:
                    yield match


    @lazy
    def participants(self):
        return [
            Participant(participant_data)
            for participant_data
            in pychal.participants.index(self.data['id'])
        ]

    @lazy
    def template(self):
        return get_template(self.data['id'], self.data['name'])


@attr.s
class Match:
    tournament = attr.ib()
    data = attr.ib()

    @lazy
    def player1(self):
        for participant in self.tournament.participants:
            if participant.data['id'] == self.data['player1_id']:
                return participant

    @lazy
    def player2(self):
        for participant in self.tournament.participants:
            if participant.data['id'] == self.data['player2_id']:
                return participant

@attr.s
class Participant:
    data = attr.ib()
