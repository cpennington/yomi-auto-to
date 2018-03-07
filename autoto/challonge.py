import attr
import challonge as pychal
import click
from lazy import lazy
import textwrap
from .db import get_template, tournaments


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

    def matches_for_round(self, round):
        for match in self.matches:
            if match.data['round'] == round:
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

    @lazy
    def slug(self):
        existing = tournaments.find_one(challonge_id=self.data['id'])
        if existing and existing['slug']:
            return existing['slug']
        else:
            slug = click.prompt(f'Slug for {self.data["name"]}?')
            tournaments.upsert({
                'challonge_id': self.data['id'],
                'slug': slug,
            }, keys=['challonge_id'])
            return slug

    @lazy
    def co_tos(self):
        return (tournaments.find_one(challonge_id=self.data['id']) or {}).get('tos', '').split(',')


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
