#! /usr/bin/env python3

import challonge as pychal
import click
import dataset
import dateutil.parser
from dateutil.tz import gettz
import itertools
import logging
import sys
import textwrap
import json
import re
import shutil
from enum import Enum
from forum import Forum
from random import Random
from gcal import Calendar
from autoto.challonge import Tournament
from autoto.db import get_template, templates


db = dataset.connect('sqlite:///autoto.db')


def pick_tournament():
    tournaments = pychal.tournaments.index()

    for id, tournament in enumerate(tournaments):
        click.echo(f"{id}: {tournament['name']}")

    choice = click.prompt("Which tournament?", type=int)

    return tournaments[choice]


def correct_user(forum, username):
    search_result = forum.search_user(username)
    search_result.raise_for_status()
    search_result = search_result.json()['users']

    if len(search_result) > 1:
        raise Exception(f"{username} is ambiguous")
    if len(search_result) == 0:
        raise Exception(f"{username} not found")
    corrected = search_result[0]['username']
    if corrected != username:
        click.confirm(f"Did you mean {corrected} rather than {username}?")

    return corrected


def send_messages(forum, pending_matchups, template):
    rounds = db['rounds']
    matches = db['matches']

    for (tournament_id, tournament_name, match_id, match_round, player1, player2) in pending_matchups:
        recorded_match = matches.find_one(match=match_id)
        already_sent = recorded_match is not None and recorded_match['sent']

        if already_sent:
            continue

        round = rounds.find_one(
            tournament=tournament_id,
            round=match_round,
        )

        if round is None:
            due_date = click.prompt(f"When is round {match_round} due?", type=dateutil.parser.parse)
            round = {
                'tournament': tournament_id,
                'round': match_round,
                'due_date': due_date,
            }
            rounds.insert(round)

        player1_name = correct_user(forum, player1)
        player2_name = correct_user(forum, player2)

        round_id = str(abs(match_round))
        if match_round < 0:
            round_id += 'L'

        title = template['title'].format(
            name=tournament_name,
            round=round_id,
            player1=player1_name,
            player2=player2_name,
            due=round['due_date'],
        )
        body = template['body'].format(
            name=tournament_name,
            round=round_id,
            player1=player1_name,
            player2=player2_name,
            due=round['due_date'],
        )

        send = "Resend" if already_sent else "Send"
        should_send = click.confirm(f"{title}\n======\n{body}\n{send}?")

        if should_send:
            try:
                resp = forum.send_private_message([player1_name, player2_name], title, body)
                resp.raise_for_status()
                matches.insert({
                    'tournament': tournament_id,
                    'round': round_id,
                    'match': match_id,
                    'player1': player1_name,
                    'player2': player2_name,
                    'thread_id': resp.json()['post']['id'],
                    'sent': True,
                })
            except:
                logging.exception(f"Unable to send matchup message. Response: {resp.json()}")
        else:
            send_later = click.confirm("Send later?")
            if not send_later:
                matches.insert({
                    'tournament': tournament_id,
                    'round': round_id,
                    'match': match_id,
                    'player1': player1_name,
                    'player2': player2_name,
                    'sent': True,
                })



def all_tournaments(domains=None):
    if domains is None:
        domains = []
    domains.append(None)

    for domain in domains:
        tournaments = pychal.tournaments.index(subdomain=domain)
        for tournament in tournaments:
            if not tournament['completed_at']:
                yield Tournament(tournament)


def matches_in_tournament(tournament):
    participant_names = {
        participant.data['id']: participant
        for participant in tournament.participants
    }

    for match in tournament.matches:
        yield (
            tournament,
            match,
            participant_names[match.data['player1_id']] if match.data['player1_id'] else None,
            participant_names[match.data['player2_id']] if match.data['player2_id'] else None,
        )


def match_is_pending(match):
    return not match.data['completed_at'] and match.data['player1_id'] and match.data['player2_id']


@click.group()
@click.option('--forum-username', prompt="Forum Username", envvar='AUTOTO_FORUM_USERNAME')
@click.password_option('--forum-password', confirmation_prompt=False, prompt="Forum Password", envvar='AUTOTO_FORUM_PASSWORD')
@click.pass_context
def autoto(ctx, forum_username, forum_password):
    ctx.obj = {
        'forum': Forum(
            'http://forums.sirlingames.com',
            forum_username,
            forum_password,
        )
    }
    shutil.copyfile('autoto.db', 'autoto.db.bak')

@autoto.group()
@click.option('--challonge-username', prompt="Challonge Username", envvar='AUTOTO_CHALLONGE_USERNAME')
@click.password_option('--challonge-api-key', confirmation_prompt=False, prompt="Challonge API Key", envvar='AUTOTO_CHALLONGE_API_KEY')
def challonge(challonge_username, challonge_api_key):
    pychal.set_credentials(challonge_username, challonge_api_key)

@challonge.command('send-pending-matches')
@click.pass_context
def send_pending_matches(ctx):
    for tournament in all_tournaments(['iyl']):
        template = tournament.template
        pending_matches = (
            (
                tournament.data['id'],
                tournament.data['name'],
                match.data['id'],
                match.data['round'],
                match.player1.data['display_name'],
                match.player2.data['display_name'],
            )
            for match
            in tournament.pending_matches
        )
        send_messages(ctx.obj['forum'], pending_matches, template)


@autoto.group()
def ranked():
    pass


@ranked.command('add')
@click.argument('user')
@click.argument('week', type=int)
def add_participant(user, week):
    participants = db['ranked_standings']
    participants.insert({
        'username': user,
        'tournament_round': 0,
        'stars': 0,
        'games_played': 0,
        'week': week,
        'active': True,
    })

@ranked.command('record-win')
@click.argument('winner')
@click.argument('loser')
@click.argument('week', type=int)
def record_win(winner, loser, week):
    matches = db['matches']
    standings = db['ranked_standings']

    match = matches.find_one(
        tournament=RANKED_TOURNAMENT_ID,
        round=week,
        player1=winner,
        player2=loser,
    ) or matches.find_one(
        tournament=RANKED_TOURNAMENT_ID,
        round=week,
        player1=loser,
        player2=winner,
    )

    if match is None:
        click.echo("Match not found", err=True)
        return 1

    for player in (winner, loser):
        existing_standings = standings.find_one(username=player, week=week+1)
        if existing_standings and existing_standings['tournament_round']:
            click.echo(f"{player} already has recorded standings for week {week}")
            return 1

    with db as tx:
        match['winner'] = winner
        tx['matches'].update(match, ['id'])

        winner_prior = tx['ranked_standings'].find_one(username=winner, week=week)
        loser_prior = tx['ranked_standings'].find_one(username=loser, week=week)

        winner_next = {
            'username': winner,
            'stars': winner_prior['stars'] + max(winner_prior['tournament_round'], loser_prior['tournament_round']) + 1,
            'games_played': winner_prior['games_played'] + 1,
            'week': week + 1,
            'tournament_round': (winner_prior['tournament_round'] + 1) % 3,
        }
        tx['ranked_standings'].upsert(winner_next, ['username', 'week'])

        loser_next = {
            'username': loser,
            'stars': max(loser_prior['stars'] - 1, 0),
            'games_played': loser_prior['games_played'] + 1,
            'week': week + 1,
            'tournament_round': 0,
        }
        tx['ranked_standings'].upsert(loser_next, ['username', 'week'])


@ranked.command('mark-inactive')
@click.argument('player')
@click.argument('week', type=int)
def mark_inactive(player, week):
    db['ranked_standings'].upsert(
        {'username': player, 'week': week, 'active': False},
        ['username', 'week']
    )


@ranked.command('mark-active')
@click.argument('player')
@click.argument('week', type=int)
def mark_active(player, week):
    db['ranked_standings'].upsert(
        {'username': player, 'week': week, 'active': True},
        ['username', 'week']
    )

class League(Enum):
    Bronze = 0
    SuperBronze = 1
    Silver = 2
    SuperSilver = 3
    Gold = 4

    @classmethod
    def from_stars(cls, stars):
        return cls(stars // 5)

    def display_name(self):
        return {
            'SuperBronze': 'Super Bronze',
            'SuperSilver': 'Super Silver'
        }.get(self.name, self.name)


def match_cost(player1, player2):
    win_cost = abs(player1['tournament_round'] - player2['tournament_round']) * 50
    league1 = League.from_stars(player1['stars'])
    league2 = League.from_stars(player2['stars'])
    league_cost = max(abs(league1.value - league2.value) - 1, 0) * 10
    return league_cost + win_cost


def greedy_matches(participants):
    rand = Random(0)
    participants = list(participants)
    rand.shuffle(participants)
    scheduling_priority = sorted(
        participants,
        key=lambda p: p['games_played'] or 0
    )

    while len(participants) > 1:
        player1 = scheduling_priority[0]
        player2 = min([p for p in participants if p != player1], key=lambda x: match_cost(player1, x))

        yield player1, player2
        participants.remove(player1)
        participants.remove(player2)
        scheduling_priority.remove(player1)
        scheduling_priority.remove(player2)


def ranked_match_id(round, player1, player2):
    return f"ranked-{round}-{player1['username']}-{player2['username']}"


RANKED_TOURNAMENT_ID = 'ranked'

@ranked.command('send-matches')
@click.pass_context
@click.argument('week', type=int)
def send_ranked_matches(ctx, week):
    standings = db['ranked_standings']

    ps = list(standings.find(week=week, active=True, order_by=['-stars', 'username']))

    matches = list(greedy_matches(ps))

    for p1, p2 in matches:
        print(f'{p1["username"]}/{p2["username"]}')

    template = get_template('ranked', 'Forums Quick Matches')
    send_messages(
        ctx.obj['forum'],
        ((
            RANKED_TOURNAMENT_ID,
            'Ranked',
            ranked_match_id(week, player1, player2),
            week,
            player1['username'],
            player2['username'],
        ) for (player1, player2) in matches),
        template,
    )


class Box:
    def __init__(self, data):
        if isinstance(data, str):
            data = data.split('\n')
        self.data = data

    @property
    def width(self):
        return max((len(row) for row in self.data))

    @property
    def height(self):
        return len(self.data)

    @classmethod
    def empty(cls, width, height):
        return cls([" "*width]*height)

    @classmethod
    def vertical(cls, *boxes):
        return Box(sum((box.data for box in boxes), []))

    def left_of(self, other):
        pad_to = max(len(row) for row in self.data)
        return Box([
            f"{{:{pad_to}}}{{}}".format(left, right)
            for (left, right)
            in zip(self.data, other.data)
        ])

    def __str__(self):
        return "\n".join(row.rstrip() for row in self.data)

class Bracket:
    def __init__(self, top, bottom, winner=None):
        self.top = top
        self.bottom = bottom
        self.winner = winner

        self.upper_corner = '\N{BOX DRAWINGS LIGHT DOWN AND LEFT}'
        self.upper_v = '\N{BOX DRAWINGS LIGHT VERTICAL}'
        self.upper_h = '\N{BOX DRAWINGS LIGHT HORIZONTAL}'
        self.lower_corner = '\N{BOX DRAWINGS LIGHT UP AND LEFT}'
        self.lower_v = '\N{BOX DRAWINGS LIGHT VERTICAL}'
        self.lower_h = '\N{BOX DRAWINGS LIGHT HORIZONTAL}'
        self.join = '\N{BOX DRAWINGS LIGHT VERTICAL AND RIGHT}'
        self.center_h = '\N{BOX DRAWINGS LIGHT HORIZONTAL}'

        if winner == getattr(top, 'winner', top):
            self.upper_corner = '\N{BOX DRAWINGS HEAVY DOWN AND LEFT}'
            self.upper_v = '\N{BOX DRAWINGS HEAVY VERTICAL}'
            self.upper_h = '\N{BOX DRAWINGS HEAVY HORIZONTAL}'
            self.join = '\N{BOX DRAWINGS DOWN LIGHT AND RIGHT UP HEAVY}'
            self.center_h = '\N{BOX DRAWINGS HEAVY HORIZONTAL}'
        elif winner == getattr(bottom, 'winner', bottom):
            self.lower_corner = '\N{BOX DRAWINGS HEAVY UP AND LEFT}'
            self.lower_v = '\N{BOX DRAWINGS HEAVY VERTICAL}'
            self.lower_h = '\N{BOX DRAWINGS HEAVY HORIZONTAL}'
            self.join = '\N{BOX DRAWINGS UP LIGHT AND RIGHT DOWN HEAVY}'
            self.center_h = '\N{BOX DRAWINGS HEAVY HORIZONTAL}'

    @property
    def names(self):
        return getattr(self.top, 'names', [self.top]) + getattr(self.bottom, 'names', [self.bottom])

    def render(self, name_width):
        if isinstance(self.top, Bracket):
            top_box = Box(self.top.render(name_width))
        else:
            top_box = Box(self.top)

        if isinstance(self.bottom, Bracket):
            bottom_box = Box(self.bottom.render(name_width))
        else:
            bottom_box = Box(self.bottom)

        max_width = max(top_box.width, bottom_box.width, name_width)
        top_padding = Box(
            [''] * (top_box.height // 2) +
            [' ' + self.upper_h * (max_width - top_box.width)] +
            [''] * (top_box.height // 2)
        )
        bottom_padding = Box(
            [''] * (bottom_box.height // 2) +
            [' ' + self.lower_h * (max_width - bottom_box.width)] +
            [''] * (bottom_box.height // 2)
        )
        border = Box.empty(max_width, 1)

        top_bar_length = top_box.height // 2
        bottom_bar_length = bottom_box.height // 2

        winner = self.winner or ''

        bar = Box(
            [""] * top_bar_length +
            [f'{self.upper_h}{self.upper_corner}'] +
            [f' {self.upper_v}'] * top_bar_length +
            [f' {self.join}{self.center_h} {winner}'] +
            [f' {self.lower_v}'] * bottom_bar_length +
            [f'{self.lower_h}{self.lower_corner}'] +
            [""] * bottom_bar_length
        )

        return str(Box.vertical(
            top_box.left_of(top_padding),
            border,
            bottom_box.left_of(bottom_padding)
        ).left_of(bar))

    def __str__(self):
        return self.render(max(len(name) for name in self.names))


def bracket_history(match):
    standings = db['ranked_standings']
    matches = db['matches']

    p1 = match['player1']
    p2 = match['player2']

    p1_standings = standings.find_one(username=p1, week=match['round'])
    p2_standings = standings.find_one(username=p2, week=match['round'])

    if p1_standings['tournament_round'] == 0:
        p1_bracket = p1
    else:
        old_matches = matches.find(
            winner=p1,
            tournament=RANKED_TOURNAMENT_ID,
            order_by=['-round']
        )
        previous_match = [
            old_match for old_match in old_matches if old_match['round'] < match['round']
        ][0]
        p1_bracket = bracket_history(previous_match)

    if p2_standings['tournament_round'] == 0:
        p2_bracket = p2
    else:
        old_matches = matches.find(
            winner=p2,
            tournament=RANKED_TOURNAMENT_ID,
            order_by=['-round']
        )
        previous_match = [
            old_match for old_match in old_matches if old_match['round'] < match['round']
        ][0]
        p2_bracket = bracket_history(previous_match)

    return Bracket(
        p1_bracket,
        p2_bracket,
        match['winner'],
    )


@ranked.command('finalize-week')
@click.pass_context
@click.argument('week', type=int)
def finalize_week(ctx, week):
    standings = db['ranked_standings']
    matches = db['matches']

    pending_matches = matches.find(week=week, winner=None)
    for match in pending_matches:
        print(match)
        return

    to_finalize = standings.find(week=week)
    for current_standings in to_finalize:
        print("Finalizing", current_standings)
        future_standings = standings.find_one(week=week+1, username=current_standings['username']) or {}
        for field, value in current_standings.items():
            if future_standings.get(field) is None:
                future_standings[field] = current_standings[field]
        future_standings.pop('id', None)
        future_standings['week'] = week + 1
        print("Finalized to", future_standings)
        standings.upsert(future_standings, keys=['id', 'username', 'week'])


@ranked.command('post-summary')
@click.pass_context
@click.argument('week', type=int)
def post_week_summary(ctx, week):
    standings = db['ranked_standings']

    ps = list(standings.find(week=week, order_by=['-active', '-stars', 'username']))

    post = []

    post.append(f'# Week {week} standings')
    for league, players in itertools.groupby(ps, lambda p: League.from_stars(p['stars'])):
        post.append(f'## {league.name} League')
        for player in players:
            post.append('* {username}{inactive}: {stars}'.format(
                username=player['username'],
                stars=":star:" * (player['stars'] % 10),
                inactive=" (inactive)" if not player['active'] else '',
            ))
    post.append('')
    post.append(f'# Week {week} matches')

    matches = db['matches'].find(tournament=RANKED_TOURNAMENT_ID, round=week)
    for match in matches:
        post.append('[details="{player1}/{player2}"]'.format(**match))
        post.append('<pre>')
        post.append(str(bracket_history(match)))
        post.append('</pre>')
        post.append('[/details]')

    print('\n'.join(post))


def tz(hours):
    return hours * 60 * 60


TZINFOS = {
    'EST': tz(-5),
    'CST': tz(-6),
    'MST': tz(-7),
    'PST': tz(-8),
    'CET': tz(1),
}

def scheduled_times(posts):
    for post in posts:
        match = re.search('<p>AutoTO: Schedule @ (.*)</p>', post['cooked'], flags=re.RegexFlag.IGNORECASE)

        if match is None:
            continue

        scheduled_time = match.group(1)
        yield (
            post['updated_at'],
            scheduled_time,
            dateutil.parser.parse(scheduled_time, tzinfos=TZINFOS),
        )


def prompt_event(title, string, date, url):
    while True:
        action = click.prompt(
            textwrap.dedent(f"""\
                Scheduling "{title}" at {date} from "{string}"
                Originally from {url}

                y(es)/n(o)/e(dit)"""),
            default='N',
        )
        if action[0].lower() == 'e':
            title = click.edit(title, require_save=False).strip()
            date = dateutil.parser.parse(click.edit(string, require_save=False),  tzinfos=TZINFOS)
        elif action[0].lower() == 'y':
            return title, date

@autoto.command()
@click.pass_context
def calendar(ctx):
    scheduled_matches = db['scheduled_matches']

    messages = ctx.obj['forum'].messages().json()['topic_list']['topics']
    for message in messages:
        posts = sorted(
            scheduled_times(ctx.obj['forum'].message_posts(message['id'])),
            reverse=True,
        )
        if posts:
            updated_at, scheduled_string, scheduled_date = posts[0]
            current_event = scheduled_matches.find_one(message_id=message['id'])
            if current_event is None:
                title, date = prompt_event(
                    message['title'],
                    scheduled_string,
                    scheduled_date,
                    ctx.obj['forum'].url(f"t/{message['id']}"),
                )
                scheduled_event = Calendar('5qcrghv93ken5kco8e0eeqo3do@group.calendar.google.com').insert_event(
                    title, date
                )
                scheduled_matches.insert({
                    'message_id': message['id'],
                    'event_id': scheduled_event,
                    'updated_at': updated_at,
                })
            elif current_event['updated_at'] < updated_at:
                title, date = prompt_event(
                    message['title'],
                    scheduled_string,
                    scheduled_date,
                    ctx.obj['forum'].url(f"t/{message['id']}"),
                )
                scheduled_event = Calendar('5qcrghv93ken5kco8e0eeqo3do@group.calendar.google.com').update_event(
                    current_event['event_id'],
                    title,
                    date,
                )
                scheduled_matches.update({
                    'message_id': message['id'],
                    'event_id': scheduled_event,
                    'updated_at': updated_at,
                }, keys=['message_id'])


@autoto.command('edit-template')
@click.argument('id')
def edit_template(id):
    existing = templates.find_one(tournament=id)
    existing['title'] = click.edit(existing['title'], require_save=False)
    existing['body'] = click.edit(existing['body'], require_save=False)
    templates.update(existing, keys=['tournament'])

if __name__ == '__main__':
    sys.exit(autoto())
