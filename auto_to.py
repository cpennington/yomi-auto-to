#! /usr/bin/env python3

import challonge as pychal
import click
import dataset
import dateutil.parser
import itertools
import logging
import sys
import textwrap
import json
from enum import Enum
from forum import Forum
from random import Random


db = dataset.connect('sqlite:///autoto.db')


MARKER = "## TEXT BELOW THIS IS IGNORED"

VARIABLES = textwrap.dedent("""\
    The following variables are available:
        {name}: The name of the tournament
        {round}: The current tournament round
        {player1}: The name of player 1
        {player2}: The name of player 2
        {due}: The duedate of the match
""")

TITLE_TEMPLATE = textwrap.dedent(f"""\
    {{name}} - R{{round}} - {{player1}}/{{player2}}
    {MARKER}
    {VARIABLES}
""")

BODY_TEMPLATE = textwrap.dedent(f"""\
    Hello, @{{player1}} and @{{player2}},

    You have until **{{due:%A, %B %d}}** @ 11:59pm GMT to complete your match.

    If you are posting to the scheduling conversation first, provide a list of dates and times when you are available to play. If you are posting to the scheduling conversation second, pick a specific start time and date for your match from those provided by your opponent. If none work for you, propose specific alternatives.

    Use WorldTimeBuddy.com to coordinate your match across different timezones. See [this post](http://forums.sirlingames.com/t/yomi-tournament-iyl-5/2639/84) for a guide.

    Reporting: Either the victor or the vanquished (or both) should post results in [this thread](http://forums.sirlingames.com/t/tournament-lums-long-odds-announcement-signup/3543).

    Best of luck!
    {MARKER}
    {VARIABLES}
""")


def pick_tournament():
    tournaments = pychal.tournaments.index()

    for id, tournament in enumerate(tournaments):
        click.echo(f"{id}: {tournament['name']}")

    choice = click.prompt("Which tournament?", type=int)

    return tournaments[choice]


def remove_marker(template):
    return template.split(MARKER)[0]


def get_template(tournament_id):
    table = db['templates']
    template = table.find_one(tournament=tournament_id)

    if not template:
        template = {
            'tournament': tournament_id,
            'title': remove_marker(click.edit(text=TITLE_TEMPLATE, require_save=False)),
            'body': remove_marker(click.edit(text=BODY_TEMPLATE, require_save=False)),
        }
        table.insert(template)

    return template


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


def all_tournaments():
    tournaments = pychal.tournaments.index()
    for tournament in tournaments:
        yield tournament


def matches_in_tournament(tournament):
    matches = pychal.matches.index(tournament['id'])
    participants = pychal.participants.index(tournament['id'])

    participant_names = {
        participant['id']: participant
        for participant in participants
    }

    for match in matches:
        yield (
            tournament,
            match,
            participant_names[match['player1_id']] if match['player1_id'] else None,
            participant_names[match['player2_id']] if match['player2_id'] else None,
        )


def match_is_pending(match):
    return not match['completed_at'] and match['player1_id'] and match['player2_id']


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

@autoto.group()
@click.option('--challonge-username', prompt="Challonge Username", envvar='AUTOTO_CHALLONGE_USERNAME')
@click.password_option('--challonge-api-key', confirmation_prompt=False, prompt="Challonge API Key", envvar='AUTOTO_CHALLONGE_API_KEY')
def challonge(challonge_username, challonge_api_key):
    pychal.set_credentials(challonge_username, challonge_api_key)

@challonge.command('send-pending-matches')
@click.pass_context
def send_pending_matches(ctx):
    for tournament in all_tournaments():
        template = get_template(tournament['id'])
        pending_matches = (
            (
                tournament['id'],
                tournament['name'],
                match['id'],
                match['round'],
                player1['display_name'],
                player2['display_name'],
            )
            for (tournament, match, player1, player2)
            in matches_in_tournament(tournament)
            if match_is_pending(match)
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
        if standings.find_one(username=player, week=week+1):
            click.echo(f"{player} already has recorded standings for week {week}")
            return 1

    with db as tx:
        match['winner'] = winner
        tx['matches'].update(match, ['id'])

        winner_prior = tx['ranked_standings'].find_one(username=winner, week=week)
        winner_next = {
            'username': winner,
            'stars': winner_prior['stars'] + winner_prior['tournament_round'] + 1,
            'games_played': winner_prior['games_played'] + 1,
            'week': week + 1,
            'tournament_round': (winner_prior['tournament_round'] + 1) % 3,
            'active': True,
        }
        tx['ranked_standings'].insert(winner_next)

        loser_prior = tx['ranked_standings'].find_one(username=loser, week=week)
        loser_next = {
            'username': loser,
            'stars': max(loser_prior['stars'] - 1, 0),
            'games_played': loser_prior['games_played'] + 1,
            'week': week + 1,
            'tournament_round': 0,
            'active': True,
        }
        tx['ranked_standings'].insert(loser_next)


@ranked.command('mark-inactive')
@click.argument('player')
@click.argument('week', type=int)
def mark_inactive(player, week):
    db['ranked_standings'].update(
        {'username': player, 'week': week, 'active': False},
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

    template = get_template('ranked')
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
        p1_bracket = bracket_history(
            matches.find_one(
                winner=p1,
                tournament=RANKED_TOURNAMENT_ID,
                order_by=['-round']
            )
        )

    if p2_standings['tournament_round'] == 0:
        p2_bracket = p2
    else:
        p2_bracket = bracket_history(
            matches.find_one(
                winner=p2,
                tournament=RANKED_TOURNAMENT_ID,
                order_by=['-round']
            )
        )

    return Bracket(
        p1_bracket,
        p2_bracket,
        match['winner'],
    )


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


@autoto.command()
@click.pass_context
def calendar(ctx):
    print(json.dumps(ctx.obj['forum'].messages().json(), indent=4))



if __name__ == '__main__':
    sys.exit(autoto())
