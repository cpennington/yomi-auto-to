#! /usr/bin/env python3

import click
import challonge
import textwrap
import dataset
import dateutil.parser
import logging
from forum import Forum
from cursesmenu import CursesMenu
from cursesmenu.items import FunctionItem


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
    tournaments = challonge.tournaments.index()

    for id, tournament in enumerate(tournaments):
        click.echo(f"{id}: {tournament['name']}")

    choice = click.prompt("Which tournament?", type=int)

    return tournaments[choice]


def remove_marker(template):
    return template.split(MARKER)[0]


def get_template(tournament):
    table = db['templates']
    template = table.find_one(tournament=tournament['id'])

    if not template:
        template = {
            'tournament': tournament['id'],
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


def send_messages(username, password, pending_matchups, template):
    forum = Forum('https://forums.sirlingames.com', username, password)
    rounds = db['rounds']
    matches = db['matches']

    for (tournament, match, player1, player2) in pending_matchups:
        recorded_match = matches.find_one(match=match['id'])
        already_sent = recorded_match is not None and recorded_match['sent']

        if already_sent:
            continue

        round = rounds.find_one(
            tournament=tournament['id'],
            round=match['round'],
        )

        if round is None:
            due_date = click.prompt(f"When is round {match['round']} due?", type=dateutil.parser.parse)
            round = {
                'tournament': tournament['id'],
                'round': match['round'],
                'due_date': due_date,
            }
            rounds.insert(round)

        player1_name = correct_user(forum, player1['display_name'])
        player2_name = correct_user(forum, player2['display_name'])

        round_id = str(abs(match['round']))
        if match['round'] < 0:
            round_id += 'L'

        title = template['title'].format(
            name=tournament['name'],
            round=round_id,
            player1=player1_name,
            player2=player2_name,
            due=round['due_date'],
        )
        body = template['body'].format(
            name=tournament['name'],
            round=round_id,
            player1=player1_name,
            player2=player2_name,
            due=round['due_date'],
        )

        send = "Resend" if already_sent else "Send"
        should_send = click.confirm(f"{title}\n======\n{body}\n{send}?")

        if should_send:
            try:
                resp = forum.private_message([player1_name, player2_name], title, body)
                resp.raise_for_status()
                matches.insert({'match': match['id'], 'sent': True})
            except:
                logging.exception(f"Unable to send matchup message. Response: {resp.json()}")


def all_tournaments():
    tournaments = challonge.tournaments.index()
    for tournament in tournaments:
        yield tournament


def matches_in_tournament(tournament):
    matches = challonge.matches.index(tournament['id'])
    participants = challonge.participants.index(tournament['id'])

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


def send_pending_matches(menu, forum_username, forum_password):
    menu.pause()

    for tournament in all_tournaments():
        template = get_template(tournament)
        pending_matches = (
            (tournament, match, player1, player2)
            for (tournament, match, player1, player2)
            in matches_in_tournament(tournament)
            if match_is_pending(match)
        )
        send_messages(forum_username, forum_password, pending_matches, template)

    menu.resume()


@click.command()
@click.option('--challonge-username', prompt="Challonge Username")
@click.password_option('--challonge-api-key', confirmation_prompt=False, prompt="Challonge API Key")
@click.option('--forum-username', prompt="Forum Username")
@click.password_option('--forum-password', confirmation_prompt=False, prompt="Forum Password")
def autoto(challonge_username, challonge_api_key, forum_username, forum_password):
    print(repr((challonge_username, challonge_api_key, forum_username, forum_password)))
    challonge.set_credentials(challonge_username, challonge_api_key)

    menu = CursesMenu("Yomi Auto-TO", "Select an action")
    menu.append_item(
        FunctionItem("Send pending matches", send_pending_matches, [menu, forum_username, forum_password])
    )
    menu.show()


if __name__ == '__main__':
    autoto(auto_envvar_prefix='AUTOTO')
