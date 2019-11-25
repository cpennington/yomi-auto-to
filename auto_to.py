#! /usr/bin/env python3

import itertools
import json
import logging
import os.path
import re
import shutil
import sys
import textwrap
import time
from datetime import date, datetime, timedelta
from enum import Enum
from fnmatch import fnmatch
from random import Random

import challonge as pychal
import click
import click_log
import dataset
import dateparser
import pytz
from tabulate import tabulate
from lazy import lazy
from mako.template import Template
from prompt_toolkit import prompt

from autoto.challonge import Tournament
from autoto.db import get_template, templates, tournaments
from forum import Forum
from gcal import Calendar

logger = logging.getLogger(__name__)
click_log.basic_config(logger)

db = dataset.connect("sqlite:///autoto.db")


def correct_user(forum, username):
    search_result = forum.search_user(username)
    search_result.raise_for_status()
    search_result = search_result.json()["users"]

    if len(search_result) > 1:
        search_result = [
            result for result in search_result if result["username"] == username
        ]
    if len(search_result) > 1:
        raise Exception(f"{username} is ambiguous ({search_result})")
    if len(search_result) == 0:
        raise Exception(f"{username} not found")
    corrected = search_result[0]["username"]
    if corrected != username:
        click.confirm(f"Did you mean {corrected} rather than {username}?")

    return corrected


def send_messages(forum, pending_matchups, template):
    rounds = db["rounds"]
    matches = db["matches"]
    tournaments = db["tournaments"]
    templates = db["templates"]

    for (
        tournament_id,
        tournament_name,
        match_id,
        match_round,
        player1,
        player2,
        ccs,
    ) in pending_matchups:
        recorded_match = matches.find_one(match=match_id)
        already_sent = recorded_match is not None and recorded_match["sent"]

        if already_sent:
            continue

        round = rounds.find_one(tournament=tournament_id, round=match_round)

        if round is None:
            prev_round = rounds.find_one(
                tournament=tournament_id, round=match_round - 1
            )
            tournament = tournaments.find_one(id=tournament_id) or templates.find_one(
                tournament=tournament_id
            )
            if prev_round:
                default = prev_round["due_date"] + timedelta(days=7)
            else:
                default = date.today() + timedelta(days=7)
            due_date = None
            while due_date is None:
                due_date = click.prompt(
                    f"When is round {match_round} of {tournament['slug']} due?",
                    type=dateparser.parse,
                    default=default,
                )
            round = {
                "tournament": tournament_id,
                "round": match_round,
                "due_date": due_date,
            }
            rounds.insert(round)

        try:
            player1_name = correct_user(forum, player1)
            player2_name = correct_user(forum, player2)
        except Exception:
            logger.exception("Unable to locate user, skipping")
            continue

        round_id = str(abs(match_round))
        if match_round < 0:
            round_id += "L"

        title = (
            Template(template["title"])
            .render(
                id=tournament_id,
                name=tournament_name,
                round=round_id,
                player1=player1_name,
                player2=player2_name,
                due=round["due_date"],
                get_tournament_metadata=get_tournament_metadata,
            )
            .strip()
        )
        body = (
            Template(template["body"])
            .render(
                id=tournament_id,
                name=tournament_name,
                round=round_id,
                player1=player1_name,
                player2=player2_name,
                due=round["due_date"],
                get_tournament_metadata=get_tournament_metadata,
            )
            .strip()
        )

        send = "Resend" if already_sent else "Send"
        recipients = [player1_name, player2_name] + ccs
        should_send = click.confirm(
            f"To: {', '.join(recipients)}\n======\n{title}\n======\n{body}\n{send}?"
        )

        if should_send:
            while True:
                try:
                    resp = forum.send_private_message(recipients, title, body)
                    resp.raise_for_status()
                    matches.insert(
                        {
                            "tournament": tournament_id,
                            "round": round_id,
                            "match": match_id,
                            "player1": player1_name,
                            "player2": player2_name,
                            "thread_id": resp.json()["post"]["topic_id"],
                            "sent": True,
                        }
                    )
                    break
                except:
                    logger.exception(
                        f"Unable to send matchup message. Response: {resp.json()}"
                    )
                    time.sleep(30)
        else:
            send_later = click.confirm("Send later?")
            if not send_later:
                matches.insert(
                    {
                        "tournament": tournament_id,
                        "round": round_id,
                        "match": match_id,
                        "player1": player1_name,
                        "player2": player2_name,
                        "sent": True,
                    }
                )


def get_tournament_metadata(name, **keys):
    metadata = db["metadata"]
    row = metadata.find_one(name=name, **keys)
    if not row:
        value = prompt(
            "{keys} {name}=".format(
                keys=" ".join("{}={!r}".format(*item) for item in keys.items()),
                name=name,
            )
        )
        row = {"name": name, "value": value}
        row.update(keys)
        metadata.insert(row)

    return row["value"]


def all_tournaments(domains=None):
    if domains is None:
        domains = ()
    domains += (None,)

    for domain in domains:
        tournaments = pychal.tournaments.index(subdomain=domain)
        for tournament in tournaments:
            if not tournament["completed_at"]:
                yield Tournament(tournament)


def matches_in_tournament(tournament):
    participant_names = {
        participant.data["id"]: participant for participant in tournament.participants
    }

    for match in tournament.matches:
        yield (
            tournament,
            match,
            participant_names[match.data["player1_id"]]
            if match.data["player1_id"]
            else None,
            participant_names[match.data["player2_id"]]
            if match.data["player2_id"]
            else None,
        )


def match_is_pending(match):
    return (
        not match.data["completed_at"]
        and match.data["player1_id"]
        and match.data["player2_id"]
    )


@click.group()
@click.option(
    "--forum-username", prompt="Forum Username", envvar="AUTOTO_FORUM_USERNAME"
)
@click.password_option(
    "--forum-password",
    confirmation_prompt=False,
    prompt="Forum Password",
    envvar="AUTOTO_FORUM_PASSWORD",
)
@click.pass_context
@click_log.simple_verbosity_option(logger)
def autoto(ctx, forum_username, forum_password):
    ctx.obj = {
        "forum": Forum("http://forums.sirlingames.com", forum_username, forum_password)
    }
    if os.path.exists("autoto.db"):
        shutil.copyfile("autoto.db", "autoto.db.bak")


@autoto.group()
@click.option(
    "--challonge-username",
    prompt="Challonge Username",
    envvar="AUTOTO_CHALLONGE_USERNAME",
)
@click.password_option(
    "--challonge-api-key",
    confirmation_prompt=False,
    prompt="Challonge API Key",
    envvar="AUTOTO_CHALLONGE_API_KEY",
)
def challonge(challonge_username, challonge_api_key):
    pychal.set_credentials(challonge_username, challonge_api_key)


@challonge.command("prompt-expiring-matches")
@click.option("--domain", "domains", multiple=True)
@click.pass_context
def prompt_expiring_matches(ctx, domains):
    tournaments = {
        tournament.data["id"]: tournament for tournament in all_tournaments(domains)
    }
    pending_matches = {
        (match.player1.data["display_name"], match.player2.data["display_name"], match.data["round"])
        for tournament in tournaments.values()
        for match in tournament.pending_matches
    }
    results = db.query(
        """
            SELECT *
            FROM matches
            JOIN rounds
            ON matches.tournament = rounds.tournament
            AND matches.round = CASE WHEN rounds.round > 0 THEN rounds.round ELSE (-rounds.round || 'L') END
            WHERE winner IS NULL
            AND rounds.due_date BETWEEN :today AND :later
            AND rounds.tournament IN ({tournaments})
        """.format(
            tournaments=",".join(repr(id) for id in tournaments.keys())
        ),
        today=date.today(),
        later=date.today() + timedelta(days=3),
    )

    print(
        tabulate(
            (
                {
                    "Tournament": tournaments[result["tournament"]].data["name"],
                    "Player 1": result["player1"],
                    "Player 2": result["player2"],
                    "Round": result["round"],
                    "Due Date": result["due_date"],
                    "Thread": "http://forums.sirlingames.com/t/{}".format(
                        result["thread_id"]
                    ),
                }
                for result in results
                if (result['player1'], result['player2'], result['round']) in pending_matches
            ),
            {},
        )
    )


@challonge.command("display-expired-matches")
@click.option("--domain", "domains", multiple=True)
@click.pass_context
def display_expired_matches(ctx, domains):
    tournaments = {
        tournament.data["id"]: tournament for tournament in all_tournaments(domains)
    }
    pending_matches = {
        (match.player1.data["display_name"], match.player2.data["display_name"], match.data["round"])
        for tournament in tournaments.values()
        for match in tournament.pending_matches
    }
    results = db.query(
        """
            SELECT *
            FROM matches
            JOIN rounds
            ON matches.tournament = rounds.tournament
            AND matches.round = CASE WHEN rounds.round > 0 THEN rounds.round ELSE (-rounds.round || 'L') END
            WHERE winner IS NULL
            AND player1 IS NOT NULL
            AND player2 IS NOT NULL
            AND due_date < :today
            AND rounds.tournament IN ({tournaments})
        """.format(
            tournaments=",".join(repr(id) for id in tournaments.keys())
        ),
        today=date.today(),
    )
    results = list(results)
    print(
        tabulate(
            (
                {
                    "Tournament": tournaments[result["tournament"]].data["name"],
                    "Player 1": result["player1"],
                    "Player 2": result["player2"],
                    "Round": result["round"],
                    "Due Date": result["due_date"],
                    "Thread": "http://forums.sirlingames.com/t/{}".format(
                        result["thread_id"]
                    ),
                }
                for result in results
                if (result['player1'], result['player2'], result['round']) in pending_matches
            ),
            {},
        )
    )


@challonge.command("send-pending-matches")
@click.option("--domain", "domains", multiple=True)
@click.pass_context
def send_pending_matches(ctx, domains):
    for tournament in all_tournaments(domains):
        template = tournament.template
        pending_matches = (
            (
                tournament.data["id"],
                tournament.data["name"],
                match.data["id"],
                match.data["round"],
                match.player1.data["display_name"],
                match.player2.data["display_name"],
                tournament.co_tos,
            )
            for match in tournament.pending_matches
        )
        send_messages(ctx.obj["forum"], pending_matches, template)


@challonge.command("send-round-matches")
@click.argument("tournament_matcher")
@click.argument("round", type=int)
@click.option("--domain", "domains", multiple=True)
@click.pass_context
def send_round_matches(ctx, tournament_matcher, round, domains):
    for tournament in all_tournaments(domains):
        if fnmatch(tournament.data["name"], tournament_matcher):
            template = tournament.template
            pending_matches = (
                (
                    tournament.data["id"],
                    tournament.data["name"],
                    match.data["id"],
                    match.data["round"],
                    match.player1.data["display_name"],
                    match.player2.data["display_name"],
                    tournament.co_tos,
                )
                for match in tournament.matches_for_round(round)
                if match.data["state"] != "complete"
            )
            send_messages(ctx.obj["forum"], pending_matches, template)


@challonge.command("add-to")
@click.argument("tournament_matcher")
@click.argument("to")
@click.option("--domain", "domains", multiple=True)
@click.pass_context
def challonge_add_to(ctx, tournament_matcher, to, domains):
    for tournament in all_tournaments(domains):
        if fnmatch(tournament.data["name"], tournament_matcher):
            ctx.invoke(add_to, slug=tournament.slug, to=to)


@autoto.group()
def ranked():
    pass


@ranked.command("add")
@click.argument("ranked_id")
@click.argument("user")
@click.argument("week", type=int)
def add_participant(ranked_id, user, week):
    participants = db["ranked_standings"]
    participants.insert(
        {
            "username": user,
            "tournament_round": 0,
            "stars": 0,
            "games_played": 0,
            "week": week,
            "active": True,
            "tournament": ranked_id,
        }
    )


@ranked.command("record-win")
@click.argument("ranked_id")
@click.argument("winner")
@click.argument("loser")
@click.argument("week", type=int)
def record_win(ranked_id, winner, loser, week):
    matches = db["matches"]
    standings = db["ranked_standings"]

    match = matches.find_one(
        tournament=ranked_id, round=week, player1=winner, player2=loser
    ) or matches.find_one(
        tournament=ranked_id, round=week, player1=loser, player2=winner
    )

    if match is None:
        click.echo("Match not found", err=True)
        return 1

    for player in (winner, loser):
        existing_standings = standings.find_one(
            username=player, week=week + 1, tournament=ranked_id
        )
        if existing_standings and existing_standings["tournament_round"]:
            click.echo(f"{player} already has recorded standings for week {week}")
            return 1

    with db as tx:
        match["winner"] = winner
        tx["matches"].update(match, ["id"])

        winner_prior = tx["ranked_standings"].find_one(
            username=winner, week=week, tournament=ranked_id
        )
        loser_prior = tx["ranked_standings"].find_one(
            username=loser, week=week, tournament=ranked_id
        )

        max_prior_round = max(
            winner_prior["tournament_round"] or 0, loser_prior["tournament_round"] or 0
        )

        winner_next = {
            "username": winner,
            "stars": winner_prior["stars"] + max_prior_round + 1,
            "games_played": winner_prior["games_played"] + 1,
            "week": week + 1,
            "tournament_round": (max_prior_round + 1) % 3,
            "tournament": ranked_id,
        }
        tx["ranked_standings"].upsert(winner_next, ["username", "week", "tournament"])

        loser_next = {
            "username": loser,
            "stars": max((loser_prior["stars"] or 0) - 1, 0),
            "games_played": (loser_prior["games_played"] or 0) + 1,
            "week": week + 1,
            "tournament_round": 0,
            "tournament": ranked_id,
        }
        tx["ranked_standings"].upsert(loser_next, ["username", "week", "tournament"])


@ranked.command("mark-inactive")
@click.argument("ranked_id")
@click.argument("player")
@click.argument("week", type=int)
def mark_inactive(ranked_id, player, week):
    db["ranked_standings"].upsert(
        {"username": player, "week": week, "active": False, "tournament": ranked_id},
        ["username", "week", "tournament"],
    )


@ranked.command("mark-active")
@click.argument("ranked_id")
@click.argument("player")
@click.argument("week", type=int)
def mark_active(ranked_id, player, week):
    db["ranked_standings"].upsert(
        {"username": player, "week": week, "active": True, "tournament": ranked_id},
        ["username", "week"],
    )


STARS_PER_LEVEL = 5


class League(Enum):
    Bronze = 0
    SuperBronze = 1
    Silver = 2
    SuperSilver = 3
    Gold = 4

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    @classmethod
    def from_stars(cls, stars):
        return cls(min(stars // STARS_PER_LEVEL, max(cls).value))

    def display_name(self):
        return {"SuperBronze": "Super Bronze", "SuperSilver": "Super Silver"}.get(
            self.name, self.name
        )


def match_cost(round, player1, player2):
    win_cost = (
        abs((player1["tournament_round"] or 1) - (player2["tournament_round"] or 1))
        * 50
    )
    league1 = League.from_stars(player1["stars"] or 0)
    league2 = League.from_stars(player2["stars"] or 0)
    league_cost = max(abs(league1.value - league2.value) - 1, 0) * 10

    matches = db["matches"]
    previously_played = list(
        matches.find(
            tournament=player1["tournament"],
            player1=player1["username"],
            player2=player2["username"],
        )
    ) + list(
        matches.find(
            tournament=player1["tournament"],
            player1=player2["username"],
            player2=player1["username"],
        )
    )
    if previously_played:
        last_played = max(match["round"] for match in previously_played)
        repeat_cost = 50 * pow(0.5, round - last_played - 1)
    else:
        repeat_cost = 0

    return league_cost + win_cost + repeat_cost


def greedy_matches(round, participants):
    rand = Random(0)
    participants = list(participants)
    rand.shuffle(participants)
    scheduling_priority = sorted(participants, key=lambda p: p["games_played"] or 0)

    while len(participants) > 1:
        player1 = scheduling_priority[0]
        player2 = min(
            [p for p in participants if p != player1],
            key=lambda x: match_cost(round, player1, x),
        )

        yield player1, player2
        participants.remove(player1)
        participants.remove(player2)
        scheduling_priority.remove(player1)
        scheduling_priority.remove(player2)


def ranked_match_id(round, player1, player2):
    return f"ranked-{round}-{player1}-{player2}"


@ranked.command("send-matches")
@click.argument("ranked_id")
@click.argument("week", type=int)
@click.pass_context
def send_ranked_matches(ctx, ranked_id, week):
    standings = db["ranked_standings"]
    matches = db["matches"]

    ps = list(
        standings.find(
            week=week,
            active=True,
            tournament=ranked_id,
            order_by=["-stars", "username"],
        )
    )

    already_scheduled_matches = matches.find(round=week, tournament=ranked_id)
    already_scheduled_players = sum(
        [[match["player1"], match["player2"]] for match in already_scheduled_matches],
        [],
    )
    ps = [
        player for player in ps if player["username"] not in already_scheduled_players
    ]

    matches = list(greedy_matches(week, ps))

    for p1, p2 in matches:
        print(f'{p1["username"]}/{p2["username"]}')

    template = get_template(ranked_id, "Forums Quick Matches")
    send_messages(
        ctx.obj["forum"],
        (
            (
                ranked_id,
                "Ranked",
                ranked_match_id(week, player1["username"], player2["username"]),
                week,
                player1["username"],
                player2["username"],
                [],
            )
            for (player1, player2) in matches
        ),
        template,
    )


@ranked.command("create-match")
@click.argument("ranked_id")
@click.argument("player1")
@click.argument("player2")
@click.argument("week", type=int)
@click.pass_context
def create_ranked_match(ctx, ranked_id, player1, player2, week):
    template = get_template("ranked", "Forums Quick Matches")
    send_messages(
        ctx.obj["forum"],
        [
            (
                ranked_id,
                "Ranked",
                ranked_match_id(week, player1, player2),
                week,
                player1,
                player2,
                [],
            )
        ],
        template,
    )


class Box:
    def __init__(self, data):
        if isinstance(data, str):
            data = data.split("\n")
        self.data = data

    @property
    def width(self):
        return max((len(row) for row in self.data))

    @property
    def height(self):
        return len(self.data)

    @classmethod
    def empty(cls, width, height):
        return cls([" " * width] * height)

    @classmethod
    def vertical(cls, *boxes):
        return Box(sum((box.data for box in boxes), []))

    def left_of(self, other):
        if self.height != other.height:
            raise ValueError(
                textwrap.dedent(
                    """\
                Can't place boxes with mismatched heights next to each other:
                Height: {self.height}
                {self}
                Height: {other.height}
                {other}
            """
                ).format(self=self, other=other)
            )

        pad_to = max(len(row) for row in self.data)
        return Box(
            [
                f"{{:{pad_to}}}{{}}".format(left, right)
                for (left, right) in zip(self.data, other.data)
            ]
        )

    def __str__(self):
        return "\n".join(row.rstrip() for row in self.data)


class SingletonBracket:
    def __init__(self, name):
        self.name = name

    @property
    def exit_row(self):
        return 0

    @property
    def height(self):
        return 1

    @property
    def names(self):
        return [self.name]

    def render(self, name_width):
        return self.name

    def __str__(self):
        return self.name + " -"


class Bracket:
    def __init__(self, top, bottom, winner=None):
        if isinstance(top, Bracket):
            self.top = top
        else:
            self.top = SingletonBracket(top)

        if isinstance(bottom, Bracket):
            self.bottom = bottom
        else:
            self.bottom = SingletonBracket(bottom)

        self.winner = winner

        self.upper_corner = "\N{BOX DRAWINGS LIGHT DOWN AND LEFT}"
        self.upper_v = "\N{BOX DRAWINGS LIGHT VERTICAL}"
        self.upper_h = "\N{BOX DRAWINGS LIGHT HORIZONTAL}"
        self.lower_corner = "\N{BOX DRAWINGS LIGHT UP AND LEFT}"
        self.lower_v = "\N{BOX DRAWINGS LIGHT VERTICAL}"
        self.lower_h = "\N{BOX DRAWINGS LIGHT HORIZONTAL}"
        self.join = "\N{BOX DRAWINGS LIGHT VERTICAL AND RIGHT}"
        self.center_h = "\N{BOX DRAWINGS LIGHT HORIZONTAL}"

        if winner == getattr(top, "winner", top):
            self.upper_corner = "\N{BOX DRAWINGS HEAVY DOWN AND LEFT}"
            self.upper_v = "\N{BOX DRAWINGS HEAVY VERTICAL}"
            self.upper_h = "\N{BOX DRAWINGS HEAVY HORIZONTAL}"
            self.join = "\N{BOX DRAWINGS DOWN LIGHT AND RIGHT UP HEAVY}"
            self.center_h = "\N{BOX DRAWINGS HEAVY HORIZONTAL}"
        elif winner == getattr(bottom, "winner", bottom):
            self.lower_corner = "\N{BOX DRAWINGS HEAVY UP AND LEFT}"
            self.lower_v = "\N{BOX DRAWINGS HEAVY VERTICAL}"
            self.lower_h = "\N{BOX DRAWINGS HEAVY HORIZONTAL}"
            self.join = "\N{BOX DRAWINGS UP LIGHT AND RIGHT DOWN HEAVY}"
            self.center_h = "\N{BOX DRAWINGS HEAVY HORIZONTAL}"

    @property
    def names(self):
        return self.top.names + self.bottom.names

    @property
    def exit_row(self):
        return self.top.height

    @property
    def height(self):
        return self.top.height + self.bottom.height + 1

    def top_box(self, name_width):
        return Box(self.top.render(name_width))

    def bottom_box(self, name_width):
        return Box(self.bottom.render(name_width))

    def render(self, name_width):
        top_box = self.top_box(name_width)
        bottom_box = self.bottom_box(name_width)

        max_width = max(top_box.width, bottom_box.width, name_width)
        top_padding = Box(
            [""] * (self.top.exit_row)
            + [" " + self.upper_h * (max_width - top_box.width)]
            + [""] * (top_box.height - 1 - self.top.exit_row)
        )

        bottom_padding = Box(
            [""] * (self.bottom.exit_row)
            + [" " + self.lower_h * (max_width - bottom_box.width)]
            + [""] * (bottom_box.height - 1 - self.bottom.exit_row)
        )

        border = Box.empty(max_width, 1)

        top_bar_length = top_box.height - 1 - self.top.exit_row
        bottom_bar_length = self.bottom.exit_row

        winner = self.winner or ""

        bar = Box(
            [""] * (top_box.height - 1 - top_bar_length)
            + [f"{self.upper_h}{self.upper_corner}"]
            + [f" {self.upper_v}"] * top_bar_length
            + [f" {self.join}{self.center_h} {winner}"]
            + [f" {self.lower_v}"] * bottom_bar_length
            + [f"{self.lower_h}{self.lower_corner}"]
            + [""] * (bottom_box.height - 1 - bottom_bar_length)
        )

        return str(
            Box.vertical(
                top_box.left_of(top_padding), border, bottom_box.left_of(bottom_padding)
            ).left_of(bar)
        )

    def __str__(self):
        return self.render(max(len(name) for name in self.names))


def bracket_history(match, ranked_id):
    standings = db["ranked_standings"]
    matches = db["matches"]

    p1 = match["player1"]
    p2 = match["player2"]

    p1_standings = standings.find_one(
        username=p1, week=match["round"], tournament=ranked_id
    )
    p2_standings = standings.find_one(
        username=p2, week=match["round"], tournament=ranked_id
    )

    if p1_standings is None or p1_standings["tournament_round"] == 0:
        p1_bracket = p1
    else:
        old_matches = sorted(
            itertools.chain(
                matches.find(player1=p1, tournament=ranked_id),
                matches.find(player2=p1, tournament=ranked_id),
            ),
            key=lambda r: r["round"],
            reverse=True,
        )
        previous_matches = [
            old_match
            for old_match in old_matches
            if old_match["round"] < match["round"]
        ]
        if previous_matches:
            previous_match = previous_matches[0]

            if previous_match["winner"] == p1:
                p1_bracket = bracket_history(previous_match, ranked_id)
            else:
                p1_bracket = p1
        else:
            p1_bracket = p1

    if p2_standings is None or p2_standings["tournament_round"] == 0:
        p2_bracket = p2
    else:
        old_matches = sorted(
            itertools.chain(
                matches.find(player1=p2, tournament=ranked_id),
                matches.find(player2=p2, tournament=ranked_id),
            ),
            key=lambda r: r["round"],
            reverse=True,
        )
        previous_matches = [
            old_match
            for old_match in old_matches
            if old_match["round"] < match["round"]
        ]

        if previous_matches:
            previous_match = previous_matches[0]

            if previous_match["winner"] == p2:
                p2_bracket = bracket_history(previous_match, ranked_id)
            else:
                p2_bracket = p2
        else:
            p2_bracket = p2

    return Bracket(p1_bracket, p2_bracket, match["winner"])


@ranked.command("finalize-all")
@click.pass_context
@click.argument("ranked_ids", nargs=-1)
def finalize_all(ctx, ranked_ids):
    if not ranked_ids:
        ranked_ids = ["ranked", "ranked-ps"]

    matches = db["matches"]

    for ranked_id in ranked_ids:
        latest_match = matches.find_one(tournament=ranked_id, order_by="-round")
        latest_week = latest_match["round"]
        ctx.invoke(finalize_week, ranked_id=ranked_id, week=latest_week)


@ranked.command("finalize-most-recent")
@click.pass_context
@click.argument("ranked_id")
def finalize_most_recent(ctx, ranked_id):
    most_recent_week = list(
        db.query(
            "SELECT MAX(round) AS week FROM matches WHERE tournament = :id",
            id=ranked_id,
        )
    )[0]["week"]
    rounds = db["rounds"]
    current_round = rounds.find_one(tournament=ranked_id, round=most_recent_week)
    current_due_date = current_round["due_date"]
    if datetime.utcnow() > current_due_date:
        ctx.invoke(finalize_week, ranked_id=ranked_id, week=most_recent_week)


@ranked.command("finalize-week")
@click.pass_context
@click.argument("ranked_id")
@click.argument("week", type=int)
def finalize_week(ctx, ranked_id, week):
    standings = db["ranked_standings"]
    matches = db["matches"]

    pending_matches = matches.find(round=week, winner=None, tournament=ranked_id)
    for match in pending_matches:
        while True:
            winner = click.prompt(
                "{}/{} winner [1/2/N]".format(match["player1"], match["player2"]),
                default="N",
                show_default=False,
            )
            if winner == "1":
                ctx.invoke(
                    record_win,
                    ranked_id=ranked_id,
                    week=week,
                    winner=match["player1"],
                    loser=match["player2"],
                )
            elif winner == "2":
                ctx.invoke(
                    record_win,
                    ranked_id=ranked_id,
                    week=week,
                    winner=match["player2"],
                    loser=match["player1"],
                )
            elif winner.lower() == "n":
                pass
            else:
                continue
            break

    to_finalize = standings.find(week=week, tournament=ranked_id)
    for current_standings in to_finalize:
        print(f"Week {week}", current_standings)
        future_standings = (
            standings.find_one(
                week=week + 1,
                username=current_standings["username"],
                tournament=ranked_id,
            )
            or {}
        )
        print(f"Week {week+1}", future_standings)
        for field in current_standings:
            if future_standings.get(field) is None:
                future_standings[field] = current_standings[field]
        future_standings.pop("id", None)
        future_standings["week"] = week + 1
        print("Finalized to", future_standings)
        standings.upsert(
            future_standings, keys=["id", "username", "week", "tournament"]
        )

    ctx.invoke(send_ranked_matches, ranked_id=ranked_id, week=week + 1)
    ctx.invoke(post_week_summary, ranked_id=ranked_id, week=week + 1)


@ranked.command("sub")
@click.argument("ranked_id")
@click.argument("player")
@click.argument("sub")
@click.argument("week", type=int)
def sub_player(ranked_id, player, sub, week):
    matches = db["matches"]
    existing_sub_match = matches.find_one(
        round=week, player1=sub, tournament=ranked_id
    ) or matches.find_one(round=week, player2=sub, tournament=ranked_id)

    if existing_sub_match:
        print(f"Sub {sub} is already scheduled for a match in week {week}")
        return

    sub_match = matches.find_one(
        round=week, player1=player, tournament=ranked_id
    ) or matches.find_one(round=week, player2=player, tournament=ranked_id)
    if sub_match["winner"]:
        print("Match being subbed has already been played")
        return

    if sub_match["player1"] == player:
        sub_match["player1"] = sub
    else:
        sub_match["player2"] = sub
    matches.update(sub_match, keys=["id"])


@ranked.command("post-summary")
@click.argument("ranked_id")
@click.pass_context
@click.argument("week", type=int)
def post_week_summary(ctx, ranked_id, week):
    standings = db["ranked_standings"]

    ps = list(
        standings.find(
            week=week, tournament=ranked_id, order_by=["-stars", "-active", "username"]
        )
    )

    post = []

    def append_player_rankings(ps):
        for league, players in itertools.groupby(
            ps, lambda p: League.from_stars(p["stars"] or 0)
        ):
            post.append(
                f"## {league.name} League ({league.value*STARS_PER_LEVEL}+ :star:)"
            )
            post.append("| Player | Points")
            post.append("|---|:---:|")
            for player in players:
                post.append(
                    "| {username} | {stars} |".format(
                        username=player["username"],
                        stars=":star:"
                        * ((player["stars"] or 0) - league.value * STARS_PER_LEVEL),
                        inactive=":knockdown:" if not player["active"] else ":psfist:",
                    )
                )

    post.append(f"# Week {week} standings")
    post.append("# :psfist: Active Roster")
    append_player_rankings([p for p in ps if p["active"]])
    post.append("---")
    post.append("# :knockdown: Inactive Roster")
    post.append('[details="Full List"]')
    append_player_rankings([p for p in ps if not p["active"]])
    post.append("[/details]")
    post.append("---")
    post.append(f"# Week {week} matches")

    matches = db["matches"].find(tournament=ranked_id, round=week)
    for match in matches:
        post.append('[details="{player1}/{player2}"]'.format(**match))
        post.append("<pre>")
        post.append(str(bracket_history(match, ranked_id)))
        post.append("</pre>")
        post.append("[/details]")

    print("\n".join(post))


def scheduled_times(posts):
    for post in posts:
        match = re.search(
            "(?:<p>|<br>)\s*AutoTO: Schedule @ ([^<]*)(?:</p>)?",
            post["cooked"],
            flags=re.RegexFlag.IGNORECASE,
        )

        if match is None:
            continue

        scheduled_time = match.group(1)
        yield (post["updated_at"], scheduled_time, dateparser.parse(scheduled_time))


def prompt_event(title, string, date, url):
    while True:
        print(date, repr(date.tzinfo))
        action = click.prompt(
            textwrap.dedent(
                f"""\
                Scheduling "{title}" at {date} from "{string}"
                Originally from {url}

                y(es)/n(o)/e(dit)"""
            ),
            default="N",
        )
        if action[0].lower() == "e":
            title = click.edit(title, require_save=False).strip()
            date = dateparser.parse(click.edit(string, require_save=False))
        elif action[0].lower() == "y":
            return title, date
        elif action[0].lower() == "n":
            return None, None


LAST_PRIVATE_MESSAGE = "last_private_message"

AUTOTO_COMMANDS = []


def autoto_command(command, private=None, public=None):
    def wrapper(fn):
        AUTOTO_COMMANDS.append(
            {"command": command, "private": private, "public": public, "callback": fn}
        )
        return fn

    return wrapper


@autoto_command(r"Schedule @ (?P<time>[^<]*)", private=".*")
def scheduled_at(ctx, context_message, matches):
    scheduled_matches = db["scheduled_matches"]

    matches = sorted(matches, key=lambda m: m["post"]["updated_at"], reverse=True)
    if matches:
        updated_at = matches[0]["post"]["updated_at"]
        scheduled_string = matches[0]["time"]
        scheduled_date = None

        while scheduled_date is None:
            try:
                scheduled_date = dateparser.parse(scheduled_string)
                if scheduled_date is None:
                    logger.exception("Unable to parse datetime, please edit")
                    scheduled_string = click.edit(scheduled_string, require_save=False)
            except ValueError:
                logger.exception("Unable to parse datetime, please edit")
                scheduled_string = click.edit(scheduled_string, require_save=False)

        current_event = scheduled_matches.find_one(message_id=context_message["id"])
        if current_event is None:
            title, date = prompt_event(
                context_message["title"],
                scheduled_string,
                scheduled_date,
                ctx.obj["forum"].url(f"t/{context_message['id']}"),
            )
            if title == None:
                return
            scheduled_event = Calendar(
                "5qcrghv93ken5kco8e0eeqo3do@group.calendar.google.com"
            ).insert_event(title, date)
            scheduled_matches.insert(
                {
                    "message_id": context_message["id"],
                    "event_id": scheduled_event,
                    "updated_at": updated_at,
                }
            )
        elif current_event["updated_at"] < updated_at:
            title, date = prompt_event(
                context_message["title"],
                scheduled_string,
                scheduled_date,
                ctx.obj["forum"].url(f"t/{context_message['id']}"),
            )
            if title == None:
                return
            scheduled_event = Calendar(
                "5qcrghv93ken5kco8e0eeqo3do@group.calendar.google.com"
            ).update_event(current_event["event_id"], title, date)
            scheduled_matches.update(
                {
                    "message_id": context_message["id"],
                    "event_id": scheduled_event,
                    "updated_at": updated_at,
                },
                keys=["message_id"],
            )


@autoto_command(
    r"Week (?P<week>\d+): @(?P<left>.*) (?P<direction>[<>]) @(?P<right>[^<]*?)",
    public="^Forum Quick Matches",
)
def assign_ranked_winner(ctx, context_message, matches):
    for match in matches:
        if match["direction"] == ">":
            winner = match["left"]
            loser = match["right"]
        else:
            winner = match["right"]
            loser = match["left"]

        ctx.invoke(record_win, winner, loser, int(match["week"]))


def merge_by_date(*message_iters):
    nexts = []

    for message_iter in message_iters:
        try:
            nexts.append((next(message_iter), message_iter))
        except StopIteration:
            pass

    while nexts:
        next_message = max(
            nexts, key=lambda next: dateparser.parse(next[0]["bumped_at"])
        )
        message, message_iter = next_message
        nexts.remove(next_message)
        yield message
        try:
            nexts.append((next(message_iter), message_iter))
        except StopIteration:
            pass


@autoto.command()
@click.option("--since", type=lambda v: dateparser.parse(v).astimezone(), default=None)
@click.pass_context
def process_autoto(ctx, since):
    dates = db["dates"]

    most_recently_processed = (
        since
        or (dates.find_one(key=LAST_PRIVATE_MESSAGE) or {})
        .get("date", datetime(2000, 1, 1, tzinfo=pytz.timezone("UTC")))
        .astimezone()
    )

    logger.debug("Most Recently Processed: %s", most_recently_processed)

    last_analyzed = most_recently_processed

    private = ctx.obj["forum"].private_messages()
    private_archived = ctx.obj["forum"].private_messages(archived=True)
    public = ctx.obj["forum"].public_threads()

    for message in merge_by_date(private, private_archived, public):
        message_date = dateparser.parse(message["bumped_at"]).astimezone()
        logger.debug("Processing message %s from %s", message["id"], message_date)
        if message_date < most_recently_processed:
            logger.debug(
                "Found a message %s older than %s, finished",
                message["id"],
                most_recently_processed,
            )
            break

        logger.debug(
            "Updating last_analyzed from %s to %s",
            last_analyzed,
            max(last_analyzed, message_date),
        )
        last_analyzed = max(last_analyzed, message_date)

        posts = [post for post in ctx.obj["forum"].message_posts(message["id"])]
        for command in AUTOTO_COMMANDS:
            matching_posts = [
                {"post": post, **match.groupdict()}
                for (match, post) in (
                    (
                        re.search(
                            fr'(?:<p>|<br>)\s*AutoTO: {command["command"]}(?:</p>)?',
                            post["cooked"],
                            flags=re.RegexFlag.IGNORECASE,
                        ),
                        post,
                    )
                    for post in posts
                )
                if match is not None
            ]
            command["callback"](ctx, message, matching_posts)

    logger.debug("Last analyzed was %s", last_analyzed)
    dates.upsert({"key": LAST_PRIVATE_MESSAGE, "date": last_analyzed}, keys=["key"])


@autoto.command("edit-template")
@click.argument("slug")
def edit_template(slug):
    existing = templates.find_one(slug=slug)
    existing["title"] = click.edit(existing["title"], require_save=False)
    existing["body"] = click.edit(existing["body"], require_save=False)
    templates.update(existing, keys=["tournament"])


@autoto.command()
@click.option(
    "--challonge-username",
    prompt="Challonge Username",
    envvar="AUTOTO_CHALLONGE_USERNAME",
)
@click.password_option(
    "--challonge-api-key",
    confirmation_prompt=False,
    prompt="Challonge API Key",
    envvar="AUTOTO_CHALLONGE_API_KEY",
)
@click.option("--since", type=lambda v: dateparser.parse(v).astimezone(), default=None)
@click.option("--domain", "domains", multiple=True)
@click.pass_context
def daily(ctx, challonge_username, challonge_api_key, since, domains):
    logger.info("Checking pending Challonge matches")
    ctx.invoke(
        challonge,
        challonge_username=challonge_username,
        challonge_api_key=challonge_api_key,
    )
    logger.info("Sending pending matches")
    ctx.invoke(send_pending_matches, domains=domains)
    logger.info("Parsing forum messages")
    ctx.invoke(process_autoto, since=since)
    logger.info("Soon to expire matches")
    ctx.invoke(prompt_expiring_matches, domains=domains)
    logger.info("Expired matches")
    ctx.invoke(display_expired_matches, domains=domains)
    # logger.info("Finalize ranked matches")
    # ctx.invoke(finalize_most_recent, ranked_id="ranked")


@autoto.command()
@click.argument("slug")
@click.argument("to")
def add_to(slug, to):
    tournament = tournaments.find_one(slug=slug) or {}
    tournament.setdefault("slug", slug)
    if tournament.get("tos"):
        tournament["tos"] = ",".join(set(tournament["tos"].split(",")) | {to})
    else:
        tournament["tos"] = to
    del tournament["id"]

    tournaments.upsert(tournament, keys=["slug"])


if __name__ == "__main__":
    sys.exit(autoto())
