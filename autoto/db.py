import click
import dataset
import textwrap


db = dataset.connect("sqlite:///autoto.db")

templates = db["templates"]
tournaments = db["tournaments"]


MARKER = "## TEXT BELOW THIS IS IGNORED"

VARIABLES = textwrap.dedent(
    """\
    The following variables are available:
        ${name}: The name of the tournament
        ${round}: The current tournament round
        ${player1}: The name of player 1
        ${player2}: The name of player 2
        ${due}: The duedate of the match
"""
)

TITLE_TEMPLATE = "${name} - R${round} - ${player1}/${player2}"

BODY_TEMPLATE = textwrap.dedent("""\
    Hello, ${player1} and ${player2},

    You have until **${due.strftime('%A, %B %d')}** @ 11:59pm GMT to complete your match.

    If you are posting to the scheduling conversation first, provide a list of
    dates and times when you are available to play. If you are posting to the
    scheduling conversation second, pick a specific start time and date for your
    match from those provided by your opponent. If none work for you, propose
    specific alternatives.

    Use the Forums Insert Date/Time feature (![image|68x45](upload://yGvbZSqI4EQfqonbqINkm33LdLS.png))
    to add localized date and time ranges to coordinate your match across different timezones.

    Once you have found a time to play, you can post `AutoTO: Schedule @ MM/DD/YYYY HH:MM AM/PM TZ`
    for example, `AutoTO: Schedule @ 4/1/2018 10:30 AM EST`), or use `AutoTO: Schedule @` and
    then the Insert Date/Time feature, and AutoTO will post it to the [Yomi Tournament Calendar](https://bit.ly/iyl-calendar).

    Reporting: Either the victor or the vanquished (or both) should post results in
    [this thread](${get_tournament_metadata('tournament-thread', tournament=name)}).

    Best of luck!
""")


def default_title(tournament_name):
    return "\n".join(
        [TITLE_TEMPLATE, MARKER, f"Title for {tournament_name}", VARIABLES]
    )


def default_body(tournament_name):
    return "\n".join([BODY_TEMPLATE, MARKER, f"Body for {tournament_name}", VARIABLES])


def remove_marker(template):
    return template.split(MARKER)[0]


def get_template(id, name):
    template = templates.find_one(tournament=id)

    if not template:
        template = {
            "tournament": id,
            "slug": click.prompt(f"Template slug (for {name})"),
            "title": remove_marker(
                click.edit(text=default_title(name), require_save=False)
            ),
            "body": remove_marker(
                click.edit(text=default_body(name), require_save=False)
            ),
        }
        templates.insert(template)

    return template
