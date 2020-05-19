import httplib2
import os
from collections import namedtuple

from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage

import datetime


# If modifying these scopes, delete your previously saved credentials
# at ~/.credentials/autoto.json

SCOPES = "https://www.googleapis.com/auth/calendar"
CLIENT_SECRET_FILE = "client_secret.json"
APPLICATION_NAME = "AutoTO"

# Flags:
#   --auth_host_name AUTH_HOST_NAME
#                         Hostname when running a local web server.
#   --noauth_local_webserver
#                         Do not run a local web server.
#   --auth_host_port[AUTH_HOST_PORT[AUTH_HOST_PORT ...]]
#                         Port web server should listen on.
#   --logging_level {DEBUG, INFO, WARNING, ERROR, CRITICAL}
#                         Set the logging level of detail.


class MemoryCache:
    _CACHE = {}

    def get(self, url):
        return MemoryCache._CACHE.get(url)

    def set(self, url, content):
        MemoryCache._CACHE[url] = content


class Flags:
    def __init__(
        self, auth_host_name, noauth_local_webserver, auth_host_port, logging_level
    ):
        self.auth_host_name = auth_host_name
        self.noauth_local_webserver = noauth_local_webserver
        self.auth_host_port = auth_host_port
        self.logging_level = logging_level


class Calendar:
    def __init__(self, calendar_id, auth_flags=None):
        self.calendar_id = calendar_id
        self.auth_flags = auth_flags or Flags(None, None, [], "WARNING")

    def get_credentials(self):
        """Gets valid user credentials from storage.

        If nothing has been stored, or if the stored credentials are invalid,
        the OAuth2 flow is completed to obtain the new credentials.

        Returns:
            Credentials, the obtained credential.
        """
        home_dir = os.path.expanduser("~")
        credential_dir = os.path.join(home_dir, ".credentials")
        if not os.path.exists(credential_dir):
            os.makedirs(credential_dir)
        credential_path = os.path.join(credential_dir, "autoto.json")

        store = Storage(credential_path)
        credentials = store.get()
        if not credentials or credentials.invalid:
            flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
            flow.user_agent = APPLICATION_NAME
            credentials = tools.run_flow(flow, store, self.auth_flags)
            print("Storing credentials to " + credential_path)
        return credentials

    def event_body(self, title, date):
        body = {
            "summary": title,
            "start": {"dateTime": date.strftime("%Y-%m-%dT%H:%M:00%z")},
            "end": {
                "dateTime": (date + datetime.timedelta(hours=1)).strftime(
                    "%Y-%m-%dT%H:%M:00%z"
                )
            },
        }
        return body

    @property
    def service(self):
        credentials = self.get_credentials()
        http = credentials.authorize(httplib2.Http())
        return discovery.build("calendar", "v3", http=http, cache=MemoryCache())

    def insert_event(self, title, date):
        result = (
            self.service.events()
            .insert(calendarId=self.calendar_id, body=self.event_body(title, date))
            .execute()
        )
        return result["id"], result["htmlLink"]

    def update_event(self, event_id, title, date):
        result = (
            self.service.events()
            .patch(
                calendarId=self.calendar_id,
                eventId=event_id,
                body=self.event_body(title, date),
            )
            .execute()
        )
        return result["id"], result["htmlLink"]
