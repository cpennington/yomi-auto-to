import requests
import time

class Forum:
    def __init__(self, baseurl, username, password):
        self.baseurl = baseurl
        self.session = requests.session()
        self.session.headers['Accept'] = 'application/json'
        csrf_resp = self.session.get(self.url('session/csrf'), data={'_': time.time()*1000})
        csrf_resp.raise_for_status()

        self.session.headers['X-CSRF-Token'] = csrf_resp.json()['csrf']

        login_resp = self.session.post(self.url('session'), data=dict(login=username, password=password))
        login_resp.raise_for_status()

    def url(self, endpoint):
        return f"{self.baseurl}/{endpoint}"

    def private_message(self, recipients, title, body):
        data = {
            'raw': body,
            'title': title,
            'unlist_topic': 'false',
            'category': '',
            'is_warning': 'false',
            'archetype': 'private_message',
            'target_usernames': ",".join(recipients),
            'typing_duration_msecs': 0,
            'composer_open_duration_msecs': 0,
            'featured_link': '',
            'nested_post': 'true',
        }
        return self.session.post(self.url('posts'), data=data)

    def search_user(self, name):
        return self.session.get(
            self.url('/u/search/users'),
            data={
                'term': name,
                'include_groups': 'false',
                'include_mentionable_groups': 'false',
                'include_messageable_groups': 'true',
                'topic_allowed_users': 'false',
            }
        )
