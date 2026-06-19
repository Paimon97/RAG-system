import requests


class WikiLoader:

    def load(self, url: str):

        response = requests.get(
            url,
            timeout=30
        )

        response.raise_for_status()

        return response.text