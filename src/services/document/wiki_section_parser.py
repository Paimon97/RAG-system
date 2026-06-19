from bs4 import BeautifulSoup


class WikiSectionParser:
    """
    Парсит MediaWiki страницу и возвращает список секций:

    [
        {
            "h1": "...",
            "h2": "...",
            "h3": "...",
            "text": "..."
        }
    ]
    """

    def extract_sections(self, html: str) -> list[dict]:

        soup = BeautifulSoup(html, "html.parser")

        # Основной контент MediaWiki
        content = (
            soup.find(id="mw-content-text")
            or soup.find(class_="mw-parser-output")
            or soup
        )

        page_title = None

        h1 = soup.find("h1")

        if h1:
            page_title = h1.get_text(" ", strip=True)

        sections = []

        current_h2 = None
        current_h3 = None

        current_content = []

        def flush_section():

            nonlocal current_content
            nonlocal current_h2
            nonlocal current_h3

            text = "\n".join(current_content).strip()

            if not text:
                return

            sections.append(
                {
                    "h1": page_title,
                    "h2": current_h2,
                    "h3": current_h3,
                    "text": text,
                }
            )

            current_content = []

        for tag in content.find_all(
            [
                "h2",
                "h3",
                "p",
                "ul",
                "ol",
                "table"
            ]
        ):

            # ---------- H2 ----------

            if tag.name == "h2":

                flush_section()

                current_h2 = tag.get_text(
                    " ",
                    strip=True
                )

                current_h3 = None

                continue

            # ---------- H3 ----------

            if tag.name == "h3":

                flush_section()

                current_h3 = tag.get_text(
                    " ",
                    strip=True
                )

                continue

            # ---------- TABLE ----------

            if tag.name == "table":

                table_text = tag.get_text(
                    " ",
                    strip=True
                )

                if table_text:
                    current_content.append(
                        table_text
                    )

                continue

            # ---------- P / UL / OL ----------

            text = tag.get_text(
                " ",
                strip=True
            )

            if text:

                current_content.append(
                    text
                )

        flush_section()

        return sections