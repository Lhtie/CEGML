"""Bounded string scanning for model response tags (including truncations)."""


def last_tag(response, tag):
    if response is None:
        return None

    opening, closing = f"<{tag}>", f"</{tag}>"
    end = response.rfind(closing)
    if end < 0:
        return None

    start = response.rfind(opening, 0, end)
    if start < 0:
        return None

    start += len(opening)
    end = response.find(closing, start)
    return response[start:end].strip()


def all_tags(response, tag):
    response = response or ""
    opening, closing = f"<{tag}>", f"</{tag}>"
    values = []
    cursor = 0

    while True:
        start = response.find(opening, cursor)
        if start < 0:
            break

        start += len(opening)
        end = response.find(closing, start)
        if end < 0:
            break

        values.append(response[start:end].strip())
        cursor = end + len(closing)

    return values
