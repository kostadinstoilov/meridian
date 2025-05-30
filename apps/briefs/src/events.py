import requests
import os
from datetime import date
from pydantic import BaseModel, field_validator
from typing import Optional
import pandas as pd
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

load_dotenv()


class Source(BaseModel):
    id: int
    name: str


class Event(BaseModel):
    id: int
    sourceId: int
    url: str
    title: str
    publishDate: datetime  # changed from date to datetime
    contentFileKey: str
    primary_location: str
    completeness: str
    content_quality: str
    event_summary_points: list[str]
    thematic_keywords: list[str]
    topic_tags: list[str]
    key_entities: list[str]
    content_focus: list[str]
    embedding: list[float]
    createdAt: datetime

    @field_validator("publishDate", mode="before")
    @classmethod
    def parse_date(cls, value):
        if value is None:
            return None

        # Handle ISO format with timezone info
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            # For older Python versions or non-standard formats
            # you might need dateutil
            from dateutil import parser

            return parser.parse(value)


def get_events(date: str = None):
    url = f"http://localhost:8787/events"

    if date:
        url += f"?date={date}"

    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {os.environ.get('MERIDIAN_SECRET_KEY')}"},
    )
    data = response.json()

    sources = [Source(**source) for source in data["sources"]]
    events = [Event(**event) for event in data["events"]]

    return sources, events
