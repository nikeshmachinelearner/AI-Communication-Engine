from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    text: str
    label: str
    start: int
    end: int


class ExtractedDate(BaseModel):
    text: str
    normalized: str | None = None
    start: int | None = None
    end: int | None = None


class ExtractedTask(BaseModel):
    text: str
    action: str | None = None
    assignee: str | None = None
    deadline: str | None = None
    start: int | None = None
    end: int | None = None


class ExtractedCommitment(BaseModel):
    text: str
    owner: str | None = None
    action: str | None = None


class ExtractionResult(BaseModel):
    text: str

    people: list[str] = Field(default_factory=list)
    organizations: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)

    dates: list[ExtractedDate] = Field(default_factory=list)
    tasks: list[ExtractedTask] = Field(default_factory=list)
    commitments: list[ExtractedCommitment] = Field(
        default_factory=list
    )

    entities: list[ExtractedEntity] = Field(default_factory=list)
    