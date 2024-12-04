import json
import os
from typing import Any, Literal, cast
from zipfile import ZipFile

from pydantic import BaseModel, Field
from pydantic_core import to_json
from typing_extensions import override

from inspect_ai._util.constants import LOG_SCHEMA_VERSION
from inspect_ai._util.content import ContentImage, ContentText
from inspect_ai._util.error import EvalError
from inspect_ai._util.file import dirname, file
from inspect_ai._util.json import jsonable_python
from inspect_ai._util.zip import (
    ZIP_COMPRESSION_LEVEL,
    ZIP_COMPRESSION_METHOD,
    ZipReader,
    ZipWriter,
)
from inspect_ai.model._chat_message import ChatMessage
from inspect_ai.scorer._metric import Score

from .._log import (
    EvalLog,
    EvalPlan,
    EvalResults,
    EvalSample,
    EvalSampleReductions,
    EvalSpec,
    EvalStats,
    sort_samples,
)
from .file import FileRecorder

# TODO: Test on S3


class SampleSummary(BaseModel):
    id: int | str
    epoch: int
    input: str | list[ChatMessage]
    target: str | list[str]
    scores: dict[str, Score] | None = Field(default=None)
    error: str | None = Field(default=None)
    limit: str | None = Field(default=None)


class LogStart(BaseModel):
    version: int
    eval: EvalSpec
    plan: EvalPlan


class LogResults(BaseModel):
    status: Literal["started", "success", "cancelled", "error"]
    stats: EvalStats
    results: EvalResults | None = Field(default=None)
    error: EvalError | None = Field(default=None)


JOURNAL_DIR = "_journal"
SUMMARY_DIR = "summaries"
SAMPLES_DIR = "samples"

START_JSON = "start.json"
RESULTS_JSON = "results.json"
REDUCTIONS_JSON = "reductions.json"
SUMMARIES_JSON = "summaries.json"
HEADER_JSON = "header.json"


class EvalRecorder(FileRecorder):
    @override
    @classmethod
    def handles_location(cls, location: str) -> bool:
        return location.endswith(".eval")

    @override
    def default_log_buffer(self) -> int:
        # .eval files are 5-8x smaller than .json files so we
        # are much less worried about flushing frequently
        return 10

    def __init__(self, log_dir: str, fs_options: dict[str, Any] = {}):
        super().__init__(log_dir, ".eval", fs_options)

        # each eval has a unique key (created from run_id and task name/version)
        # which we use to track the output path, accumulated data, and event counter
        self.data: dict[str, ZipLogFile] = {}

    @override
    def log_init(self, eval: EvalSpec, location: str | None = None) -> str:
        # file to write to
        zip_file = location or self._log_file_path(eval)

        # create zip wrapper
        zip_log_file = ZipLogFile(file=zip_file)

        # create new zip file or read existing summaries/header
        if not os.path.exists(zip_file):
            with file(zip_file, "wb") as f:
                with ZipFile(
                    f,
                    "w",
                    compression=ZIP_COMPRESSION_METHOD,
                    compresslevel=ZIP_COMPRESSION_LEVEL,
                ):
                    log_start: LogStart | None = None
                    summary_counter = 0
                    summaries: list[SampleSummary] = []
        else:
            with file(zip_file, "rb") as f:
                zip = ZipReader(f)
                log_start = _read_start(zip)
                summary_counter = _read_summary_counter(zip)
                summaries = _read_all_summaries(zip, summary_counter)

        # initialise the zip file
        zip_log_file.init(log_start, summary_counter, summaries)

        # track zip
        self.data[self._log_file_key(eval)] = zip_log_file

        # return file path
        return zip_file

    @override
    def log_start(self, eval: EvalSpec, plan: EvalPlan) -> None:
        start = LogStart(version=LOG_SCHEMA_VERSION, eval=eval, plan=plan)
        self._write(eval, _journal_path(START_JSON), start)

        log = self.data[self._log_file_key(eval)]  # noqa: F841
        log.log_start = start

    @override
    def log_sample(self, eval: EvalSpec, sample: EvalSample) -> None:
        log = self.data[self._log_file_key(eval)]  # noqa: F841
        log.samples.append(sample)

    @override
    def flush(self, eval: EvalSpec) -> None:
        # write the buffered samples
        self._write_buffered_samples(eval)

    @override
    def log_finish(
        self,
        eval: EvalSpec,
        status: Literal["started", "success", "cancelled", "error"],
        stats: EvalStats,
        results: EvalResults | None,
        reductions: list[EvalSampleReductions] | None,
        error: EvalError | None = None,
    ) -> EvalLog:
        # get the key and log
        key = self._log_file_key(eval)
        log = self.data[key]

        # write the buffered samples
        self._write_buffered_samples(eval)

        # write consolidated summaries
        self._write(eval, SUMMARIES_JSON, log.summaries)

        # write reductions
        if reductions is not None:
            self._write(
                eval,
                REDUCTIONS_JSON,
                reductions,
            )

        # Get the results
        log_results = LogResults(
            status=status, stats=stats, results=results, error=error
        )

        # add the results to the original eval log from start.json
        log_start = log.log_start
        if log_start is None:
            raise RuntimeError("Unexpectedly issing the log start value")

        eval_header = EvalLog(
            version=log_start.version,
            eval=log_start.eval,
            plan=log_start.plan,
            results=log_results.results,
            stats=log_results.stats,
            status=log_results.status,
            error=log_results.error,
        )

        # write the results
        self._write(eval, HEADER_JSON, eval_header)

        # stop tracking this eval
        del self.data[key]

        # return the full EvalLog
        return self.read_log(log.file)

    @classmethod
    @override
    def read_log(cls, location: str, header_only: bool = False) -> EvalLog:
        with file(location, "rb") as z:
            zip = ZipReader(z)
            evalLog = _read_header(zip, location)
            if REDUCTIONS_JSON in zip.filenames():
                reductions = [
                    EvalSampleReductions(**reduction)
                    for reduction in _read_json(zip, REDUCTIONS_JSON)
                ]
                if evalLog.results is not None:
                    evalLog.reductions = reductions

            samples: list[EvalSample] | None = None
            if not header_only:
                samples = []
                for name in zip.filenames():
                    if name.startswith(f"{SAMPLES_DIR}/") and name.endswith(".json"):
                        samples.append(EvalSample(**_read_json(zip, name)))
                sort_samples(samples)
                evalLog.samples = samples
            return evalLog

    @override
    @classmethod
    def read_log_sample(
        cls, location: str, id: str | int, epoch: int = 1
    ) -> EvalSample:
        with file(location, "rb") as z:
            zip = ZipReader(z)
            sample_file = _sample_filename(id, epoch)
            if sample_file in zip.filenames():
                return EvalSample(**_read_json(zip, sample_file))
            else:
                raise IndexError(
                    f"Sample id {id} for epoch {epoch} not found in log {location}"
                )

    @classmethod
    @override
    def write_log(cls, location: str, log: EvalLog) -> None:
        # write using the recorder (so we get all of the extra streams)
        recorder = EvalRecorder(dirname(location))
        recorder.log_init(log.eval, location)
        recorder.log_start(log.eval, log.plan)
        for sample in log.samples or []:
            recorder.log_sample(log.eval, sample)
        recorder.log_finish(
            log.eval, log.status, log.stats, log.results, log.reductions, log.error
        )

    # write to the zip file
    def _write(self, eval: EvalSpec, filename: str, data: Any) -> None:
        log = self.data[self._log_file_key(eval)]

        with file(log.file, "rb") as f:
            reader = ZipReader(f)
        with file(log.file, "ab") as f:
            with ZipWriter(f, reader) as zip:
                zip.write(filename, zip_file_data(data))

    # write buffered samples to the zip file
    def _write_buffered_samples(self, eval: EvalSpec) -> None:
        # get the log
        log = self.data[self._log_file_key(eval)]

        # Write the buffered samples
        summaries: list[SampleSummary] = []
        for sample in log.samples:
            # Write the sample
            self._write(eval, _sample_filename(sample.id, sample.epoch), sample)

            # Capture the summary
            summaries.append(
                SampleSummary(
                    id=sample.id,
                    epoch=sample.epoch,
                    input=text_inputs(sample.input),
                    target=sample.target,
                    scores=sample.scores,
                    error=sample.error.message if sample.error is not None else None,
                    limit=f"{sample.limit.type}" if sample.limit is not None else None,
                )
            )
        log.samples.clear()

        # write intermediary summaries and add to master list
        if len(summaries) > 0:
            log.summary_counter += 1
            summary_file = _journal_summary_file(log.summary_counter)
            summary_path = _journal_summary_path(summary_file)
            self._write(eval, summary_path, summaries)
            log.summaries.extend(summaries)


def zip_file_data(data: Any) -> bytes:
    return to_json(
        value=jsonable_python(data),
        indent=2,
        exclude_none=True,
        fallback=lambda _x: None,
    )


def text_inputs(inputs: str | list[ChatMessage]) -> str | list[ChatMessage]:
    # Clean the input of any images
    if isinstance(inputs, list):
        input: list[ChatMessage] = []
        for message in inputs:
            if not isinstance(message.content, str):
                filtered_content: list[ContentText | ContentImage] = []
                for content in message.content:
                    if content.type != "image":
                        filtered_content.append(content)
                if len(filtered_content) == 0:
                    filtered_content.append(ContentText(text="(Image)"))
                message.content = filtered_content
                input.append(message)

        return input
    else:
        return inputs


class ZipLogFile:
    def __init__(self, file: str) -> None:
        self.file = file
        self.samples: list[EvalSample] = []
        self.summary_counter = 0
        self.summaries: list[SampleSummary] = []
        self.log_start: LogStart | None = None

    def init(
        self,
        log_start: LogStart | None,
        summary_counter: int,
        summaries: list[SampleSummary],
    ) -> None:
        self.summary_counter = summary_counter
        self.summaries = summaries
        self.log_start = log_start


def _read_start(zip: ZipReader) -> LogStart | None:
    start_path = _journal_path(START_JSON)
    if start_path in zip.filenames():
        return cast(LogStart, _read_json(zip, start_path))
    else:
        return None


def _read_summary_counter(zip: ZipReader) -> int:
    current_count = 0
    for name in zip.filenames():
        if name.startswith(_journal_summary_path()) and name.endswith(".json"):
            this_count = int(name.split("/")[-1].split(".")[0])
            current_count = max(this_count, current_count)
    return current_count


def _read_all_summaries(zip: ZipReader, count: int) -> list[SampleSummary]:
    if SUMMARIES_JSON in zip.filenames():
        summaries_raw = _read_json(zip, SUMMARIES_JSON)
        print(json.dumps(summaries_raw, indent=2))
        if isinstance(summaries_raw, list):
            return [SampleSummary(**value) for value in summaries_raw]
        else:
            raise ValueError(
                f"Expected a list of summaries when reading {SUMMARIES_JSON}"
            )
    else:
        summaries: list[SampleSummary] = []
        for i in range(1, count):
            summary_file = _journal_summary_file(i)
            summary_path = _journal_summary_path(summary_file)
            summary = _read_json(zip, summary_path)
            if isinstance(summary, list):
                summaries.extend([SampleSummary(**value) for value in summary])
            else:
                raise ValueError(
                    f"Expected a list of summaries when reading {summary_file}"
                )
        return summaries


def _read_header(zip: ZipReader, location: str) -> EvalLog:
    # first see if the header is here
    if HEADER_JSON in zip.filenames():
        log = EvalLog(**_read_json(zip, HEADER_JSON))
        log.location = location
        return log
    else:
        start = LogStart(**_read_json(zip, _journal_path(START_JSON)))
        return EvalLog(
            version=start.version, eval=start.eval, plan=start.plan, location=location
        )


def _sample_filename(id: str | int, epoch: int) -> str:
    return f"{SAMPLES_DIR}/{id}_epoch_{epoch}.json"


def _read_json(zip: ZipReader, filename: str) -> Any:
    contents = zip.read(filename)
    return json.loads(contents.decode())


def _journal_path(file: str) -> str:
    return JOURNAL_DIR + "/" + file


def _journal_summary_path(file: str | None = None) -> str:
    if file is None:
        return _journal_path(SUMMARY_DIR)
    else:
        return f"{_journal_path(SUMMARY_DIR)}/{file}"


def _journal_summary_file(index: int) -> str:
    return f"{index}.json"
