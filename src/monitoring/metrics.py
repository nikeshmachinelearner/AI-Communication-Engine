from prometheus_client import Counter, Gauge, Histogram


REQUEST_COUNT = Counter(
    "communication_engine_requests_total",
    "Total number of API requests.",
    ["endpoint", "method", "status"],
)

REQUEST_LATENCY = Histogram(
    "communication_engine_request_duration_seconds",
    "API request duration in seconds.",
    ["endpoint"],
    buckets=(
        0.01,
        0.025,
        0.05,
        0.1,
        0.2,
        0.3,
        0.5,
        1.0,
        2.0,
        5.0,
    ),
)

COMPONENT_LATENCY = Histogram(
    "communication_engine_component_duration_seconds",
    "ML component execution duration in seconds.",
    ["component"],
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.2,
        0.3,
        0.5,
        1.0,
        2.0,
        5.0,
    ),
)

COMPONENT_ERRORS = Counter(
    "communication_engine_component_errors_total",
    "Total component execution errors.",
    ["component"],
)

MODEL_AVAILABLE = Gauge(
    "communication_engine_model_available",
    "Whether a model component is available.",
    ["component"],
)