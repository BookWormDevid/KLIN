BEGIN;

CREATE TABLE alembic_version (
    version_num VARCHAR(32) NOT NULL,
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);

INFO  [alembic.runtime.migration] Running upgrade  -> 7065f6dd6444, initial
-- Running upgrade  -> 7065f6dd6444

CREATE TABLE klin (
    response_url VARCHAR,
    video_path VARCHAR NOT NULL,
    state VARCHAR NOT NULL,
    x3d VARCHAR,
    mae VARCHAR,
    yolo VARCHAR,
    all_classes VARCHAR[],
    objects VARCHAR[],
    id UUID NOT NULL,
    is_removed BOOLEAN NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    PRIMARY KEY (id),
    UNIQUE (id)
);

CREATE INDEX ix_klin_created_at ON klin (created_at);

CREATE INDEX ix_klin_is_removed ON klin (is_removed);

CREATE TABLE klin_stream (
    camera_id VARCHAR,
    camera_url VARCHAR,
    state VARCHAR NOT NULL,
    x3d VARCHAR,
    mae VARCHAR,
    yolo VARCHAR,
    all_classes VARCHAR[],
    objects VARCHAR[],
    id UUID NOT NULL,
    is_removed BOOLEAN NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    PRIMARY KEY (id),
    UNIQUE (id)
);

CREATE INDEX ix_klin_stream_created_at ON klin_stream (created_at);

CREATE INDEX ix_klin_stream_is_removed ON klin_stream (is_removed);

INSERT INTO alembic_version (version_num) VALUES ('7065f6dd6444') RETURNING alembic_version.version_num;

COMMIT;
