-- auto-generated definition
create table character_settings
(
    id                 serial
        primary key,
    name               varchar(100) not null,
    system_instruction text         not null,
    created_at         timestamp default CURRENT_TIMESTAMP
);

alter table character_settings
    owner to postgres;


-- auto-generated definition
create table chat_messages
(
    id           serial
        primary key,
    character_id integer     not null
        references character_settings
            on delete cascade,
    role         varchar(20) not null
        constraint chat_messages_role_check
            check ((role)::text = ANY (ARRAY [('user'::character varying)::text, ('model'::character varying)::text])),
    content      text        not null,
    model        varchar(100),
    media_path   varchar(255),
    media_type   varchar(20),
    timestamp    timestamp default CURRENT_TIMESTAMP,
    is_extracted boolean   default false
);

alter table chat_messages
    owner to postgres;

create index idx_chat_messages_character_id
    on chat_messages (character_id);

create index idx_chat_messages_timestamp
    on chat_messages (timestamp);

-- auto-generated definition
create table core_fact_memories
(
    id               serial
        primary key,
    character_id     integer          not null,
    fact_text        text             not null,
    embedding        vector(768),
    category         varchar(50)      not null,
    stability_score  double precision not null,
    validation_score double precision default 1.0,
    is_archived      boolean          default false,
    evidence_span    text,
    created_at       timestamp        default CURRENT_TIMESTAMP,
    updated_at       timestamp        default CURRENT_TIMESTAMP
);

alter table core_fact_memories
    owner to postgres;

create index idx_core_char
    on core_fact_memories (character_id);

-- auto-generated definition
create table episodic_memories
(
    id                  serial
        primary key,
    character_id        integer not null,
    content             text    not null,
    emotion_intensity   double precision,
    promotion_candidate boolean   default true,
    created_at          timestamp default CURRENT_TIMESTAMP
);

alter table episodic_memories
    owner to postgres;

