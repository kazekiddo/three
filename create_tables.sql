-- 创建角色设定表
CREATE TABLE character_settings (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    system_instruction TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建聊天记录表
CREATE TABLE chat_messages (
    id SERIAL PRIMARY KEY,
    character_id INTEGER NOT NULL REFERENCES character_settings(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
    message TEXT NOT NULL,
    model VARCHAR(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引提升查询性能
CREATE INDEX idx_chat_messages_character_id ON chat_messages(character_id);
CREATE INDEX idx_chat_messages_timestamp ON chat_messages(timestamp);

-- 插入默认角色
INSERT INTO character_settings (name, system_instruction) 
VALUES ('默认助手', '你是一个友好、专业的AI助手');
