use litert_lm::Conversation;
use anyhow::Context;
use std::fmt::{Display, Formatter};
use console::{style, Key, Term};
use serde_json::{json, Value};
use std::io::Write;

pub fn chat_loop(
    conversation: Conversation,
    initial_prompt: &str,
) -> anyhow::Result<()> {
    let term = Term::stdout();
    let mut editor = LineEditor::new(term.clone());

    term.write_line("Type `exit` or `quit` to leave.")?;

    run_chat_repl(&term, &mut editor, &conversation, initial_prompt)?;

    Ok(())
}

fn run_chat_repl(
    term: &Term,
    editor: &mut LineEditor,
    conversation: &Conversation,
    initial_prompt: &str,
) -> anyhow::Result<()> {

    let greet_response = send_user_text_with_tools(conversation, initial_prompt)
        .context("initial prompt")?;
    term.write_line(&style(&greet_response).bold().green().to_string())?;

    loop {
        let input = match editor.read_line(">>> ")? {
            Some(s) => s,
            None => break Ok(()),
        };

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        match ReplCommand::from_str(trimmed) {
            ReplCommand::Exit => break Ok(()),
            ReplCommand::ChatBotInput(user_input) => {
                let response = send_user_text_with_tools(conversation, &user_input)
                    .context("send_user_text")?;
                if response.is_empty() {
                    anyhow::bail!("model returned empty output (try increasing tokens)");
                }
                term.write_line(&style(&response).bold().green().to_string())?;
            }
        }
    }
}

fn send_user_text_with_tools(conversation: &Conversation, text: &str) -> anyhow::Result<String> {
    let user_message = json!({
        "role": "user",
        "content": [{"type": "text", "text": text}],
    });

    let mut message = conversation
        .send_message_value(&user_message, None)
        .context("conversation.send_message_value(user)")?;

    loop {
        let out_text = extract_message_text_value(&message).unwrap_or_default();

        let tool_calls = message.get("tool_calls").and_then(|v| v.as_array());
        let Some(tool_calls) = tool_calls else {
            return Ok(out_text);
        };
        if tool_calls.is_empty() {
            return Ok(out_text);
        }

        let mut tool_outputs: Vec<Value> = Vec::with_capacity(tool_calls.len());

        for tool_call in tool_calls {
            let function = tool_call.get("function").unwrap_or(tool_call);
            let name = function.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let args = function.get("arguments").unwrap_or(&Value::Null);

            if name == crate::date_time_tool::DateTimeTool::NAME {
                tool_outputs.push(crate::date_time_tool::DateTimeTool.call(args));
            } else {
                tool_outputs.push(json!({
                    "name": name,
                    "response": {
                        "error": format!("unknown tool: {name}")
                    }
                }));
            }
        }

        // Send all tool outputs in a single tool message.
        let tool_message = json!({
            "role": "tool",
            "content": tool_outputs,
        });

        message = conversation
            .send_message_value(&tool_message, None)
            .context("conversation.send_message_value(tool)")?;
    }
}

fn extract_message_text_value(message_json: &Value) -> Option<String> {
    let content = message_json.get("content")?;

    if let Some(arr) = content.as_array() {
        let mut out = String::new();
        for item in arr {
            let t = item.get("type").and_then(|x| x.as_str()).unwrap_or("");
            if t == "text" {
                if let Some(s) = item.get("text").and_then(|x| x.as_str()) {
                    out.push_str(s);
                }
            } else if let Some(s) = item.get("text").and_then(|x| x.as_str()) {
                out.push_str(s);
            }
        }
        return Some(out);
    }

    if let Some(s) = content.as_str() {
        return Some(s.to_string());
    }

    None
}

enum ReplCommand {
    Exit,
    ChatBotInput(String),
}

impl ReplCommand {
    fn from_str(s: &str) -> Self {
        match s {
            "exit" | "quit" => ReplCommand::Exit,
            s => ReplCommand::ChatBotInput(s.to_string()),
        }
    }
}

impl Display for ReplCommand {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ReplCommand::Exit => write!(f, "exit"),
            ReplCommand::ChatBotInput(input) => write!(f, "{}", input),
        }
    }
}

struct LineEditor {
    term: Term,
    history: Vec<String>,
    history_pos: Option<usize>,
}

impl LineEditor {
    fn new(term: Term) -> Self {
        Self {
            term,
            history: Vec::new(),
            history_pos: None,
        }
    }

    fn read_line(&mut self, prompt: &str) -> anyhow::Result<Option<String>> {
        if !self.term.is_term() {
            // Non-interactive: fall back to a plain line read.
            write!(&self.term, "{prompt}")?;
            self.term.flush()?;
            let line = self.term.read_line()?;
            return Ok(Some(line));
        }

        let mut buf = String::new();
        let mut cursor_chars: usize = 0;
        self.history_pos = None;

        self.render(prompt, &buf, cursor_chars)?;

        loop {
            match self.term.read_key()? {
                Key::Enter => {
                    self.term.write_line("")?;
                    if !buf.trim().is_empty() {
                        self.history.push(buf.clone());
                    }
                    return Ok(Some(buf));
                }
                Key::CtrlC => {
                    self.term.clear_line()?;
                    self.term.write_line("^C")?;
                    return Ok(Some(String::new()));
                }
                Key::Escape => {
                    buf.clear();
                    cursor_chars = 0;
                }
                Key::Backspace => {
                    if cursor_chars > 0 {
                        let start = char_to_byte_idx(&buf, cursor_chars - 1);
                        let end = char_to_byte_idx(&buf, cursor_chars);
                        buf.replace_range(start..end, "");
                        cursor_chars -= 1;
                    }
                }
                Key::Del => {
                    let len = buf.chars().count();
                    if cursor_chars < len {
                        let start = char_to_byte_idx(&buf, cursor_chars);
                        let end = char_to_byte_idx(&buf, cursor_chars + 1);
                        buf.replace_range(start..end, "");
                    }
                }
                Key::ArrowLeft => {
                    cursor_chars = cursor_chars.saturating_sub(1);
                }
                Key::ArrowRight => {
                    let len = buf.chars().count();
                    cursor_chars = (cursor_chars + 1).min(len);
                }
                Key::Home => cursor_chars = 0,
                Key::End => cursor_chars = buf.chars().count(),
                Key::ArrowUp => {
                    if self.history.is_empty() {
                        // nothing
                    } else {
                        let next_pos = match self.history_pos {
                            None => self.history.len() - 1,
                            Some(p) => p.saturating_sub(1),
                        };
                        self.history_pos = Some(next_pos);
                        buf = self.history[next_pos].clone();
                        cursor_chars = buf.chars().count();
                    }
                }
                Key::ArrowDown => {
                    let Some(p) = self.history_pos else {
                        // nothing
                        self.render(prompt, &buf, cursor_chars)?;
                        continue;
                    };

                    if p + 1 < self.history.len() {
                        let next_pos = p + 1;
                        self.history_pos = Some(next_pos);
                        buf = self.history[next_pos].clone();
                        cursor_chars = buf.chars().count();
                    } else {
                        self.history_pos = None;
                        buf.clear();
                        cursor_chars = 0;
                    }
                }
                Key::Char(c) => {
                    let byte_idx = char_to_byte_idx(&buf, cursor_chars);
                    buf.insert(byte_idx, c);
                    cursor_chars += 1;
                }
                _ => {}
            }

            self.render(prompt, &buf, cursor_chars)?;
        }
    }

    fn render(&self, prompt: &str, buf: &str, cursor_chars: usize) -> anyhow::Result<()> {
        self.term.clear_line()?;
        write!(&self.term, "{prompt}{buf}")?;
        self.term.flush()?;

        // We wrote the full buffer; cursor is at end. Move left to desired position.
        let cursor_byte = char_to_byte_idx(buf, cursor_chars);
        let suffix = &buf[cursor_byte..];
        let suffix_width = console::measure_text_width(suffix);
        if suffix_width > 0 {
            self.term.move_cursor_left(suffix_width)?;
        }
        Ok(())
    }
}

fn char_to_byte_idx(s: &str, char_idx: usize) -> usize {
    if char_idx == 0 {
        return 0;
    }
    match s.char_indices().nth(char_idx) {
        Some((byte_idx, _)) => byte_idx,
        None => s.len(),
    }
}
