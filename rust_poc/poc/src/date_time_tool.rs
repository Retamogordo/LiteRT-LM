use chrono::{Utc, Datelike};
use serde_json::{json, Value};


#[derive(Debug, Default, Clone, Copy)]
pub struct DateTimeTool;

impl DateTimeTool {
    pub const NAME: &'static str = "get_current_datetime_utc";
    pub const DESCRIPTION: &'static str =
        "Return the current UTC date-time as a string like 2025-03-01T12:00:00Z.";

    /// Tool declaration JSON (for `JsonPreface.tools`).
    pub fn declaration(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": Self::NAME,
                "description": Self::DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        })
    }

    /// Executes the tool and returns a tool-response payload.
    ///
    /// This object is intended to be placed into a message with role `"tool"`.
    pub fn call(&self, _arguments: &Value) -> Value {
        json!({
            "name": Self::NAME,
            "response": {
                "datetime": current_datetime_utc_string()
            }
        })
    }
}

fn current_datetime_utc_string() -> String {
    let now = Utc::now();
    let datetime_string = now.format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let week_day = now.weekday().to_string();
    format!("{}, {}", datetime_string, week_day)
}
