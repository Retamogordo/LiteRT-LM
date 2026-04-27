use anyhow::{anyhow, Context, Result};
use litert_lm_sys::{
    LiteRtLmApi, LiteRtLmConversation, LiteRtLmEngine, LiteRtLmEngineSettings, LiteRtLmInputData,
    LiteRtLmConversationConfig, LiteRtLmInputDataType, LiteRtLmJsonResponse, LiteRtLmResponses,
    LiteRtLmSession, LiteRtLmSessionConfig,
};
use serde_json::Value;
use std::ffi::{CStr, CString};
use std::os::raw::c_void;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

#[derive(Clone)]
pub struct Api {
    inner: Arc<LiteRtLmApi>,
}

impl Api {
    pub unsafe fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let api = LiteRtLmApi::load(path).context("loading LiteRT-LM C API .so")?;
        Ok(Self {
            inner: Arc::new(api),
        })
    }

    pub fn set_min_log_level(&self, level: i32) {
        unsafe { (self.inner.litert_lm_set_min_log_level)(level) }
    }
}

pub struct EngineSettings {
    api: Arc<LiteRtLmApi>,
    ptr: NonNull<LiteRtLmEngineSettings>,
}

impl EngineSettings {
    pub fn new(api: &Api, model_path: &str, backend: &str) -> Result<Self> {
        let model_path = CString::new(model_path)?;
        let backend = CString::new(backend)?;

        let raw = unsafe {
            (api.inner.litert_lm_engine_settings_create)(
                model_path.as_ptr(),
                backend.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        let ptr = NonNull::new(raw).ok_or_else(|| anyhow!("litert_lm_engine_settings_create returned null"))?;
        Ok(Self {
            api: api.inner.clone(),
            ptr,
        })
    }

    pub fn set_max_num_tokens(&mut self, max_num_tokens: i32) {
        unsafe { (self.api.litert_lm_engine_settings_set_max_num_tokens)(self.ptr.as_ptr(), max_num_tokens) }
    }
}

impl Drop for EngineSettings {
    fn drop(&mut self) {
        unsafe { (self.api.litert_lm_engine_settings_delete)(self.ptr.as_ptr()) }
    }
}

pub struct Engine {
    api: Arc<LiteRtLmApi>,
    ptr: NonNull<LiteRtLmEngine>,
}

impl Engine {
    pub fn new(api: &Api, settings: &EngineSettings) -> Result<Self> {
        let raw = unsafe { (api.inner.litert_lm_engine_create)(settings.ptr.as_ptr()) };
        let ptr = NonNull::new(raw).ok_or_else(|| anyhow!("litert_lm_engine_create returned null"))?;
        Ok(Self {
            api: api.inner.clone(),
            ptr,
        })
    }

    pub fn create_session(&self, config: Option<&mut SessionConfig>) -> Result<Session> {
        let config_ptr = config
            .map(|c| c.ptr.as_ptr())
            .unwrap_or(std::ptr::null_mut());

        let raw = unsafe { (self.api.litert_lm_engine_create_session)(self.ptr.as_ptr(), config_ptr) };
        let ptr = NonNull::new(raw).ok_or_else(|| anyhow!("litert_lm_engine_create_session returned null"))?;
        Ok(Session {
            api: self.api.clone(),
            ptr,
        })
    }

    pub fn create_conversation(&self) -> Result<Conversation> {
        self.create_conversation_with_config(None)
    }

    pub fn create_conversation_with_config(&self, config: Option<&ConversationConfig>) -> Result<Conversation> {
        let config_ptr = config
            .map(|c| c.ptr.as_ptr())
            .unwrap_or(std::ptr::null_mut());

        let raw = unsafe { (self.api.litert_lm_conversation_create)(self.ptr.as_ptr(), config_ptr) };
        let ptr = NonNull::new(raw).ok_or_else(|| anyhow!("litert_lm_conversation_create returned null"))?;
        Ok(Conversation {
            api: self.api.clone(),
            ptr,
        })
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { (self.api.litert_lm_engine_delete)(self.ptr.as_ptr()) }
    }
}

pub struct SessionConfig {
    api: Arc<LiteRtLmApi>,
    ptr: NonNull<LiteRtLmSessionConfig>,
}

impl SessionConfig {
    pub fn new(api: &Api) -> Result<Self> {
        let raw = unsafe { (api.inner.litert_lm_session_config_create)() };
        let ptr = NonNull::new(raw).ok_or_else(|| anyhow!("litert_lm_session_config_create returned null"))?;
        Ok(Self {
            api: api.inner.clone(),
            ptr,
        })
    }

    pub fn set_max_output_tokens(&mut self, max_output_tokens: i32) {
        unsafe { (self.api.litert_lm_session_config_set_max_output_tokens)(self.ptr.as_ptr(), max_output_tokens) }
    }
}

impl Drop for SessionConfig {
    fn drop(&mut self) {
        unsafe { (self.api.litert_lm_session_config_delete)(self.ptr.as_ptr()) }
    }
}

pub struct ConversationConfig {
    api: Arc<LiteRtLmApi>,
    ptr: NonNull<LiteRtLmConversationConfig>,
}

impl ConversationConfig {
    /// Creates a conversation config with optional JSON preface components.
    ///
    /// - `tools` should be a JSON array of tool definitions.
    /// - `system_message` should be a single message JSON object (role/content).
    /// - `messages` should be a JSON array of message objects.
    pub fn create(
        engine: &Engine,
        session_config: Option<&SessionConfig>,
        system_message: Option<&Value>,
        tools: Option<&Value>,
        messages: Option<&Value>,
        enable_constrained_decoding: bool,
    ) -> Result<Self> {
        let system_message_c = system_message
            .map(|v| CString::new(v.to_string()))
            .transpose()
            .context("building system_message CString")?;
        let tools_c = tools
            .map(|v| CString::new(v.to_string()))
            .transpose()
            .context("building tools CString")?;
        let messages_c = messages
            .map(|v| CString::new(v.to_string()))
            .transpose()
            .context("building messages CString")?;

        let raw = unsafe {
            (engine.api.litert_lm_conversation_config_create)(
                engine.ptr.as_ptr(),
                session_config
                    .map(|c| c.ptr.as_ptr() as *const LiteRtLmSessionConfig)
                    .unwrap_or(std::ptr::null()),
                system_message_c
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null()),
                tools_c
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null()),
                messages_c
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null()),
                enable_constrained_decoding,
            )
        };

        let ptr = NonNull::new(raw).ok_or_else(|| anyhow!("litert_lm_conversation_config_create returned null"))?;
        Ok(Self {
            api: engine.api.clone(),
            ptr,
        })
    }
}

impl Drop for ConversationConfig {
    fn drop(&mut self) {
        unsafe { (self.api.litert_lm_conversation_config_delete)(self.ptr.as_ptr()) }
    }
}

pub struct Session {
    api: Arc<LiteRtLmApi>,
    ptr: NonNull<LiteRtLmSession>,
}

impl Session {
    pub fn generate_text(&self, prompt: &str) -> Result<String> {
        let prompt_c = CString::new(prompt)?;
        let input = LiteRtLmInputData {
            type_: LiteRtLmInputDataType::kLiteRtLmInputDataTypeText,
            data: prompt_c.as_ptr() as *const c_void,
            size: prompt.len(),
        };

        let responses = unsafe { (self.api.litert_lm_session_generate_content)(self.ptr.as_ptr(), &input, 1) };
        
        let responses = Responses::from_raw(self.api.clone(), responses)?;

        let n = responses.num_candidates();
        if n <= 0 {
            return Err(anyhow!("no response candidates returned"));
        }
        responses.text_at(0).ok_or_else(|| anyhow!("null response text"))
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe { (self.api.litert_lm_session_delete)(self.ptr.as_ptr()) }
    }
}

pub struct Conversation {
    api: Arc<LiteRtLmApi>,
    ptr: NonNull<LiteRtLmConversation>,
}

impl Conversation {
    /// Sends a pre-formatted message JSON string and returns the raw JSON response.
    pub fn send_message_json(&self, message_json: &str, extra_context: Option<&str>) -> Result<String> {
        let message_c = CString::new(message_json).context("building message_json CString")?;
        let extra_context_c = extra_context
            .map(CString::new)
            .transpose()
            .context("building extra_context CString")?;

        let raw = unsafe {
            (self.api.litert_lm_conversation_send_message)(
                self.ptr.as_ptr(),
                message_c.as_ptr(),
                extra_context_c
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null()),
            )
        };
        let response = JsonResponse::from_raw(self.api.clone(), raw)?;
        let s = response
            .as_str()
            .context("json_response_get_string returned null")?;
        Ok(s.to_string())
    }

    /// Sends a structured message (as JSON) and returns the structured JSON response.
    pub fn send_message_value(&self, message: &Value, extra_context: Option<&str>) -> Result<Value> {
        let raw = self.send_message_json(&message.to_string(), extra_context)?;
        serde_json::from_str(&raw).context("parsing response JSON")
    }

    pub fn send_user_text(&self, text: &str) -> Result<String> {
        let message_json = serde_json::json!({
            "role": "user",
            "content": [{"type": "text", "text": text}],
        });
        let s = self.send_message_json(&message_json.to_string(), None)?;

        // Prefer returning plain assistant text for a REPL.
        Ok(extract_message_text(&s).unwrap_or_else(|| s.to_string()))
    }
}

impl Drop for Conversation {
    fn drop(&mut self) {
        unsafe { (self.api.litert_lm_conversation_delete)(self.ptr.as_ptr()) }
    }
}

pub struct Responses {
    api: Arc<LiteRtLmApi>,
    ptr: NonNull<LiteRtLmResponses>,
}

impl Responses {
    fn from_raw(api: Arc<LiteRtLmApi>, raw: *mut LiteRtLmResponses) -> Result<Self> {
        let ptr = NonNull::new(raw).ok_or_else(|| anyhow!("litert_lm_session_generate_content returned null"))?;
        Ok(Self { api, ptr })
    }

    pub fn num_candidates(&self) -> i32 {
        unsafe { (self.api.litert_lm_responses_get_num_candidates)(self.ptr.as_ptr()) }
    }

    pub fn text_at(&self, index: i32) -> Option<String> {
        let p = unsafe { (self.api.litert_lm_responses_get_response_text_at)(self.ptr.as_ptr(), index) };
        if p.is_null() {
            return None;
        }
        let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().to_string();
        Some(s)
    }
}

impl Drop for Responses {
    fn drop(&mut self) {
        unsafe { (self.api.litert_lm_responses_delete)(self.ptr.as_ptr()) }
    }
}

pub struct JsonResponse {
    api: Arc<LiteRtLmApi>,
    ptr: NonNull<LiteRtLmJsonResponse>,
}

impl JsonResponse {
    fn from_raw(api: Arc<LiteRtLmApi>, raw: *mut LiteRtLmJsonResponse) -> Result<Self> {
        let ptr = NonNull::new(raw).ok_or_else(|| anyhow!("litert_lm_conversation_send_message returned null"))?;
        Ok(Self { api, ptr })
    }

    pub fn as_str(&self) -> Option<&str> {
        let p = unsafe { (self.api.litert_lm_json_response_get_string)(self.ptr.as_ptr()) };
        if p.is_null() {
            return None;
        }
        unsafe { CStr::from_ptr(p) }.to_str().ok()
    }
}

impl Drop for JsonResponse {
    fn drop(&mut self) {
        unsafe { (self.api.litert_lm_json_response_delete)(self.ptr.as_ptr()) }
    }
}

fn extract_message_text(message_json: &str) -> Option<String> {
    let v: Value = serde_json::from_str(message_json).ok()?;
    let content = v.get("content")?;

    // Common shape: content = [{ "type": "text", "text": "..." }, ...]
    if let Some(arr) = content.as_array() {
        let mut out = String::new();
        for item in arr {
            let t = item.get("type").and_then(|x| x.as_str()).unwrap_or("");
            if t == "text" {
                if let Some(s) = item.get("text").and_then(|x| x.as_str()) {
                    out.push_str(s);
                }
            } else if let Some(s) = item.get("text").and_then(|x| x.as_str()) {
                // Best-effort fallback for unknown types that still carry text.
                out.push_str(s);
            }
        }
        return Some(out);
    }

    // Sometimes content is just a string.
    if let Some(s) = content.as_str() {
        return Some(s.to_string());
    }

    None
}
