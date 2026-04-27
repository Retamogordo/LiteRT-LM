#![allow(non_camel_case_types)]

use libloading::Library;
use std::ffi::c_char;
use std::os::raw::c_void;
use std::path::Path;

#[repr(C)]
pub struct LiteRtLmEngine(c_void);
#[repr(C)]
pub struct LiteRtLmSession(c_void);
#[repr(C)]
pub struct LiteRtLmResponses(c_void);
#[repr(C)]
pub struct LiteRtLmEngineSettings(c_void);
#[repr(C)]
pub struct LiteRtLmBenchmarkInfo(c_void);
#[repr(C)]
pub struct LiteRtLmConversation(c_void);
#[repr(C)]
pub struct LiteRtLmJsonResponse(c_void);
#[repr(C)]
pub struct LiteRtLmSessionConfig(c_void);
#[repr(C)]
pub struct LiteRtLmConversationConfig(c_void);

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum LiteRtLmSamplerType {
    kLiteRtLmSamplerTypeUnspecified = 0,
    kLiteRtLmSamplerTypeTopK = 1,
    kLiteRtLmSamplerTypeTopP = 2,
    kLiteRtLmSamplerTypeGreedy = 3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LiteRtLmSamplerParams {
    pub type_: LiteRtLmSamplerType,
    pub top_k: i32,
    pub top_p: f32,
    pub temperature: f32,
    pub seed: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum LiteRtLmInputDataType {
    kLiteRtLmInputDataTypeText = 0,
    kLiteRtLmInputDataTypeImage = 1,
    kLiteRtLmInputDataTypeImageEnd = 2,
    kLiteRtLmInputDataTypeAudio = 3,
    kLiteRtLmInputDataTypeAudioEnd = 4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LiteRtLmInputData {
    pub type_: LiteRtLmInputDataType,
    pub data: *const c_void,
    pub size: usize,
}

pub type LiteRtLmStreamCallback =
    unsafe extern "C" fn(callback_data: *mut c_void, chunk: *const c_char, is_final: bool, error_msg: *const c_char);

#[derive(Clone)]
pub struct LiteRtLmApi {
    _lib: std::sync::Arc<Library>,

    pub litert_lm_set_min_log_level: unsafe extern "C" fn(level: i32),

    pub litert_lm_engine_settings_create: unsafe extern "C" fn(
        model_path: *const c_char,
        backend_str: *const c_char,
        vision_backend_str: *const c_char,
        audio_backend_str: *const c_char,
    ) -> *mut LiteRtLmEngineSettings,
    pub litert_lm_engine_settings_delete: unsafe extern "C" fn(settings: *mut LiteRtLmEngineSettings),
    pub litert_lm_engine_settings_set_max_num_tokens:
        unsafe extern "C" fn(settings: *mut LiteRtLmEngineSettings, max_num_tokens: i32),

    pub litert_lm_engine_create:
        unsafe extern "C" fn(settings: *const LiteRtLmEngineSettings) -> *mut LiteRtLmEngine,
    pub litert_lm_engine_delete: unsafe extern "C" fn(engine: *mut LiteRtLmEngine),

    pub litert_lm_session_config_create: unsafe extern "C" fn() -> *mut LiteRtLmSessionConfig,
    pub litert_lm_session_config_set_max_output_tokens:
        unsafe extern "C" fn(config: *mut LiteRtLmSessionConfig, max_output_tokens: i32),
    pub litert_lm_session_config_set_sampler_params: unsafe extern "C" fn(
        config: *mut LiteRtLmSessionConfig,
        sampler_params: *const LiteRtLmSamplerParams,
    ),
    pub litert_lm_session_config_delete: unsafe extern "C" fn(config: *mut LiteRtLmSessionConfig),

    pub litert_lm_conversation_config_create: unsafe extern "C" fn(
        engine: *mut LiteRtLmEngine,
        session_config: *const LiteRtLmSessionConfig,
        system_message_json: *const c_char,
        tools_json: *const c_char,
        messages_json: *const c_char,
        enable_constrained_decoding: bool,
    ) -> *mut LiteRtLmConversationConfig,
    pub litert_lm_conversation_config_delete:
        unsafe extern "C" fn(config: *mut LiteRtLmConversationConfig),

    pub litert_lm_engine_create_session:
        unsafe extern "C" fn(engine: *mut LiteRtLmEngine, config: *mut LiteRtLmSessionConfig) -> *mut LiteRtLmSession,
    pub litert_lm_session_delete: unsafe extern "C" fn(session: *mut LiteRtLmSession),

    pub litert_lm_session_generate_content: unsafe extern "C" fn(
        session: *mut LiteRtLmSession,
        inputs: *const LiteRtLmInputData,
        num_inputs: usize,
    ) -> *mut LiteRtLmResponses,

    pub litert_lm_session_generate_content_stream: unsafe extern "C" fn(
        session: *mut LiteRtLmSession,
        inputs: *const LiteRtLmInputData,
        num_inputs: usize,
        callback: LiteRtLmStreamCallback,
        callback_data: *mut c_void,
    ) -> i32,

    pub litert_lm_responses_delete: unsafe extern "C" fn(responses: *mut LiteRtLmResponses),
    pub litert_lm_responses_get_num_candidates: unsafe extern "C" fn(responses: *const LiteRtLmResponses) -> i32,
    pub litert_lm_responses_get_response_text_at:
        unsafe extern "C" fn(responses: *const LiteRtLmResponses, index: i32) -> *const c_char,

    pub litert_lm_conversation_create: unsafe extern "C" fn(
        engine: *mut LiteRtLmEngine,
        config: *mut LiteRtLmConversationConfig,
    ) -> *mut LiteRtLmConversation,
    pub litert_lm_conversation_delete: unsafe extern "C" fn(conversation: *mut LiteRtLmConversation),

    pub litert_lm_conversation_send_message: unsafe extern "C" fn(
        conversation: *mut LiteRtLmConversation,
        message_json: *const c_char,
        extra_context: *const c_char,
    ) -> *mut LiteRtLmJsonResponse,

    pub litert_lm_json_response_delete: unsafe extern "C" fn(response: *mut LiteRtLmJsonResponse),
    pub litert_lm_json_response_get_string:
        unsafe extern "C" fn(response: *const LiteRtLmJsonResponse) -> *const c_char,
}

impl LiteRtLmApi {
    pub unsafe fn load<P: AsRef<Path>>(path: P) -> Result<Self, libloading::Error> {
        let lib = std::sync::Arc::new(Library::new(path.as_ref())?);

        unsafe fn load_sym<T: Copy>(lib: &Library, name: &[u8]) -> Result<T, libloading::Error> {
            let sym: libloading::Symbol<T> = lib.get(name)?;
            Ok(*sym)
        }

        Ok(Self {
            _lib: lib.clone(),

            litert_lm_set_min_log_level: load_sym(&lib, b"litert_lm_set_min_log_level\0")?,

            litert_lm_engine_settings_create: load_sym(&lib, b"litert_lm_engine_settings_create\0")?,
            litert_lm_engine_settings_delete: load_sym(&lib, b"litert_lm_engine_settings_delete\0")?,
            litert_lm_engine_settings_set_max_num_tokens: load_sym(
                &lib,
                b"litert_lm_engine_settings_set_max_num_tokens\0",
            )?,

            litert_lm_engine_create: load_sym(&lib, b"litert_lm_engine_create\0")?,
            litert_lm_engine_delete: load_sym(&lib, b"litert_lm_engine_delete\0")?,

            litert_lm_session_config_create: load_sym(&lib, b"litert_lm_session_config_create\0")?,
            litert_lm_session_config_set_max_output_tokens: load_sym(
                &lib,
                b"litert_lm_session_config_set_max_output_tokens\0",
            )?,
            litert_lm_session_config_set_sampler_params: load_sym(
                &lib,
                b"litert_lm_session_config_set_sampler_params\0",
            )?,
            litert_lm_session_config_delete: load_sym(&lib, b"litert_lm_session_config_delete\0")?,

            litert_lm_conversation_config_create: load_sym(
                &lib,
                b"litert_lm_conversation_config_create\0",
            )?,
            litert_lm_conversation_config_delete: load_sym(
                &lib,
                b"litert_lm_conversation_config_delete\0",
            )?,

            litert_lm_engine_create_session: load_sym(&lib, b"litert_lm_engine_create_session\0")?,
            litert_lm_session_delete: load_sym(&lib, b"litert_lm_session_delete\0")?,

            litert_lm_session_generate_content: load_sym(&lib, b"litert_lm_session_generate_content\0")?,
            litert_lm_session_generate_content_stream: load_sym(
                &lib,
                b"litert_lm_session_generate_content_stream\0",
            )?,

            litert_lm_responses_delete: load_sym(&lib, b"litert_lm_responses_delete\0")?,
            litert_lm_responses_get_num_candidates: load_sym(
                &lib,
                b"litert_lm_responses_get_num_candidates\0",
            )?,
            litert_lm_responses_get_response_text_at: load_sym(
                &lib,
                b"litert_lm_responses_get_response_text_at\0",
            )?,

            litert_lm_conversation_create: load_sym(&lib, b"litert_lm_conversation_create\0")?,
            litert_lm_conversation_delete: load_sym(&lib, b"litert_lm_conversation_delete\0")?,
            litert_lm_conversation_send_message: load_sym(
                &lib,
                b"litert_lm_conversation_send_message\0",
            )?,
            litert_lm_json_response_delete: load_sym(&lib, b"litert_lm_json_response_delete\0")?,
            litert_lm_json_response_get_string: load_sym(
                &lib,
                b"litert_lm_json_response_get_string\0",
            )?,
        })
    }
}

