## LiteRT-LM Rust POC (Linux)

This is a small Cargo workspace that calls LiteRT-LM via its existing C API (`c/engine.h`) by **dynamically loading** the Bazel-built shared library.

### Build the native `.so` (Linux x86_64)

This repo pins Bazel 7.6.1; if your system `bazel` is older, use the repo-local `bazelisk` already downloaded in the root.

Build the C API shared library:

```bash
BAZELISK_HOME="$PWD/.bazelisk-cache" XDG_CACHE_HOME="$PWD/.bazelisk-cache" \
  ./bazelisk build -c opt \
  --define=LITERT_LM_FST_CONSTRAINTS_DISABLED=1 \
  --define=xnn_enable_avx512fp16=false \
  --define=xnn_enable_avxvnniint8=false \
  --copt=-fpermissive --cxxopt=-fpermissive \
  --action_env=CC=gcc --action_env=CXX=g++ \
  --host_action_env=CC=gcc --host_action_env=CXX=g++ \
  --repo_env=CC=gcc --repo_env=CXX=g++ \
  //c:litert_lm_c_api.so
```

Optionally, stage a runnable bundle (shared lib + prebuilt deps in one directory):

```bash
./bazelisk build -c opt \
  --define=LITERT_LM_FST_CONSTRAINTS_DISABLED=1 \
  --define=xnn_enable_avx512fp16=false \
  --define=xnn_enable_avxvnniint8=false \
  --copt=-fpermissive --cxxopt=-fpermissive \
  --action_env=CC=gcc --action_env=CXX=g++ \
  --host_action_env=CC=gcc --host_action_env=CXX=g++ \
  --repo_env=CC=gcc --repo_env=CXX=g++ \
  //c:litert_lm_c_api_bundle_linux
```

### Run the Rust POC

From `rust_poc/`:

```bash
cargo run -p poc -- \
  --so /path/to/litert_lm_c_api.so \
  --model /path/to/model.litertlm \
  --max-tokens <max-tokens> \
  --min-log-level 4
```

You can also set env vars:

```bash
export LITERT_LM_SO=/path/to/litert_lm_c_api.so
export LITERT_LM_MODEL=/path/to/model.litertlm
```

