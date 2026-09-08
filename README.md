# In-Cabin Voice Agent

A safety-oriented prototype for controlling vehicle functions by voice. The project turns a spoken request into a structured intent, applies explicit safety checks, and executes only the actions that remain valid at the moment of execution.

This is not a general-purpose assistant for a car. It is an experiment in a narrower question: **how should a vehicle interface behave when a request is incomplete, uncertain, or unsafe?**

## What it does

- Transcribes speech locally with `faster-whisper`.
- Uses one structured-model call to identify an intent and its arguments.
- Supports climate, phone and message actions, seat and lighting settings, navigation, media, and cabin-sensor queries.
- Keeps execution, confirmation, and response generation in deterministic local code.

## Safety model

Every command passes through seven code-enforced checks before it can change vehicle state:

1. Reject low-confidence transcription and ask the user to repeat it.
2. Require all values needed to execute an action.
3. Clamp valid numeric values to defined operating limits.
4. Classify an action as allowed, confirmation-required, or refused.
5. Expire confirmations after a short time window.
6. Re-check relevant vehicle state immediately before execution.
7. Surface execution failures instead of reporting a false success.

The language model cannot approve its own action. A confirmation only unlocks locally stored pending state, and some conditions—such as unsafe seat movement at high speed—remain blocked even after confirmation.

## Architecture

```text
Audio → local ASR → intent extraction (one model call) → safety gate → deterministic execution → response
```

The model produces a typed intent; local code resolves relative values, owns pending state, applies policy, and changes simulated vehicle state. This separation keeps critical behavior testable and avoids treating model output as an executable command.

## Evaluation

The repository evaluates tool selection, argument resolution, safety-gate behavior, and correct non-execution. A 23-case suite is split into 14 tuning cases and 9 holdout cases; vehicle state is reset for every case.

The latest repeated run reports 100% on those four metrics across five repetitions of each split. This is a small prototype test set, not evidence of production-level reliability. The more durable result is the enforced control flow: missing values, unsafe requests, and untrusted instruction-like text cannot bypass the local gate.

## Run it

```bash
pip install -r requirements.txt

python car_agent.py --selftest
python car_agent.py --say "Set the air conditioning to 22 degrees"
python car_agent.py --listen sample.wav
python car_agent.py --eval --split holdout
```

`--selftest` runs without an API key. Other commands require `ANTHROPIC_API_KEY`. Model, reasoning effort, and ASR size are configurable with `CAR_AGENT_MODEL`, `CAR_AGENT_EFFORT`, and `CAR_AGENT_ASR`.

## Scope and limitations

The project simulates vehicle controls; it does not interface with a real vehicle. The evaluation set is intentionally small, and speech-recognition accuracy has not been measured at production scale. It is a portfolio prototype for safety boundaries and interaction design, not a deployable automotive system.
