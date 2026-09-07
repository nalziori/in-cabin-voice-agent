# CLAUDE.md — 작업 지침

프로젝트 소개와 설계 근거는 `README.md`에 있다. 이 파일은 **코드를 고칠 때 지켜야 할 것**만 적는다.
지원 맥락·마감·이력서 관련 메모는 `NOTES.private.md`(git 추적 제외)에 있다.

## 전제

실제 차량에 연결되지 않은 목업이지만 **실제 차량에서 동작한다고 가정하고** 설계한다.
도구 스키마·센서 값을 그럴듯하게 지어내지 말고 실제 차량이 노출하는 형태에 맞출 것.

**온디바이스는 가정이다.** 의도 추론은 클라우드 API를 호출하며 경량 모델의 스탠드인이고,
**교체 지점은 `parse_intent()` 하나**다. 이 함수만 로컬 모델로 바꾸면 나머지는 그대로 돈다.
따라서 다른 계층에 API 의존을 새로 만들지 않는다.

음성은 실제로 로컬에서만 처리된다(faster-whisper). 이건 가정이 아니라 구현이므로 깨지 말 것.

## 지켜야 할 불변 조건

- **모든 차량 제어는 `_run()` 한 곳만 통과한다.** 게이트를 우회하는 별도 실행 경로를 만들지 않는다.
- **확인이 필요한 동작의 실행 인자는 `pending`에서만 온다.** 모델이 만든 값으로 확정 실행하지 않는다.
  모델은 `confirm: yes`까지만 말할 수 있다.
- **되돌릴 수 없거나 외부로 나가는 동작**(전화·메시지)과 **주행 중 자세 변경**(등받이·슬라이드)은 확인을 거친다.
- **확인이 만능이 아니다.** `safety_gate()`의 `"refuse"`는 `confirmed=True`로도 열리지 않는다.
  확인만으로 전부 열 수 있으면 안전 판단이 "응"이라고 말하는 승객에게 넘어간다.
- **확인에는 유효기간이 있고**(`CONFIRM_TTL_S`), **묻는 사이 상태가 바뀌면 담아 둔 인자를 버린다**
  (`_state_key()`). 상대값은 절대값으로 굳어서 pending에 담기므로 시간이 지나면 뜻이 달라진다.
- **ASR도 신뢰 경계다.** 확신이 낮은 전사는 텍스트를 넘기지 않고 되묻는다 — `LIMITS`는 범위 안의
  오독("3도"→"8도")을 못 잡는다.
- **실행 실패를 성공이라고 답하지 않는다.** 상태 변경은 `_apply()`로 분리돼 있고 `_run()`이 감싼다.
- **값이 없으면 추측하지 않고 되묻는다.** 방향만 있는 발화("조명 좀 어둡게")도 실행하지 않는다.
  평가 세트에서 "호출하지 않는 게 정답"인 케이스를 빼지 않는다.
- **모델 출력은 신뢰 경계다.** 새 숫자 인자를 추가하면 `LIMITS`에도 범위를 추가한다.
- **발화에 섞인 지시문은 데이터로 취급한다.**
- **holdout 케이스는 튜닝에 쓰지 않는다.**
- **센서 스냅샷은 user 메시지에 붙인다.** 시스템 프롬프트는 고정 — 매 요청 바뀌는 값을 system에
  넣으면 프롬프트 캐시가 매번 깨진다.
- **발화 캐시(`_INTENT_CACHE`)는 `pending`이 없을 때만 쓴다.** "3도"·"네" 같은 문맥 의존 발화는
  직전 질문에 따라 뜻이 달라지므로, pending이 있으면 같은 문자열이어도 캐시를 보지 않는다.

## 재현성

이 프로젝트의 숫자는 다시 돌려서 같은 값이 나와야 한다.

- 클린 체크아웃에서 `pip install -r requirements.txt` 후 바로 동작해야 한다.
- **`--selftest`는 API 키 없이 항상 통과해야 한다.** 게이트·해석·대화흐름·채점이 깨지면 여기서 잡힌다.
  로직을 고치면 selftest도 같이 고친다.
- 평가 케이스마다 `reset_vehicle()`로 시작 상태를 고정한다. 상대값의 정답이 시작 상태에 의존하므로
  이건 정돈이 아니라 정확성 문제다.
- 수치를 인용할 때는 그 수치를 만든 명령과 표본 크기를 함께 적는다.
- **실측 전 숫자를 문서에 먼저 쓰지 않는다.**

테스트용 음성 파일은 저장소에 넣지 않는다(`.gitignore`). Windows에서 재생성:

```powershell
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice('Microsoft Heami Desktop'); $s.SetOutputToWaveFile('sample.wav')
$s.Speak('에어컨 온도 좀 높여줘'); $s.Dispose()
```

## 코드 지도

| 위치 | 역할 |
|---|---|
| `_VEHICLE_INIT` / `VEHICLE` / `reset_vehicle()` | 차량 상태 + 센서 목업 |
| `LIMITS` | 값 범위. 모델 출력을 자르는 곳 |
| `transcribe()` | [0] 로컬 ASR. 오디오는 이 함수 밖으로 나가지 않는다. `(텍스트, 오류)` 반환 — 확신이 낮으면 텍스트를 안 넘긴다 |
| `Intent` / `SYSTEM` / `parse_intent()` / `_INTENT_CACHE` | [1] 의도 추론. **LLM 유일 지점** + 동일 발화 캐시 |
| `safety_gate()` / `_run()` / `_apply()` | [2][3] 3단계 게이트(통과·확인·거부)와 단일 실행 경로. 상태 변경은 `_apply()`가 하고 `_run()`이 실패를 잡는다 |
| `CONFIRM_TTL_S` / `_state_key()` | 확인의 유효기간과 실행 직전 상태 재검사 |
| `_resolve()` | Intent → (기능, 인자). 상대값 적용 + 범위 클램프 |
| `dispatch()` | 되묻기·확인·실행 분기 |
| `_answer()` / `_describe()` | [4] 응답 문장 템플릿 |
| `CASES` / `score()` / `run_eval()` | 평가 |
| `selftest()` | API 없이 도는 검증 |

## 명령어

```bash
python car_agent.py --selftest                        # 키 없이 로직 검증
python car_agent.py --say "에어컨 22도로"
python car_agent.py --say "온도 좀 높여줘" --say "3도"   # --say는 누적, 대화가 이어진다
python car_agent.py --listen sample.wav
python car_agent.py --eval [--split tune|holdout]
```

환경변수: `CAR_AGENT_MODEL`(기본 `claude-opus-5`), `CAR_AGENT_EFFORT`(기본 `low`),
`CAR_AGENT_ASR`(기본 `base`).

## 남은 일

- **relative 오분류 별도 집계** — "3도 올려" 류 상대값 케이스가 절대값으로 잘못 해석되는지.
  tune·holdout 각 1건씩만 있어 표본이 작다.
- 이번 세션 실측치(정확도·지연·비용·캐시 적중)를 이력서·포트폴리오에 반영 — `NOTES.private.md` 참고.
