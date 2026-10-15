#!/usr/bin/env python
"""
test_interpreter.py — test the interpreter page's server side (/interpreter-ws)
without starting the whole server:

  doctor_text -> candidates -> play {id} -> audio_url, and the safety rules:
    (a) nothing close enough  -> no candidates, "sentence not found", no audio
    (b) candidates never carry audio; audio exists only after play {id}
    (c) a sentence with a number or a dose is marked for a deliberate pick,
        and a different number is flagged
    (d) the threshold is the one calibrate_matcher.py measured

  quick bar: list_quick -> the sentences chosen most often, then the starters
  of pattani_dataset/quick_sentences.json. The avatar page has one bar for
  each way of translating: Thai sentences said in Pattani, and (for the
  Pattani -> Thai translator) Pattani sentences said in Thai.

  patient screen: patient_text -> patient_translation {id, heard, thai,
  audio_url, online}; without internet the text still comes, without audio;
  rate {id, rating} (test mode) appends to the ratings file.

Run:  uv run python test_interpreter.py            typed text only
      uv run python test_interpreter.py --speech   also spoken Thai and Malay
                                                   (needs the GPU and Ollama)
Prerequisites: uv run python build_phrasebook.py && uv run python calibrate_matcher.py
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
os.environ.setdefault("HF_HOME", str(ROOT / "models"))
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.open_llm_vtuber.interpreter import service  # noqa: E402
from src.open_llm_vtuber.interpreter.sentence_bank import NOT_FOUND  # noqa: E402
from src.open_llm_vtuber.routes import init_interpreter_route  # noqa: E402

failures: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"   {'OK  ' if ok else 'FAIL'} {what}")
    if not ok:
        failures.append(what)


def wav_base64(path: Path) -> str:
    """A recording as the page sends it: 16 kHz mono 16-bit WAV, base64."""
    import io

    import soundfile as sf
    from faster_whisper.audio import decode_audio

    buffer = io.BytesIO()
    sf.write(
        buffer,
        decode_audio(str(path), sampling_rate=16000),
        16000,
        format="WAV",
        subtype="PCM_16",
    )
    return base64.b64encode(buffer.getvalue()).decode()


def speak(text: str, voice: str, path: Path) -> bool:
    import asyncio

    import edge_tts

    for _ in range(5):
        if path.exists() and path.stat().st_size > 1500:
            return True
        try:
            asyncio.run(edge_tts.Communicate(text, voice).save(str(path)))
        except Exception:
            pass
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech", action="store_true", help="also test spoken input")
    args = ap.parse_args()

    clips = ROOT / "cache" / "interpreter_test"
    clips.mkdir(parents=True, exist_ok=True)
    # keep what the test chooses out of the real quick bar (logs/)
    service.USAGE_FILE = clips / "usage.json"
    service.USAGE_FILE.unlink(missing_ok=True)
    quick_file = json.loads(
        (ROOT / "pattani_dataset" / "quick_sentences.json").read_text("utf-8")
    )
    starters, patient_starters = quick_file["thai_pattani"], quick_file["pattani_thai"]

    app = FastAPI()
    app.include_router(init_interpreter_route())
    threshold = json.loads(
        (ROOT / "pattani_dataset" / "matcher_threshold.json").read_text(
            encoding="utf-8"
        )
    )["threshold"]

    with TestClient(app).websocket_connect("/interpreter-ws") as ws:

        def ask(**message) -> dict:
            ws.send_json(message)
            return ws.receive_json()

        def listed_all(category: str) -> list[dict]:
            return ask(type="list_sentences", category=category)["sentences"]

        print("[1] categories and sentences (input method 1: no speech recognition)")
        categories = ask(type="list_categories")["categories"]
        check(len(categories) > 10, f"{len(categories)} categories listed")
        listed = ask(type="list_sentences", category=categories[0]["name"])["sentences"]
        check(
            len(listed) == categories[0]["count"],
            f"'{categories[0]['name']}' lists its {len(listed)} sentences",
        )

        print("[1b] quick bar: starters first, then what is chosen most often")
        quick = ask(type="list_quick")["sentences"]
        check(
            [q["thai"] for q in quick] == starters and len(starters) == 12,
            f"unused: the {len(quick)} starter sentences, all of them in the bank",
        )
        check(
            all(q["count"] == 0 and "pattani" not in q for q in quick),
            "quick sentences carry no audio or Pattani: playing still needs a pick",
        )
        dose = next(
            s for s in listed_all("คำศัพท์พื้นฐานเกี่ยวกับยา") if s["thai"] == "วันละสามครั้ง"
        )
        for sentence_id in (dose["id"], dose["id"], quick[5]["id"]):
            ask(type="play", id=sentence_id)
        after = ask(type="list_quick")["sentences"]
        check(
            [(q["thai"], q["count"]) for q in after[:2]]
            == [("วันละสามครั้ง", 2), (quick[5]["thai"], 1)]
            and len(after) == 12
            and [q["thai"] for q in after[2:]]
            == [t for t in starters if t != quick[5]["thai"]][:10],
            f"after use: {[(q['thai'], q['count']) for q in after[:3]]} ..., starters follow",
        )
        check(
            after[0]["check_numbers"] is True,
            "a sentence with a number keeps its warning in the quick bar",
        )
        saved = json.loads(service.USAGE_FILE.read_text("utf-8"))
        check(
            saved
            == {
                "thai_pattani": {"วันละสามครั้ง": 2, quick[5]["thai"]: 1},
                "pattani_thai": {},
            },
            f"choices saved for the next start: {saved}",
        )
        # the avatar page says a quick phrase through the same service
        mouth = next(q for q in after if q["thai"] == "อ้าปาก")
        phrase = service.shared_service().quick_phrase(mouth["id"])
        check(
            phrase is not None
            and phrase["pattani"] == ask(type="play", id=mouth["id"])["pattani"]
            and phrase["spoken"] == phrase["pattani"].replace("-", "")
            and phrase["latin"] == "nga-ngo",
            f"avatar page: 'อ้าปาก' -> the bank's own Pattani {phrase and phrase['pattani']!r}, not a translation",
        )
        check(
            phrase["said"] == "อ้าปาก"
            and phrase["spoken"] == "งางอ"
            and phrase["display"] == "งางอ \nnga-ngo",
            "Thai -> Pattani: the Thai is what the user said, the Pattani is spoken",
        )
        check(
            service.shared_service().quick_phrase("no-such-id") is None,
            "avatar page: an unknown quick phrase is refused",
        )

        print("[1c] quick bar of the Pattani -> Thai translator: Pattani sentences")
        shared = service.shared_service()
        patient = shared.quick(service.PATTANI_THAI)
        bank = shared.bank.sentences
        check(
            patient["direction"] == "pattani_thai"
            and [q["thai"] for q in patient["sentences"]] == patient_starters
            and len(patient_starters) == 12,
            f"its own {len(patient['sentences'])} starter sentences, all of them in the bank",
        )
        check(
            all(
                q["label"] == bank[q["id"]].pattani.replace("-", "")
                and q["label"] != q["thai"]
                for q in patient["sentences"]
            ),
            f"shown in Pattani, as the bank has it: {[q['label'] for q in patient['sentences'][:4]]} ...",
        )
        check(
            all(q["label"] == q["thai"] for q in shared.quick()["sentences"]),
            "the Thai -> Pattani bar is shown in Thai",
        )
        head = patient["sentences"][0]
        phrase = shared.quick_phrase(head["id"], service.PATTANI_THAI)
        check(
            phrase["said"] == head["label"]
            and phrase["spoken"] == phrase["display"] == "ปวดหัว",
            f"Pattani -> Thai: {phrase['said']!r} is what the user said, the Thai {phrase['spoken']!r} is spoken",
        )
        saved = json.loads(service.USAGE_FILE.read_text("utf-8"))
        check(
            saved["pattani_thai"] == {"ปวดหัว": 1}
            and "ปวดหัว" not in saved["thai_pattani"]
            and shared.quick(service.PATTANI_THAI)["sentences"][0]["count"] == 1
            and all(q["thai"] != "ปวดหัว" for q in shared.quick()["sentences"]),
            "each bar keeps its own counts",
        )
        old_usage = shared._by_direction({"สวัสดี": 3, "กินยา": 1}, dict)
        old_starters = shared._by_direction(["สวัสดี"], list)
        check(
            old_usage == {"thai_pattani": {"สวัสดี": 3, "กินยา": 1}, "pattani_thai": {}}
            and old_starters == {"thai_pattani": ["สวัสดี"], "pattani_thai": []},
            "counts and starters saved before there were two bars are kept as the Thai -> Pattani ones",
        )

        class Session:  # as much of a session as quick_direction looks at
            def __init__(self, path, translating=True):
                self.character_config = shared._character(path)
                self.translate_engine = object() if translating else None

        check(
            service.quick_direction(Session(service.PATIENT_CHARACTER))
            == "pattani_thai"
            and service.quick_direction(Session(service.DOCTOR_CHARACTER))
            == "thai_pattani"
            and service.quick_direction(Session(service.PATIENT_CHARACTER, False))
            == "thai_pattani",
            "the bar is chosen by the character: Pattani -> Thai translator gets the Pattani bar, the others the Thai bar",
        )

        print("[2] typed Thai -> candidates -> play")
        reply = ask(type="doctor_text", text="ยื่นแขนซ้ายมาครับ", request=7)
        top = reply["candidates"][0] if reply["candidates"] else {}
        check(
            reply["type"] == "candidates" and reply.get("request") == 7,
            "answer is 'candidates' for the same request",
        )
        check(
            top.get("thai") == "ยื่นแขนซ้ายมา",
            f"best candidate: {top.get('thai')!r} ({top.get('score')})",
        )
        check(
            1 <= len(reply["candidates"]) <= 3,
            f"{len(reply['candidates'])} candidates offered (1-3)",
        )
        check(
            all(
                "audio_url" not in c and "pattani" not in c for c in reply["candidates"]
            ),
            "candidates carry no audio: nothing plays before a pick",
        )
        check(
            all(c["score"] >= threshold for c in reply["candidates"]),
            f"all candidates at or above the measured threshold {threshold}",
        )
        played = ask(type="play", id=top.get("id", ""))
        check(
            played["type"] == "audio_url" and played["thai"] == "ยื่นแขนซ้ายมา",
            f"play -> audio_url for {played.get('thai')!r}",
        )
        check(
            "กิรี" in played.get("pattani", ""),
            f"Pattani text is the word bank's: {played.get('pattani')!r}",
        )
        url = played.get("audio_url") or ""
        check(
            url.startswith("/cache/")
            and (ROOT / url.lstrip("/")).stat().st_size > 1000,
            f"audio file exists at {url}",
        )

        print("[3] safety rule (a): not in the bank -> not found, no candidates")
        for text in ("ห้องน้ำอยู่ทางไหน", "ต้องผ่าตัดพรุ่งนี้เช้า", "วันนี้อากาศดีมาก"):
            reply = ask(type="doctor_text", text=text)
            check(
                reply["candidates"] == [] and reply.get("message") == NOT_FOUND,
                f"{text!r} -> {reply.get('message')!r}",
            )

        print("[4] safety rule (c): numbers and doses")
        reply = ask(type="doctor_text", text="วันละสามครั้ง")
        top = reply["candidates"][0] if reply["candidates"] else {}
        check(
            top.get("thai") == "วันละสามครั้ง" and top.get("check_numbers") is True,
            "'วันละสามครั้ง' is found and marked for a deliberate pick",
        )
        check(
            top.get("number_mismatch") is False,
            "... and its number is the one that was said",
        )
        reply = ask(type="doctor_text", text="วันละสองครั้ง")
        check(
            all(
                c["number_mismatch"] and c["check_numbers"] for c in reply["candidates"]
            ),
            f"'วันละสองครั้ง' (not in the bank): {len(reply['candidates'])} candidates, every one flagged as a different number",
        )
        check(
            ask(type="play", id="no-such-id")["type"] == "error",
            "play with an unknown id is an error, not audio",
        )

        print("[5] patient screen: translation, network status, test-mode rating")
        ratings = clips / "ratings.jsonl"
        ratings.unlink(missing_ok=True)
        service.RATINGS_FILE = ratings  # keep test ratings out of logs/
        online = ask(type="network_status")
        check(
            online["type"] == "network_status" and isinstance(online["online"], bool),
            f"network_status -> online={online.get('online')}",
        )
        reply = ask(type="patient_text", text="Sakit perut", request="p1")
        check(
            reply["type"] == "patient_translation"
            and reply.get("request") == "p1"
            and reply.get("heard") == "Sakit perut"
            and reply.get("thai") == "ปวดท้อง",
            f"'Sakit perut' -> {reply.get('thai')!r}, with what was heard",
        )
        check(
            bool(reply.get("audio_url")) == online["online"]
            and reply.get("online") == online["online"],
            f"online={reply.get('online')}: audio {reply.get('audio_url')}",
        )
        rated = ask(type="rate", id=reply.get("id", ""), rating="minor_error")
        rated = ask(type="rate", id=reply.get("id", ""), rating="correct")
        rows = [json.loads(line) for line in ratings.read_text("utf-8").splitlines()]
        check(
            rated == {"type": "rated", "id": reply.get("id"), "rating": "correct"}
            and [r["rating"] for r in rows] == ["minor_error", "correct"]
            and rows[-1]["heard"] == "Sakit perut"
            and rows[-1]["thai"] == "ปวดท้อง"
            and rows[-1]["source"] == "typed",
            f"rating saved with the Malay and the Thai; re-rating adds a line ({len(rows)} lines)",
        )
        check(
            ask(type="rate", id=reply.get("id", ""), rating="perfect")["type"]
            == "error"
            and ask(type="rate", id="no-such-id", rating="correct")["type"] == "error"
            and len(ratings.read_text("utf-8").splitlines()) == 2,
            "an unknown rating or translation id is refused and not saved",
        )
        # no internet: nothing answers on this port
        reachable, trusted = service.VOICE_HOST, service.NETWORK_CHECK_SECONDS
        service.VOICE_HOST, service.NETWORK_CHECK_SECONDS = ("127.0.0.1", 9), 0
        offline = ask(type="network_status")
        reply = ask(type="patient_text", text="Rasa loya")
        check(
            offline["online"] is False
            and reply.get("thai") == "คลื่นไส้"
            and reply.get("audio_url") is None
            and reply.get("online") is False,
            f"without internet: text only ({reply.get('thai')!r}, audio {reply.get('audio_url')})",
        )
        service.VOICE_HOST, service.NETWORK_CHECK_SECONDS = reachable, trusted

        if args.speech:
            print("[6] spoken Thai -> heard -> candidates")
            for n, (said, wanted) in enumerate(
                [
                    ("ช่วยอ้าปากหน่อยครับ", "อ้าปาก"),
                    ("หายใจเข้าลึกๆ", "หายใจเข้าลึกๆ"),
                    ("ทานยานี้หลังอาหารนะคะ", "ทานยานี้หลังอาหาร"),
                ]
            ):
                clip = clips / f"doctor_{n}.mp3"
                if not speak(said, "th-TH-NiwatNeural", clip):
                    check(False, f"could not make a recording of {said!r}")
                    continue
                reply = ask(type="doctor_audio", audio=wav_base64(clip))
                thais = [c["thai"] for c in reply["candidates"]]
                check(
                    wanted in thais,
                    f"said {said!r}, heard {reply.get('heard')!r} -> {thais}",
                )
            print("[7] patient speaking: Malay speech -> Thai (machine translation)")
            clip = clips / "patient_0.mp3"
            if speak("Tekan di sini sebentar", "ms-MY-YasminNeural", clip):
                reply = ask(type="patient_audio", audio=wav_base64(clip))
                check(
                    reply["type"] == "patient_translation"
                    and reply.get("thai") == "กดแผลตรงนี้ไว้สักครู่",
                    f"heard {reply.get('heard')!r} -> {reply.get('thai')!r}",
                )
                check(
                    bool(reply.get("audio_url")),
                    f"Thai audio at {reply.get('audio_url')}",
                )

    if failures:
        print(f"\n❌ {len(failures)} check(s) failed", file=sys.stderr)
        return 1
    print("\n==================== INTERPRETER TEST PASSED ====================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
