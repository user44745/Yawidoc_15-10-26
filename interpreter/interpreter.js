// Interpreter page: patient speaking (one button, machine translation into
// Thai) / doctor speaking (standard sentences). Talks to /interpreter-ws; see
// init_interpreter_route(). Open with ?test=1 for test mode (rating section).

const $ = (id) => document.getElementById(id);
const player = $("player");
const TEST_MODE = new URLSearchParams(location.search).get("test") === "1";
const NETWORK_POLL_MS = 15000;
const TRANSLATION_TIMEOUT_MS = 120000;  // the first request also loads the models
let socket = null;
let lastRequest = 0;       // answers to older requests are dropped
let currentAudio = null;   // url of the sentence on screen

function setStatus(text) { $("status").textContent = text; }
function connected() { return socket !== null && socket.readyState === WebSocket.OPEN; }

function connect() {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${scheme}://${location.host}/interpreter-ws`);
    socket.onopen = () => {
        setStatus("เชื่อมต่อแล้ว");
        send({ type: "network_status" });
        send({ type: "list_categories" });
        send({ type: "list_quick" });
        if (patientState !== "listening") setPatientState("ready");
    };
    socket.onclose = () => {
        setStatus("ขาดการเชื่อมต่อ กำลังลองใหม่…");
        setNetwork(null);
        // a translation that was on its way is lost with the connection
        if (patientState !== "listening") patientFailed("ขาดการเชื่อมต่อกับระบบ");
        setTimeout(connect, 2000);
    };
    socket.onmessage = (event) => handle(JSON.parse(event.data));
}

function send(message) {
    if (connected()) {
        socket.send(JSON.stringify(message));
        return true;
    }
    setStatus("ยังไม่ได้เชื่อมต่อ");
    return false;
}

function handle(message) {
    if (message.type === "categories") showCategories(message.categories);
    else if (message.type === "sentences") showSentences(message.sentences);
    else if (message.type === "quick") showQuick(message.sentences);
    else if (message.type === "candidates") {
        if (message.request !== undefined && message.request !== lastRequest) return;
        showCandidates(message);
    }
    else if (message.type === "audio_url") showSelected(message);
    else if (message.type === "patient_translation") showPatient(message);
    else if (message.type === "network_status") setNetwork(message.online);
    else if (message.type === "rated") showRated(message);
    else if (message.type === "error") {
        if (typeof message.request === "string") {
            // a patient request ("p3"); an old one was already given up
            if (message.request === patientRequestId()) patientFailed("แปลไม่สำเร็จ ลองอีกครั้ง");
        } else {
            setStatus("ผิดพลาด: " + message.message);
        }
    }
}

// ---------- network: the voice needs the internet, the text does not ----------

function setNetwork(online) {
    const pill = $("network");
    pill.classList.toggle("online", online === true);
    pill.classList.toggle("offline", online === false);
    pill.textContent = online === true ? "ออนไลน์"
        : online === false ? "ไม่มีอินเทอร์เน็ต: แสดงข้อความเท่านั้น"
        : "ไม่ทราบสถานะอินเทอร์เน็ต";
}
setInterval(() => { if (connected()) send({ type: "network_status" }); }, NETWORK_POLL_MS);

// ---------- audio ----------

function play(url) {
    if (!url) return;
    player.src = url;
    player.play().catch(() => setStatus("กดเล่นซ้ำเพื่อฟังเสียง"));
}

const recorder = new AudioRecorder();
async function blobToBase64(blob) {
    const bytes = new Uint8Array(await blob.arrayBuffer());
    let binary = "";
    for (let i = 0; i < bytes.length; i += 0x8000) {
        binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    }
    return btoa(binary);
}

// The recorder never answers when the browser cannot decode what it recorded.
function finishRecording() {
    return Promise.race([
        recorder.stop(),
        new Promise((_, reject) => setTimeout(() => reject(new Error("recording timed out")), 10000)),
    ]);
}

// Stop a recording that is no longer wanted (screen cleared, speaker changed).
async function abandonRecording() {
    if (!recorder.isActive()) return;
    try { await finishRecording(); } catch (error) { console.error(error); }
}

// ---------- patient speaking ----------

let patientState = "ready";      // ready | listening | translating | error
let patientRequest = 0;
let patientTimer = null;
let patientAudio = null;
let patientTranslation = null;   // id of the translation on screen
let talkBusy = false;            // between a press and the microphone answering
const patientRequestId = () => "p" + patientRequest;
const STATE_TEXT = { ready: "พร้อม", listening: "กำลังฟัง", translating: "กำลังแปล" };

function setPatientState(state, message) {
    patientState = state;
    const pill = $("patient-state");
    pill.className = "state " + state;
    pill.textContent = state === "error" ? message : STATE_TEXT[state];
    const button = $("talk");
    const hint = document.createElement("small");
    button.classList.toggle("recording", state === "listening");
    if (state === "listening") {
        button.textContent = "■ กดเพื่อหยุด";
        hint.textContent = "กำลังอัดเสียง พูดได้เลย";
    } else if (state === "translating") {
        button.textContent = "กำลังแปล…";
        hint.textContent = "โปรดรอสักครู่";
    } else {
        button.textContent = "🎤 กดเพื่อพูด";
        hint.textContent = "กดอีกครั้งเมื่อพูดจบ";
    }
    button.appendChild(hint);
    button.disabled = state === "translating" || (state !== "listening" && !connected());
}

function patientFailed(message) {
    clearTimeout(patientTimer);
    patientRequest += 1;   // a late answer to the failed request is dropped
    setPatientState("error", message);
}

function waitForTranslation() {
    clearTimeout(patientTimer);
    patientTimer = setTimeout(() => patientFailed("ใช้เวลานานเกินไป ลองอีกครั้ง"), TRANSLATION_TIMEOUT_MS);
}

$("talk").onclick = async () => {
    if (talkBusy || patientState === "translating") return;
    talkBusy = true;
    try {
        if (patientState !== "listening") {
            player.pause();
            try {
                await recorder.start();
            } catch (error) {
                setPatientState("error", "ใช้ไมโครโฟนไม่ได้");
                return;
            }
            setPatientState("listening");
            return;
        }
        setPatientState("translating");
        let wav;
        try {
            wav = await finishRecording();
        } catch (error) {
            patientFailed("อัดเสียงไม่สำเร็จ ลองอีกครั้ง");
            return;
        }
        patientRequest += 1;
        const sent = send({ type: "patient_audio", audio: await blobToBase64(wav), request: patientRequestId() });
        if (sent) waitForTranslation();
        else patientFailed("ขาดการเชื่อมต่อกับระบบ");
    } finally {
        talkBusy = false;
    }
};

function showPatient(message) {
    if (message.request !== undefined && message.request !== patientRequestId()) return;
    clearTimeout(patientTimer);
    setNetwork(message.online);
    const thai = message.thai || "";
    $("patient-heard").textContent = message.heard || "";
    $("patient-thai").textContent = thai;
    $("patient-thai").classList.toggle("long", thai.length > 40);
    patientAudio = message.audio_url || null;
    patientTranslation = message.id || null;
    $("patient-replay").disabled = !patientAudio;
    $("patient-note").textContent = !thai || patientAudio ? ""
        : message.online === false ? "ไม่มีอินเทอร์เน็ต: แสดงข้อความเท่านั้น"
        : "สร้างเสียงไม่สำเร็จ — อ่านข้อความแทน";
    showRating(Boolean(patientTranslation));
    if (message.heard) setPatientState("ready");
    else setPatientState("error", "ไม่ได้ยินเสียงพูด ลองอีกครั้ง");
    play(patientAudio);
}

async function clearPatient() {
    patientRequest += 1;   // an answer still on its way is not shown
    clearTimeout(patientTimer);
    player.pause();
    await abandonRecording();
    $("patient-heard").textContent = "";
    $("patient-thai").textContent = "";
    $("patient-thai").classList.remove("long");
    $("patient-note").textContent = "";
    patientAudio = null;
    patientTranslation = null;
    $("patient-replay").disabled = true;
    showRating(false);
    if (connected()) setPatientState("ready");
    else setPatientState("error", "ขาดการเชื่อมต่อกับระบบ");
}
$("patient-clear").onclick = clearPatient;
$("patient-replay").onclick = () => play(patientAudio);

// ---------- test mode: rate the translation on screen ----------

const ratingButtons = [...document.querySelectorAll("#rating [data-rating]")];
function showRating(visible) {
    $("rating").hidden = !(TEST_MODE && visible);
    for (const button of ratingButtons) button.setAttribute("aria-pressed", "false");
    $("rating-note").textContent = "";
}
for (const button of ratingButtons) {
    button.onclick = () => {
        if (!patientTranslation) return;
        if (!send({ type: "rate", id: patientTranslation, rating: button.dataset.rating })) {
            $("rating-note").textContent = "บันทึกไม่ได้: ขาดการเชื่อมต่อ";
        }
    };
}
function showRated(message) {
    if (message.id !== patientTranslation) return;
    for (const button of ratingButtons) {
        button.setAttribute("aria-pressed", String(button.dataset.rating === message.rating));
    }
    $("rating-note").textContent = "บันทึกแล้ว (กดปุ่มอื่นเพื่อแก้ไข)";
}
function sendPatientText() {
    const text = $("patient-text").value.trim();
    if (!text || patientState === "translating" || patientState === "listening") return;
    patientRequest += 1;
    if (send({ type: "patient_text", text, request: patientRequestId() })) {
        setPatientState("translating");
        waitForTranslation();
    } else {
        patientFailed("ขาดการเชื่อมต่อกับระบบ");
    }
}
$("patient-send").onclick = sendPatientText;
$("patient-text").addEventListener("keydown", (event) => { if (event.key === "Enter") sendPatientText(); });
$("test-badge").hidden = !TEST_MODE;
$("patient-typing").hidden = !TEST_MODE;

// ---------- doctor speaking ----------

function showCategories(categories) {
    const box = $("categories");
    box.replaceChildren();
    for (const category of categories) {
        const button = document.createElement("button");
        button.textContent = `${category.name} (${category.count})`;
        button.setAttribute("aria-pressed", "false");
        button.onclick = () => {
            for (const other of box.children) other.setAttribute("aria-pressed", "false");
            button.setAttribute("aria-pressed", "true");
            send({ type: "list_sentences", category: category.name });
        };
        box.appendChild(button);
    }
}

// The doctor picks a sentence: playing always starts here. A sentence with
// a number or a dose asks once more before it plays.
function choose(sentence) {
    if (sentence.check_numbers || sentence.number_mismatch) {
        const ok = confirm(`ประโยคนี้มีตัวเลขหรือความถี่\n\n"${sentence.thai}"\n\nยืนยันว่าต้องการประโยคนี้?`);
        if (!ok) return;
    }
    setStatus("กำลังเตรียมเสียง…");
    send({ type: "play", id: sentence.id });
}

// Quick bar: the sentences chosen most often. It is filled when the doctor
// screen opens and stays as it is while in use, so buttons do not move.
function showQuick(sentences) {
    $("quick-card").hidden = !sentences.length;
    $("quick").replaceChildren(...sentences.map((sentence) => {
        const button = document.createElement("button");
        button.textContent = (sentence.check_numbers ? "⚠ " : "") + sentence.thai;
        button.classList.toggle("numbers", sentence.check_numbers);
        button.onclick = () => choose(sentence);
        return button;
    }));
}

function sentenceButton(sentence, extra) {
    const button = document.createElement("button");
    const text = document.createElement("span");
    text.textContent = sentence.thai;
    if (sentence.number_mismatch || sentence.check_numbers) {
        const flag = document.createElement("span");
        flag.className = "flag";
        flag.textContent = sentence.number_mismatch
            ? "⚠ ตัวเลขไม่ตรงกับที่พูด — ตรวจก่อนเลือก"
            : "⚠ มีตัวเลขหรือความถี่ — ตรวจให้ตรงก่อนเลือก";
        text.appendChild(flag);
    }
    button.appendChild(text);
    if (extra) {
        const score = document.createElement("span");
        score.className = "score";
        score.textContent = extra;
        button.appendChild(score);
    }
    button.onclick = () => choose(sentence);
    return button;
}

function showSentences(sentences) {
    $("sentences").replaceChildren(...sentences.map((s) => sentenceButton(s)));
}

function showCandidates(message) {
    const box = $("candidates");
    const heard = $("doctor-heard");
    heard.hidden = !message.heard;
    if (message.heard) heard.textContent = `ได้ยินว่า: ${message.heard}`;
    setStatus("เชื่อมต่อแล้ว");
    if (!message.candidates.length) {
        const none = document.createElement("div");
        none.className = "notfound";
        none.textContent = message.query ? message.message : "";
        box.replaceChildren(...(message.query ? [none] : []));
        return;
    }
    box.replaceChildren(...message.candidates.map(
        (c) => sentenceButton(c, `ตรงกัน ${Math.round(c.score * 100)}%`)));
}

function showSelected(message) {
    $("selected").hidden = false;
    $("selected-thai").textContent = message.thai;
    $("selected-pattani").textContent = message.pattani;
    $("selected-latin").textContent = message.latin;
    currentAudio = message.audio_url;
    $("replay").disabled = !currentAudio;
    $("audio-note").textContent = currentAudio ? "" : "ไม่มีเสียง — ให้ผู้ป่วยอ่านข้อความแทน";
    setStatus("เชื่อมต่อแล้ว");
    $("selected").scrollIntoView({ behavior: "smooth", block: "nearest" });
    play(currentAudio);   // the doctor has just chosen this sentence
}

let typing = null;
$("search").addEventListener("input", () => {
    clearTimeout(typing);
    typing = setTimeout(searchNow, 400);
});
$("search").addEventListener("keydown", (event) => {
    if (event.key === "Enter") { clearTimeout(typing); searchNow(); }
});
function searchNow() {
    const text = $("search").value.trim();
    lastRequest += 1;
    if (!text) { showCandidates({ candidates: [], query: "" }); return; }
    send({ type: "doctor_text", text, request: lastRequest });
}
$("replay").onclick = () => play(currentAudio);

const DOCTOR_MIC_LABEL = "🎤 พูด";
function resetDoctorMic() {
    $("doctor-mic").classList.remove("recording");
    $("doctor-mic").textContent = DOCTOR_MIC_LABEL;
    $("doctor-mic").disabled = false;
}
$("doctor-mic").onclick = async () => {
    const button = $("doctor-mic");
    if (!recorder.isActive()) {
        try {
            await recorder.start();
        } catch (error) {
            setStatus("ใช้ไมโครโฟนไม่ได้: " + error.message);
            return;
        }
        button.classList.add("recording");
        button.textContent = "■ หยุด";
        return;
    }
    button.disabled = true;
    let wav;
    try {
        wav = await finishRecording();
    } catch (error) {
        setStatus("อัดเสียงไม่สำเร็จ ลองอีกครั้ง");
        return;
    } finally {
        resetDoctorMic();
    }
    setStatus("กำลังฟัง…");
    lastRequest += 1;
    send({ type: "doctor_audio", audio: await blobToBase64(wav), request: lastRequest });
};

// ---------- who is speaking ----------

async function setMode(doctor) {
    $("mode-doctor").setAttribute("aria-pressed", String(doctor));
    $("mode-patient").setAttribute("aria-pressed", String(!doctor));
    $("doctor").hidden = !doctor;
    $("patient").hidden = doctor;
    player.pause();
    if (doctor) send({ type: "list_quick" });
    // a recording started for the other speaker is not sent
    await abandonRecording();
    resetDoctorMic();
    if (patientState === "listening") setPatientState(connected() ? "ready" : "error", "ขาดการเชื่อมต่อกับระบบ");
}
$("mode-doctor").onclick = () => setMode(true);
$("mode-patient").onclick = () => setMode(false);

setPatientState("ready");
connect();
