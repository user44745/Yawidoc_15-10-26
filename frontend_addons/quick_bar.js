// Quick-phrase bar of the avatar page.
//
// A row of the standard sentences used most often. One press and the avatar
// says the sentence in the other language — as the sentence bank has it, not
// a translation. The bar follows the way the character translates:
//
//     Thai -> Pattani translator   Thai sentences, said in Pattani
//     Pattani -> Thai translator   Pattani sentences, said in Thai
//
// Every other character gets the Thai sentences; one whose own voice cannot
// read Thai script borrows the voice of the Thai -> Pattani translator.
//
// The frontend is a compiled bundle, so this is a script of its own, loaded
// by frontend/index.html before the app. It talks over the page's own
// connection to /client-ws:
//
//     fetch-quick-phrases  -> quick-phrases {direction, phrases: [{id, label,
//                             thai, check_numbers, count}]}
//     quick-phrase {id, direction}
//                          -> an ordinary conversation turn (text and audio)

(() => {
    const NativeWebSocket = window.WebSocket;
    const BUSY_TIMEOUT_MS = 45000;   // in case the end of a turn never arrives
    let socket = null;               // the page's open connection to /client-ws
    let phrases = [];
    let direction = "thai_pattani";  // which way the character translates
    const DIRECTION_TEXT = { thai_pattani: "ไทย → ปัตตานี", pattani_thai: "ปัตตานี → ไทย" };
    let busy = false;                // the avatar is answering
    let busyTimer = null;
    let bar = null;
    let row = null;
    let tag = null;                  // says which way the bar translates

    // ---------- the page's connection ----------

    class WatchedWebSocket extends NativeWebSocket {
        constructor(url, protocols) {
            if (protocols === undefined) super(url); else super(url, protocols);
            if (String(url).includes("/client-ws")) watch(this);
        }

        send(data) {
            // the user stopped the avatar: the bar can be used again
            if (this === socket && typeof data === "string" && data.includes('"interrupt-signal"')) setBusy(false);
            super.send(data);
        }
    }
    window.WebSocket = WatchedWebSocket;

    function watch(ws) {
        ws.addEventListener("open", () => { socket = ws; fetchPhrases(); });
        ws.addEventListener("close", () => {
            if (socket !== ws) return;
            socket = null;
            phrases = [];
            setBusy(false);
            render();
        });
        ws.addEventListener("message", (event) => {
            if (ws !== socket || typeof event.data !== "string") return;
            let message;
            try { message = JSON.parse(event.data); } catch (error) { return; }
            if (message.type === "quick-phrases") {
                phrases = Array.isArray(message.phrases) ? message.phrases : [];
                direction = message.direction === "pattani_thai" ? "pattani_thai" : "thai_pattani";
                render();
            } else if (message.type === "set-model-and-conf" || message.type === "config-switched") {
                fetchPhrases();   // the session was set up again
            } else if (message.type === "control" && message.text === "conversation-chain-start") {
                setBusy(true);
            } else if (message.type === "control" && message.text === "conversation-chain-end") {
                setBusy(false);
            } else if (message.type === "error") {
                setBusy(false);
                fetchPhrases();   // the bar may be the one of another character
            }
        });
    }

    function sendToServer(message) {
        if (!socket || socket.readyState !== NativeWebSocket.OPEN) return false;
        socket.send(JSON.stringify(message));
        return true;
    }
    function fetchPhrases() { sendToServer({ type: "fetch-quick-phrases" }); }

    // ---------- the bar ----------

    // The look of the page itself: its panels (footer), inputs, pills and
    // font, so that the bar reads as part of it.
    const STYLE = `
        #quick-bar { position: fixed; top: 14px; left: 572px; right: 12px; z-index: 2000;
            display: flex; align-items: flex-start; gap: 10px; width: fit-content; max-width: calc(100vw - 24px);
            padding: 8px 12px; border-radius: 12px; background: rgb(39, 39, 42); border: 1px solid rgb(82, 82, 91);
            color: rgba(255, 255, 255, .92); box-shadow: 0 4px 12px rgba(0, 0, 0, .4);
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", "Leelawadee UI", Tahoma, Helvetica, Arial, sans-serif;
            font-size: 14px; line-height: 1.5; }
        #quick-bar[hidden] { display: none; }
        #quick-bar .quick-title { border: 0; background: none; color: #fff; font: inherit; font-weight: 600;
            white-space: nowrap; cursor: pointer; padding: 7px 2px 0; }
        #quick-bar .quick-title:hover { color: rgb(34, 197, 94); }
        #quick-bar .quick-head { display: flex; flex-direction: column; align-items: flex-start; }
        #quick-bar .quick-direction { font-size: 12px; color: rgba(255, 255, 255, .64); white-space: nowrap; padding: 0 2px 2px; }
        #quick-bar .quick-row { display: flex; flex-wrap: wrap; gap: 8px; max-height: 38vh; overflow-y: auto;
            scrollbar-width: thin; scrollbar-color: rgb(82, 82, 91) transparent; }
        #quick-bar.collapsed .quick-row { display: none; }
        #quick-bar .quick-row button { flex: none; font: inherit; font-weight: 500; white-space: nowrap; cursor: pointer;
            color: rgba(255, 255, 255, .92); background: rgb(63, 63, 70); border: 1px solid transparent;
            border-radius: 20px; padding: 6px 16px; transition: background .15s; }
        #quick-bar .quick-row button:hover:not(:disabled) { background: rgb(82, 82, 91); }
        #quick-bar .quick-row button:active:not(:disabled) { background: rgb(34, 197, 94); color: #fff; }
        #quick-bar .quick-row button.numbers { border-color: rgb(234, 179, 8); }
        #quick-bar .quick-row button:disabled { opacity: .45; cursor: default; }
        #quick-bar button:focus-visible { outline: 2px solid rgb(34, 197, 94); outline-offset: 2px; }
    `;

    function build() {
        const style = document.createElement("style");
        style.textContent = STYLE;
        document.head.appendChild(style);
        bar = document.createElement("div");
        bar.id = "quick-bar";
        bar.hidden = true;
        const title = document.createElement("button");
        title.className = "quick-title";
        title.title = "ซ่อน / แสดงแถบประโยคใช้บ่อย";
        row = document.createElement("div");
        row.className = "quick-row";
        const setCollapsed = (collapsed) => {
            bar.classList.toggle("collapsed", collapsed);
            title.textContent = collapsed ? "ใช้บ่อย ▸" : "ใช้บ่อย ▾";
            title.setAttribute("aria-expanded", String(!collapsed));
        };
        title.onclick = () => {
            const collapsed = !bar.classList.contains("collapsed");
            setCollapsed(collapsed);
            try { localStorage.setItem("quickBarCollapsed", collapsed ? "1" : ""); } catch (error) { /* private mode */ }
        };
        let collapsed = false;
        try { collapsed = localStorage.getItem("quickBarCollapsed") === "1"; } catch (error) { /* private mode */ }
        setCollapsed(collapsed);
        const head = document.createElement("div");
        head.className = "quick-head";
        tag = document.createElement("span");
        tag.className = "quick-direction";
        head.append(title, tag);
        bar.append(head, row);
        document.body.appendChild(bar);
        render();
        place();
        window.addEventListener("resize", place);
        setInterval(place, 1000);   // the side panel opens and closes
    }

    // Over the top of the avatar, clear of the side panel and of the
    // connection status in the avatar's top left corner.
    function place() {
        if (!bar || bar.hidden) return;
        const canvas = document.getElementById("canvas");
        const area = canvas ? canvas.getBoundingClientRect() : { left: 0, top: 0, right: window.innerWidth, width: window.innerWidth };
        const wide = area.width >= 640;
        bar.style.left = Math.round(area.left + (wide ? 132 : 12)) + "px";
        bar.style.top = Math.round(Math.max(area.top, 0) + (wide ? 14 : 64)) + "px";
        bar.style.right = Math.round(Math.max(window.innerWidth - area.right, 0) + 12) + "px";
        bar.style.maxWidth = "none";
    }

    function render() {
        if (!bar) return;
        bar.hidden = phrases.length === 0;
        tag.textContent = DIRECTION_TEXT[direction];
        place();
        row.replaceChildren(...phrases.map((phrase) => {
            const button = document.createElement("button");
            // the sentence in the language it is said from; Thai for Thai ->
            // Pattani, Pattani for Pattani -> Thai
            const label = phrase.label || phrase.thai;
            button.textContent = (phrase.check_numbers ? "⚠ " : "") + label;
            if (label !== phrase.thai) button.title = phrase.thai;   // what it means
            button.classList.toggle("numbers", Boolean(phrase.check_numbers));
            button.disabled = busy;
            button.onclick = () => choose(phrase);
            return button;
        }));
    }

    function setBusy(value) {
        busy = value;
        clearTimeout(busyTimer);
        if (busy) busyTimer = setTimeout(() => setBusy(false), BUSY_TIMEOUT_MS);
        if (row) for (const button of row.children) button.disabled = busy;
    }

    // A sentence with a number or a dose asks once more before it is said.
    function choose(phrase) {
        if (busy) return;
        if (phrase.check_numbers) {
            const ok = confirm(`ประโยคนี้มีตัวเลขหรือความถี่\n\n"${phrase.label || phrase.thai}"\n\nยืนยันว่าต้องการประโยคนี้?`);
            if (!ok) return;
        }
        if (sendToServer({ type: "quick-phrase", id: phrase.id, direction })) setBusy(true);
    }

    if (document.body) build();
    else document.addEventListener("DOMContentLoaded", build);
})();
