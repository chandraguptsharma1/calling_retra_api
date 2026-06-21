const axios = require("axios");
const WebSocket = require("ws");
const { WebSocketServer } = require("ws");
const fs = require("fs");
const { URLSearchParams } = require("url");
const path = require("path");
const wss = new WebSocketServer({ noServer: true });
const Sequelize = require("sequelize");
var sequelizemodel = require("../../mysql");

//Exotel credentials
const EXOTEL_SID = 'retraventures1m'
const EXOTEL_API_KEY = 'e0d5fe0c177d04d54c29462694b97e8ef9f8ee375f03dd11'
const EXOTEL_API_TOKEN = 'd6c07ba4e08e144ba69026d3918dd29b3ef962836a990c49'
const EXOTEL_SUBDOMAIN = 'api.in.exotel.com'

//Your Exophone / CallerId
const EXOTEL_CALLER_ID = '02247790345'
const EXOTEL_APP_ID = 38648
//IMPORTANT:
//This is the URL Exotel will hit to start your flow(ExoML / Flow URL).
//Put your existing working Url here(the one you already use in connect.json)
const EXOTEL_FLOW_URL = 'http://my.exotel.com/retraventures1m/exoml/start_voice/926'

//testing
const ELEVEN_API_KEY = 'sk_7caac1783184afb539e036899bc9b147e3a781940fac5d1b'
const ELEVEN_AGENT_ID = 'agent_9001k636djfbfa7vhgdkjds36avp'

// const ELEVEN_API_KEY = 'sk_367b8bde848f24c2be29c1afc986e3a5c281c0642bca468e'
// const ELEVEN_AGENT_ID = 'agent_9601kb2v4n0ke6vsvc5ccg2qnycw'
const PUBLIC_BASE_URL = 'https://42c0-2409-40e3-3004-d8c-14ea-97e3-c71e-79a8.ngrok-free.app'

const batches = new Map();  // batchId -> { sseClients:Set(res) }
const callCtx = new Map();  // callSid -> { batchId, index, customerName, dueAmount, mobile }
const callFinalOverride = new Map(); // callSid -> { status, detail, source }
const conversationBuffer = {};





module.exports = {
    callTrigger,
    exotelStatusCallback,
    callEvents,
    wss
};

function isClosingMessage(text = "") {
    const t = String(text).toLowerCase().trim();

    const hasThanks =
        t.includes("आपके समय के लिए धन्यवाद") ||
        t.includes("कॉल करने के लिए धन्यवाद") ||
        t.includes("धन्यवाद") ||
        t.includes("thank you");

    const hasEndLine =
        t.includes("हमारा सपोर्ट स्पेशलिस्ट") ||
        t.includes("हमारा support specialist") ||
        t.includes("नमस्ते") ||
        t.includes("bye") ||
        t.includes("goodbye") ||
        t.includes("अलविदा") ||
        t.includes("फिर बात करेंगे");

    return hasThanks && hasEndLine;
}

async function callTrigger(req, res) {
    try {

        const {
            batchId,
            index,
            rowId,
            case_uuid,
            mobile,
            customerName,
            dueAmount
        } = req.body;

        console.log("request body ==> ", req.body);

        if (!batchId || index === undefined || !mobile) {
            return res.status(400).json({
                ok: false,
                error: "batchId, index, mobile required"
            });
        }

        /* ======================
           1️⃣ Trigger Exotel
        ====================== */

        const callSid = await triggerExotelOnly(mobile);

        saveCallDetails({
            callSid,
            case_uuid,
            mobile,
            customerName,
            dueAmount,
            batchId,
            rowId
        });

        /* ======================
           2️⃣ Save Context
        ====================== */

        callCtx.set(callSid, {
            batchId,
            index,
            rowId,
            case_uuid,      // ⭐ IMPORTANT
            mobile,
            customerName,
            dueAmount
        });

        console.log("✅ callSid:", callSid);
        console.log("✅ SAVED CTX:", callCtx.get(callSid));

        /* ======================
           3️⃣ Notify Angular
        ====================== */

        pushToBatch(batchId, {
            type: "CALL_TRIGGERED",
            index,
            callSid,
            mobile
        });

        return res.json({
            ok: true,
            callSid
        });

    } catch (e) {

        console.log("❌ callTrigger error:", e);

        return res.status(500).json({
            ok: false,
            error: e.message
        });
    }
}
async function exotelStatusCallback(req, res) {
    try {
        const p = req.body || {};

        const callSid =
            p.CallSid ||
            p.call_sid ||
            p.callSid;

        const status = String(p.Status || p.CallStatus || "").toLowerCase();

        console.log("✅ EXOTEL CALLBACK:", { callSid, status });

        const ctx = callSid ? callCtx.get(callSid) : null;

        if (!ctx) {
            console.log("⚠️ No ctx found for:", callSid);
            return res.status(200).send("ok");
        }

        const override = callFinalOverride.get(callSid);

        if (override) {
            console.log("⛔ Exotel callback ignored due to override:", {
                callSid,
                exotelStatus: status,
                override
            });

            return res.status(200).send("ok");
        }
        /* ========================
           1️⃣ Map Status
        ======================== */

        let aiStatus = "CALL_TRIGGERED";
        let talkStatus = "CALLING";
        let latestStatus = status;

        if (status === "busy") {
            aiStatus = "FAILED";
            talkStatus = "BUSY";
            latestStatus = "Customer Busy";
        }

        if (status === "no-answer") {
            aiStatus = "FAILED";
            talkStatus = "NO_ANSWER";
            latestStatus = "No Answer";
        }

        if (status === "completed") {
            aiStatus = "SUCCESS";
            talkStatus = "COMPLETED";
            latestStatus = "Call Completed";
        }

        if (status === "failed") {
            aiStatus = "FAILED";
            talkStatus = "FAILED";
            latestStatus = "Call Failed";
        }

        /* ========================
           2️⃣ Push LIVE STATUS
        ======================== */

        pushToBatch(ctx.batchId, {
            type: "CALL_STATUS",
            index: ctx.index,
            callSid,
            status,
            aiStatus,
            talkStatus,
            latestStatus
        });

        /* ========================
           3️⃣ Update DB
        ======================== */

        console.log("✅ AI status updated in DB start");

        // // ✅ 2) DB update (yahi chahiye tumhe)
        // await updateAiStatusInDb({
        //     id: Number(ctx.rowId),
        //     ai_status: aiStatus,
        //     talk_status: talkStatus,
        //     latest_status: latestStatus,
        //     last_call_sid: callSid
        // });

        console.log("✅ AI status updated in DB finish");

        /* ========================
           4️⃣ ONLY ON FINAL → PUSH CALL_FINAL
        ======================== */

        const finalStates = ["completed", "busy", "failed", "no-answer", "canceled"];

        if (finalStates.includes(status)) {

            await flushConversation(callSid);
            await updateCallEnd(callSid, status);


            await saveCallNote(
                ctx.case_uuid,
                ctx.mobile,
                status
            );

            pushToBatch(ctx.batchId, {
                type: "CALL_FINAL",
                index: ctx.index,
                callSid,
                status
            });

            callCtx.delete(callSid);
            callFinalOverride.delete(callSid);
        }

        res.status(200).send("ok");

    } catch (err) {
        console.log("❌ CALLBACK ERROR:", err.message);
        res.status(200).send("ok");
    }
};

function callEvents(req, res) {
    try {
        const batchId = req.query.batchId;

        if (!batchId) {
            return res.status(400).end("batchId required");
        }

        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache, no-transform");
        res.setHeader("Connection", "keep-alive");
        res.setHeader("X-Accel-Buffering", "no");

        if (res.flushHeaders) res.flushHeaders();

        let batch = batches.get(batchId);
        if (!batch) {
            batch = { sseClients: new Set() };
            batches.set(batchId, batch);
        }

        batch.sseClients.add(res);

        console.log("✅ SSE CONNECTED:", batchId, "clients:", batch.sseClients.size);

        // initial event
        res.write(`data: ${JSON.stringify({ type: "CONNECTED", batchId })}\n\n`);

        // keep alive for ngrok / proxies
        const heartbeat = setInterval(() => {
            try {
                res.write(`data: ${JSON.stringify({ type: "PING", t: Date.now() })}\n\n`);
            } catch (e) { }
        }, 15000);

        req.on("close", () => {
            clearInterval(heartbeat);

            const b = batches.get(batchId);
            if (b) {
                b.sseClients.delete(res);
                console.log("❌ SSE CLOSED:", batchId, "clients left:", b.sseClients.size);

                if (b.sseClients.size === 0) {
                    batches.delete(batchId);
                }
            }
        });

    } catch (err) {
        console.log("❌ callEvents error:", err.message);
        res.end();
    }
}

async function triggerExotelOnly(mobile) {

    // direct use constants
    const EXOTEL_SID_LOCAL = EXOTEL_SID;
    const KEY = EXOTEL_API_KEY;
    const TOKEN = EXOTEL_API_TOKEN;
    const SUB = EXOTEL_SUBDOMAIN;
    const CALLER_ID = EXOTEL_CALLER_ID;
    const EXOTEL_APP_ID_LOCAL = EXOTEL_APP_ID;
    const BASE = PUBLIC_BASE_URL;

    console.log("========== EXOTEL ENV CHECK ==========");
    console.log("EXOTEL_SID:", EXOTEL_SID_LOCAL);
    console.log("EXOTEL_API_KEY:", KEY ? "✅ Present" : "❌ Missing");
    console.log("EXOTEL_API_TOKEN:", TOKEN ? "✅ Present" : "❌ Missing");
    console.log("EXOTEL_SUBDOMAIN:", SUB);
    console.log("EXOTEL_CALLER_ID:", CALLER_ID);
    console.log("EXOTEL_APP_ID:", EXOTEL_APP_ID_LOCAL);
    console.log("PUBLIC_BASE_URL:", BASE);
    console.log("======================================");

    if (!BASE) throw new Error("PUBLIC_BASE_URL missing");

    const url = `https://${SUB}/v1/Accounts/${EXOTEL_SID_LOCAL}/Calls/connect.json`;
    const exomlAppUrl = `https://my.exotel.com/${EXOTEL_SID_LOCAL}/exoml/start_voice/${EXOTEL_APP_ID_LOCAL}`;

    const statusCb = `${BASE}/exotel/status`;

    const body = new URLSearchParams({
        From: mobile,
        CallerId: CALLER_ID,
        Url: exomlAppUrl,
        CallType: "trans",
        StatusCallback: statusCb,
    });

    const auth = Buffer.from(`${KEY}:${TOKEN}`).toString("base64");

    const exotelResp = await axios.post(url, body, {
        headers: {
            Authorization: `Basic ${auth}`,
            "Content-Type": "application/x-www-form-urlencoded",
        },
        timeout: 20000,
    });

    let callSid = null;

    if (exotelResp && exotelResp.data) {
        if (exotelResp.data.Call && exotelResp.data.Call.Sid) {
            callSid = exotelResp.data.Call.Sid;
        } else if (exotelResp.data.CallSid) {
            callSid = exotelResp.data.CallSid;
        } else if (exotelResp.data.Sid) {
            callSid = exotelResp.data.Sid;
        } else if (
            exotelResp.data.response &&
            exotelResp.data.response.Call &&
            exotelResp.data.response.Call.Sid
        ) {
            callSid = exotelResp.data.response.Call.Sid;
        }
    }

    if (!callSid) {
        console.log("⚠️ Exotel response:", JSON.stringify(exotelResp.data));
        throw new Error("CallSid not found in Exotel response");
    }

    return callSid;
}

function pushToBatch(batchId, payload) {
    const batch = batches.get(batchId);

    console.log("📤 pushToBatch:", {
        batchId,
        hasBatch: !!batch,
        clients: batch ? batch.sseClients.size : 0,
        payload
    });

    if (!batch) return;

    for (const client of batch.sseClients) {
        try {
            client.write(`data: ${JSON.stringify(payload)}\n\n`);
        } catch (e) {
            console.log("❌ SSE write error:", e.message);
        }
    }
}

function parseElevenError(error) {
    const status = error && error.response ? error.response.status : undefined;
    const data = error && error.response ? error.response.data : undefined;
    const detail =
        (data && data.detail) ||
        (data && data.message) ||
        (error && error.message) ||
        "Unknown ElevenLabs error";


    const text = String(detail).toLowerCase();

    if (
        status === 402 ||
        text.includes("quota") ||
        text.includes("credit") ||
        text.includes("exceeds your quota") ||
        text.includes("insufficient")
    ) {
        return {
            code: "ELEVEN_QUOTA_EXCEEDED",
            detail
        };
    }

    if (status === 401) {
        return {
            code: "ELEVEN_AUTH_FAILED",
            detail
        };
    }

    if (status === 429) {
        return {
            code: "ELEVEN_RATE_LIMITED",
            detail
        };
    }

    return {
        code: "ELEVEN_API_ERROR",
        detail
    };
}


function parseElevenCloseReason(code, reasonText = "") {
    if (Number(code) === 1002) {
        return {
            code: "ELEVEN_QUOTA_EXCEEDED",
            detail: reasonText || "This request exceeds your quota limit."
        };
    }

    return null;
}

/* =========================
   ELEVENLABS WS
========================= */
async function getElevenSignedUrl(agentId) {
    try {
        const resp = await axios.get(
            "https://api.elevenlabs.io/v1/convai/conversation/get-signed-url",
            {
                params: { agent_id: agentId },
                headers: { "xi-api-key": ELEVEN_API_KEY },
                timeout: 10000,
            }
        );

        return resp.data.signed_url;
    } catch (error) {
        const parsed = parseElevenError(error);
        console.log("❌ getElevenSignedUrl failed:", parsed);
        throw new Error(parsed.code + "::" + parsed.detail);
    }
}

function getCustomerNameForVoice(name) {
    if (!name) return "";

    return String(name)
        .trim()
        .replace(/\s+/g, " ")
        .toLowerCase()
        .split(" ")
        .map(w => w.charAt(0).toUpperCase() + w.slice(1))
        .join(" ");
}

function numberToHindiWords(num) {
    num = Math.floor(Number(num || 0));
    if (!num) return "शून्य";

    const words = {
        0: "", 1: "एक", 2: "दो", 3: "तीन", 4: "चार", 5: "पाँच", 6: "छह", 7: "सात", 8: "आठ", 9: "नौ",
        10: "दस", 11: "ग्यारह", 12: "बारह", 13: "तेरह", 14: "चौदह", 15: "पंद्रह", 16: "सोलह", 17: "सत्रह", 18: "अठारह", 19: "उन्नीस",
        20: "बीस", 21: "इक्कीस", 22: "बाईस", 23: "तेइस", 24: "चौबीस", 25: "पच्चीस", 26: "छब्बीस", 27: "सत्ताईस", 28: "अट्ठाईस", 29: "उनतीस",
        30: "तीस", 31: "इकतीस", 32: "बत्तीस", 33: "तैंतीस", 34: "चौंतीस", 35: "पैंतीस", 36: "छत्तीस", 37: "सैंतीस", 38: "अड़तीस", 39: "उनतालीस",
        40: "चालीस", 41: "इकतालीस", 42: "बयालीस", 43: "तैंतालीस", 44: "चवालीस", 45: "पैंतालीस", 46: "छियालीस", 47: "सैंतालीस", 48: "अड़तालीस", 49: "उनचास",
        50: "पचास", 51: "इक्यावन", 52: "बावन", 53: "तिरेपन", 54: "चौवन", 55: "पचपन", 56: "छप्पन", 57: "सत्तावन", 58: "अट्ठावन", 59: "उनसठ",
        60: "साठ", 61: "इकसठ", 62: "बासठ", 63: "तिरसठ", 64: "चौंसठ", 65: "पैंसठ", 66: "छियासठ", 67: "सड़सठ", 68: "अड़सठ", 69: "उनहत्तर",
        70: "सत्तर", 71: "इकहत्तर", 72: "बहत्तर", 73: "तिहत्तर", 74: "चौहत्तर", 75: "पचहत्तर", 76: "छिहत्तर", 77: "सतहत्तर", 78: "अठहत्तर", 79: "उन्नासी",
        80: "अस्सी", 81: "इक्यासी", 82: "बयासी", 83: "तिरासी", 84: "चौरासी", 85: "पचासी", 86: "छियासी", 87: "सत्तासी", 88: "अट्ठासी", 89: "नवासी",
        90: "नब्बे", 91: "इक्यानवे", 92: "बानवे", 93: "तिरानवे", 94: "चौरानवे", 95: "पचानवे", 96: "छियानवे", 97: "सत्तानवे", 98: "अट्ठानवे", 99: "निन्यानवे"
    };

    function belowThousand(n) {
        let out = "";
        const h = Math.floor(n / 100);
        const r = n % 100;

        if (h > 0) {
            out += h === 1 ? "एक सौ" : words[h] + " सौ";
        }
        if (r > 0) {
            out += (out ? " " : "") + words[r];
        }
        return out;
    }

    const lakh = Math.floor(num / 100000);
    const thousand = Math.floor((num % 100000) / 1000);
    const rest = num % 1000;

    let result = "";
    if (lakh > 0) result += belowThousand(lakh) + " लाख ";
    if (thousand > 0) result += belowThousand(thousand) + " हज़ार ";
    if (rest > 0) result += belowThousand(rest);

    return result.trim();
}

function getDueAmountForVoice(amount) {
    const n = Math.round(Number(amount || 0));
    if (!n) return "शून्य रुपये";
    return numberToHindiWords(n) + " रुपये";
}

async function connectEleven(agentId, ctx, callSid, state) {
    const signedUrl = await getElevenSignedUrl(agentId);
    const elWs = new WebSocket(signedUrl);

    let noResponsePromptCount = 0;
    let userResponded = false;
    let firstAgentMessageSeen = false;
    let lastNoResponsePromptAt = 0;

    let conversationStarted = false;

    let allowNoResponseCheck = true;
    let repliedSinceLastPrompt = false;
    let elevenFailureReason = null;


    console.log("🟡 RAW dueAmount:", ctx ? ctx.dueAmount : "");

    const customerNameForVoice = getCustomerNameForVoice(ctx ? ctx.customerName : "");
    const dueAmountForVoice = getDueAmountForVoice(ctx ? ctx.dueAmount : "");

    function isMeaningfulUserReply(text) {
        const t = String(text || "").trim().toLowerCase();

        if (!t) return false;
        if (t === "...") return false;

        const smallValidReplies = [
            "haan", "ha", "hmm", "hm", "ji", "hello",
            "boliye", "kaun", "kon", "yes", "haan ji",
            "ji boliye", "bolo", "sun raha hu", "sun rahi hu"
        ];

        if (smallValidReplies.includes(t)) return true;

        return t.length >= 2;
    }

    function isNoResponseAgentPrompt(text) {
        const t = normalizeText(text);

        return (
            t.includes("क्या आप अभी भी यहाँ हैं") ||
            t.includes("क्या आप वहाँ हैं") ||
            t.includes("क्या आप सुन पा रहे हैं") ||
            t.includes("कृपया बताइए") ||
            t.includes("कृपया मुझे बताएं") ||
            t.includes("क्या आप बात कर सकते हैं") ||
            t.includes("क्या आप अभी बात कर सकते हैं") ||
            t.includes("are you there") ||
            t.includes("can you hear me")
        );
    }

    console.log("🟢 customerNameForVoice:", customerNameForVoice);
    console.log("🟢 dueAmountForVoice:", dueAmountForVoice);


    elWs.on("open", () => {
        console.log("✅ Eleven WS open");
        elWs.send(
            JSON.stringify({
                type: "conversation_initiation_client_data",
                conversation_config_override: {
                    agent: {
                        language: "hi",

                    }
                },

                // ✅ THIS is the main thing you want
                dynamic_variables: {
                    customer_name: customerNameForVoice,
                    due_amount: dueAmountForVoice,
                }

                // conversation_initiation_client_data: {
                //     conversation_config_override: {
                //         conversation: {
                //             text_only: false,
                //             agent_output_audio_format: "pcm_16000",
                //             user_input_audio_format: "pcm_16000",
                //             model_id: "eleven_multilingual_v2",
                //             agent: { language: "hi", voice: { voice_id: "6pVydnYcVtMsrrSeUKs6" } }
                //         }
                //     }
                // },
                // dynamic_variables: {
                //     agent_name: agentName,
                //     customer_name: customerName,
                //     due_amount: dueAmount,
                //     due_date: dueDate
                // }
            })
        );
    });

    elWs.on("message", (data) => {
        let msg;
        try {
            msg = JSON.parse(data.toString());
        } catch {
            return;
        }

        if (msg.type === "conversation_initiation_metadata") {
            console.log("🎛️ Eleven formats:", msg.conversation_initiation_metadata_event);
            return;
        }

        if (msg.type === "audio") {
            elWs.emit("eleven_audio", msg.audio_event.audio_base_64);
            return;
        }

        if (msg.type === "user_transcript") {
            const userText = msg.user_transcription_event.user_transcript || "";
            console.log("📝 user:", userText);

            addConversation(callSid, "user", userText);

            // koi bhi transcript aaya = banda line pe hai
            const normalizedUserText = String(userText || "").trim();

            if (
                normalizedUserText &&
                normalizedUserText !== "..." &&
                normalizedUserText !== "." &&
                normalizedUserText !== ".."
            ) {
                elWs.emit("clear_no_reply_timer");
            }

            if (isMeaningfulUserReply(userText)) {
                repliedSinceLastPrompt = true;
                conversationStarted = true;
                noResponsePromptCount = 0;
                elWs.emit("meaningful_user_reply", userText);
            }

            return;
        }

        if (msg.type === "agent_response") {
            const agentText = msg.agent_response_event.agent_response || "";
            console.log("🤖 agent:", agentText);

            addConversation(callSid, "agent", agentText);

            if (!firstAgentMessageSeen) {
                firstAgentMessageSeen = true;
                elWs.emit("agent_first_message");
                return;
            }

            if (conversationStarted && isClosingMessage(agentText)) {
                console.log("✅ Closing message detected after actual conversation");
                elWs.emit("agent_closing_message", agentText);
                return;
            }

            if (!allowNoResponseCheck) {
                return;
            }

            const isNoRespPrompt = isNoResponseAgentPrompt(agentText);
            console.log("🧠 normalized:", normalizeText(agentText));
            console.log("🧠 isNoResponse:", isNoRespPrompt);
            console.log("🧠 repliedSinceLastPrompt:", repliedSinceLastPrompt);

            if (isNoRespPrompt) {
                const now = Date.now();

                if (now - lastNoResponsePromptAt < 2500) {
                    return;
                }

                lastNoResponsePromptAt = now;

                if (!repliedSinceLastPrompt) {
                    noResponsePromptCount++;
                    console.log("🔇 noResponsePromptCount:", noResponsePromptCount);

                    if (noResponsePromptCount === 1) {
                        elWs.emit("start_no_reply_timer");
                    }

                    if (noResponsePromptCount >= 3) {
                        console.log("❌ No user response after 3 prompts. Ending call.");
                        elWs.emit("force_disconnect", {
                            reason: "NO_RESPONSE_AFTER_3_PROMPTS"
                        });
                        return;
                    }
                } else {
                    noResponsePromptCount = 0;
                }

                // next prompt ke liye reset
                repliedSinceLastPrompt = false;
            }

            return;
        }

        if (msg.type === "error") {
            const errText =
                (msg && msg.error && msg.error.message) ||
                (msg && msg.message) ||
                JSON.stringify(msg);

            console.log("❌ Eleven WS message error:", errText);

            const lower = String(errText).toLowerCase();

            if (
                lower.includes("quota") ||
                lower.includes("credit") ||
                lower.includes("exceeds your quota") ||
                lower.includes("insufficient")
            ) {
                elevenFailureReason = {
                    code: "ELEVEN_QUOTA_EXCEEDED",
                    detail: errText
                };

                addConversation(callSid, "system", "ElevenLabs quota exceeded");

                elWs.emit("force_disconnect", {
                    reason: "ELEVEN_QUOTA_EXCEEDED",
                    detail: errText
                });

                return;
            }

            elevenFailureReason = {
                code: "ELEVEN_WS_ERROR",
                detail: errText
            };

            addConversation(callSid, "system", `Eleven WS error: ${errText}`);
            return;
        }
    });

    elWs.on("close", async (code, reason) => {
        const closeReason = reason ? reason.toString() : "";

        console.log("❌ Eleven WS closed", {
            code,
            reason: closeReason,
            gracefulShutdown: state.gracefulShutdown,
            finalCallHandled: state.finalCallHandled,
            elevenFailureReason
        });

        // ✅ only 1002 should be handled even if gracefulShutdown is true
        if (!elevenFailureReason) {
            const parsedCloseError = parseElevenCloseReason(code, closeReason);
            if (parsedCloseError) {
                elevenFailureReason = parsedCloseError;
            }
        }

        // ✅ 1002 quota case -> always force disconnect
        if (elevenFailureReason && elevenFailureReason.code === "ELEVEN_QUOTA_EXCEEDED") {
            console.log("📴 Eleven 1002 quota detected. Force disconnecting call.");

            elWs.emit("force_disconnect", {
                reason: "ELEVEN_QUOTA_EXCEEDED",
                detail: elevenFailureReason.detail || "This request exceeds your quota limit."
            });
            return;
        }

        // ignore close after normal/graceful end
        if (state.gracefulShutdown || state.finalCallHandled) {
            console.log("ℹ️ Ignoring Eleven close after graceful shutdown");
            return;
        }

        console.log("ℹ️ Non-quota Eleven close ignored");
    });
    elWs.on("error", (e) => {
        const errText = (e && e.message) || "Unknown Eleven websocket error";


        console.log("❌ Eleven WS error:", errText);

        const lower = String(errText).toLowerCase();

        if (
            lower.includes("quota") ||
            lower.includes("credit") ||
            lower.includes("exceeds your quota") ||
            lower.includes("insufficient")
        ) {
            elevenFailureReason = {
                code: "ELEVEN_QUOTA_EXCEEDED",
                detail: errText
            };

            addConversation(callSid, "system", "ElevenLabs quota exceeded");

            elWs.emit("force_disconnect", {
                reason: "ELEVEN_QUOTA_EXCEEDED",
                detail: errText
            });

            return;
        }

        elevenFailureReason = {
            code: "ELEVEN_WS_ERROR",
            detail: errText
        };
    });
    return elWs;
}

/* =========================
   EXOTEL WS SERVER
========================= */


wss.on("connection", async (exoWs, req) => {
    const state = {
        gracefulShutdown: false,
        finalCallHandled: false
    };

    function safeCompleteAndClose() {
        state.gracefulShutdown = true;
        state.finalCallHandled = true;

        setTimeout(() => {
            try { clearNoReplyTimer(); } catch { }
            try { player.stop(); } catch { }

            try {
                if (elWs && elWs.readyState === elWs.OPEN) {
                    elWs.close(1000, "graceful_end");
                }
            } catch { }

            try {
                if (exoWs && exoWs.readyState === exoWs.OPEN) {
                    exoWs.close();
                }
            } catch { }
        }, 5000);
    }
    console.log(
        "✅ Exotel WS CONNECTED:",
        req && req.headers ? req.headers.host : ""
    );

    const agentId = ELEVEN_AGENT_ID;
    let elWs = null;

    // audio player (smooth) for Exotel playback
    const player = createExotelPlayer(exoWs);

    let noReplyTimer = null;

    function clearNoReplyTimer() {
        if (noReplyTimer) {
            clearTimeout(noReplyTimer);
            noReplyTimer = null;
            console.log("🧹 No-reply timer cleared");
        }
    }

    function startNoReplyTimer() {
        clearNoReplyTimer();

        noReplyTimer = setTimeout(() => {
            console.log("⏰ No meaningful reply within timeout. Disconnecting call.");

            if (elWs) {
                elWs.emit("force_disconnect", {
                    reason: "NO_RESPONSE_TIMEOUT"
                });
            }
        }, 35000);
    }

    // call buffers
    let callSid = `call_${Date.now()}`;
    let pcmChunks = [];

    // // connect to eleven
    // try {
    //     elWs = await connectEleven(agentId);
    // } catch (e) {
    //     console.log("❌ Eleven connect failed:", e.message);
    // }

    // // Eleven -> Exotel (pcm_16000 -> pcm_8000)
    // if (elWs) {
    //     elWs.on("eleven_audio", (audioB64) => {
    //         if (exoWs.readyState !== exoWs.OPEN) return;

    //         const pcm16k = Buffer.from(audioB64, "base64");
    //         const pcm8k = downsample16kTo8k(pcm16k);

    //         // smooth playback (no cut-cut)
    //         player.push(pcm8k);
    //     });
    // }

    exoWs.on("message", async (buf) => {
        let msg;
        try {
            msg = JSON.parse(buf.toString());
        } catch {
            return;
        }

        const logLine = JSON.stringify({ t: Date.now(), ...msg }) + "\n";

        // START
        if (msg.event === "start") {
            if (msg && msg.start) {
                callSid =
                    msg.start.call_sid ||
                    msg.start.callSid ||
                    msg.start.CallSid ||
                    msg.start.call_id ||
                    msg.callSid ||
                    msg.call_id ||
                    callSid;
            } else {
                callSid =
                    (msg && msg.callSid) ||
                    (msg && msg.call_id) ||
                    callSid;
            }

            // ✅ Get ctx
            const ctx = callCtx.get(callSid);


            if (ctx) {
                console.log("✅ CTX FOUND:", ctx);
            } else {
                console.log("⚠️ CTX NOT FOUND for:", callSid);
            }

            try {
                elWs = await connectEleven(agentId, ctx, callSid, state);



                // ✅ listener yahi lagana hai
                elWs.on("eleven_audio", (audioB64) => {
                    if (exoWs.readyState !== exoWs.OPEN) return;

                    const pcm16k = Buffer.from(audioB64, "base64");
                    const pcm8k = downsample16kTo8k(pcm16k);

                    player.push(pcm8k);
                });

                elWs.on("meaningful_user_reply", () => {
                    console.log("✅ Meaningful user reply received");
                    clearNoReplyTimer();
                });

                elWs.on("clear_no_reply_timer", () => {
                    clearNoReplyTimer();
                });

                elWs.on("start_no_reply_timer", () => {
                    console.log("⏰ Starting no-reply timer");
                    startNoReplyTimer();
                });

                elWs.on("agent_first_message", () => {
                    console.log("🗣️ Agent first greeting sent");
                });

                elWs.on("force_disconnect", async (meta) => {
                    if (state.finalCallHandled) {
                        console.log("ℹ️ force_disconnect ignored, final already handled");
                        return;
                    }

                    state.finalCallHandled = true;

                    console.log("📴 Force disconnecting call:", meta);

                    const finalReason = (meta && meta.reason) || "FORCED_DISCONNECT";

                    callFinalOverride.set(callSid, {
                        status: finalReason,
                        detail: (meta && meta.detail) || "",
                        source: "ELEVEN"
                    });

                    try {
                        if (ctx && ctx.case_uuid && ctx.mobile) {
                            if (finalReason === "NO_RESPONSE_AFTER_3_PROMPTS") {
                                await saveCallNote(ctx.case_uuid, ctx.mobile, "no-response");
                            } else if (finalReason === "ELEVEN_QUOTA_EXCEEDED") {
                                await saveCallNote(ctx.case_uuid, ctx.mobile, "failed");
                            } else {
                                await saveCallNote(ctx.case_uuid, ctx.mobile, "failed");
                            }
                        }
                    } catch (e) {
                        console.log("❌ saveCallNote(force_disconnect) error:", e.message);
                    }

                    try {

                        await updateCallEnd(callSid, finalReason);
                    } catch (e) {
                        console.log("❌ updateCallEnd(force_disconnect) error:", e.message);
                    }

                    try {
                        if (ctx && ctx.batchId !== undefined && ctx.index !== undefined) {
                            pushToBatch(ctx.batchId, {
                                type: "CALL_FINAL",
                                index: ctx.index,
                                callSid,
                                status: finalReason,
                                detail: (meta && meta.detail) || ""
                            });
                        } else {
                            console.log("⚠️ CALL_FINAL not pushed because ctx missing for:", callSid);
                        }
                    } catch (e) {
                        console.log("❌ pushToBatch(force_disconnect) error:", e.message);
                    }

                    setTimeout(() => {
                        callFinalOverride.delete(callSid);
                    }, 120000); // 2 min safe window

                    try { clearNoReplyTimer(); } catch { }
                    try { player.stop(); } catch { }
                    try { if (elWs) elWs.close(); } catch { }
                    try { exoWs.close(); } catch { }
                });

                elWs.on("agent_closing_message", async (agentText) => {
                    if (state.finalCallHandled) {
                        console.log("ℹ️ agent_closing_message ignored, final already handled");
                        return;
                    }
                    console.log("📴 Agent requested graceful call end:", agentText);

                    state.gracefulShutdown = true;
                    state.finalCallHandled = true;

                    try {
                        if (ctx && ctx.case_uuid && ctx.mobile) {
                            await saveCallNote(ctx.case_uuid, ctx.mobile, "completed");
                        }
                    } catch (e) {
                        console.log("❌ saveCallNote(agent_closing_message) error:", e.message);
                    }

                    try {
                        await updateCallEnd(callSid, "completed");
                    } catch (e) {
                        console.log("❌ updateCallEnd(agent_closing_message) error:", e.message);
                    }

                    try {
                        pushToBatch(ctx.batchId, {
                            type: "CALL_FINAL",
                            index: ctx.index,
                            callSid,
                            status: "completed",
                            detail: ""
                        });
                    } catch (e) {
                        console.log("❌ pushToBatch(agent_closing_message) error:", e.message);
                    }

                    safeCompleteAndClose();
                });
            } catch (e) {
                console.log("❌ Eleven connect failed:", e.message);
            }


            fs.appendFileSync(fileForCall(callSid, "events.jsonl"), logLine);
            console.log("📞 START:", callSid);
            return;
        }

        // MEDIA
        if (
            msg.event === "media" &&
            msg.media &&
            msg.media.payload
        ) {
            const payloadB64 = msg.media.payload;

            // Save caller audio (assumed PCM16 8k)
            const pcm8k = Buffer.from(payloadB64, "base64");
            pcmChunks.push(pcm8k);
            fs.appendFileSync(fileForCall(callSid, "events.jsonl"), logLine);

            // Exotel -> Eleven expects ulaw_8000 (from your metadata)
            if (elWs && elWs.readyState === elWs.OPEN) {
                const ulaw8k = pcm16ToUlaw(pcm8k);
                elWs.send(JSON.stringify({ user_audio_chunk: ulaw8k.toString("base64") }));
            }

            return;
        }

        // STOP
        if (msg.event === "stop") {
            fs.appendFileSync(fileForCall(callSid, "events.jsonl"), logLine);
            console.log("📴 STOP:", callSid);

            clearNoReplyTimer();

            const pcm = Buffer.concat(pcmChunks);
            const wavPath = fileForCall(callSid, "wav");
            // writeWavFile(wavPath, pcm, { sampleRate: 8000, channels: 1, bitsPerSample: 16 });
            // console.log("✅ WAV SAVED:", wavPath, "size:", pcm.length);

            pcmChunks = [];
            return;
        }



        fs.appendFileSync(fileForCall(callSid, "events.jsonl"), logLine);
    });

    exoWs.on("close", () => {
        console.log("❌ Exotel WS closed:", callSid);
        try {
            clearNoReplyTimer();
            player.stop();
        } catch { }
        try {
            if (elWs) {
                elWs.close();
            }
        } catch { }
    });
});

// Smooth player: send PCM8k to Exotel in 100ms frames (1600 bytes = 100ms @ 8kHz PCM16 mono)
function createExotelPlayer(exoWs, { chunkBytes = 1600, intervalMs = 100 } = {}) {

    let queue = Buffer.alloc(0);
    let timer = null;
    let ended = false;

    const pump = () => {

        if (exoWs.readyState !== exoWs.OPEN) return stop();

        if (queue.length < chunkBytes) {

            if (ended && queue.length === 0) {
                stop();
                exoWs.emit("player_finished");
            }

            return;
        }

        const chunk = queue.subarray(0, chunkBytes);
        queue = queue.subarray(chunkBytes);

        exoWs.send(
            JSON.stringify({
                event: "media",
                media: { payload: chunk.toString("base64") }
            })
        );
    };

    const start = () => {
        if (timer) return;
        timer = setInterval(pump, intervalMs);
    };

    const stop = () => {
        if (timer) clearInterval(timer);
        timer = null;
        queue = Buffer.alloc(0);
    };

    const push = (buf) => {
        queue = Buffer.concat([queue, buf]);
        start();
    };

    const markEnd = () => {
        ended = true;
    };

    return { push, stop, markEnd };
}

// Downsample PCM16 from 16k -> 8k (avg pairs)
function downsample16kTo8k(pcm16kBuf) {
    const src = pcm16leToInt16(pcm16kBuf);
    const outLen = Math.floor(src.length / 2);
    const out = new Int16Array(outLen);

    for (let i = 0, j = 0; j < outLen; j++, i += 2) {
        out[j] = ((src[i] + src[i + 1]) / 2) | 0;
    }
    return int16ToPcm16le(out);
}

const DATA_DIR = path.join(process.cwd(), "call_logs");
if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });

function fileForCall(callSid, ext) {
    return path.join(DATA_DIR, `${callSid}.${ext}`);
}

/* =========================
   AUDIO UTILS
   - PCM16 <-> uLaw
   - Resample 16k<->8k
========================= */

// PCM16LE buffer -> Int16Array view
function pcm16leToInt16(pcmBuf) {
    return new Int16Array(pcmBuf.buffer, pcmBuf.byteOffset, Math.floor(pcmBuf.byteLength / 2));
}

// Int16Array -> PCM16LE Buffer
function int16ToPcm16le(int16) {
    return Buffer.from(int16.buffer, int16.byteOffset, int16.byteLength);
}

// μ-law encode (PCM16 -> uLaw8)
function pcm16ToUlaw(pcmBuf) {
    const pcm = pcm16leToInt16(pcmBuf);
    const out = Buffer.alloc(pcm.length);
    for (let i = 0; i < pcm.length; i++) out[i] = linearToUlawSample(pcm[i]);
    return out;
}

function linearToUlawSample(sample) {
    const BIAS = 0x84;
    const CLIP = 32635;

    let sign = (sample >> 8) & 0x80;
    if (sign !== 0) sample = -sample;
    if (sample > CLIP) sample = CLIP;

    sample = sample + BIAS;

    let exponent = 7;
    for (let expMask = 0x4000; (sample & expMask) === 0 && exponent > 0; expMask >>= 1) exponent--;

    const mantissa = (sample >> (exponent + 3)) & 0x0f;
    const ulaw = ~(sign | (exponent << 4) | mantissa);
    return ulaw & 0xff;
}

async function saveCallNote(caseUuid, mobile, status) {

    try {

        let remarks = "No Response";
        let actionStatusCode = 1725;

        if (status === "completed") {
            remarks = "Call Completed";
            actionStatusCode = 1710;
        }

        if (status === "busy") {
            remarks = "Customer Busy";
            actionStatusCode = 1715;
        }

        const payload = {
            case_uuid: caseUuid,
            contactMode: 11,
            actionStatusCode: actionStatusCode,
            remarks: remarks,
            contactPerson: mobile,
            userTransKey: "1003--18",
            isPaymentDone: "N",
            isPTP: "N",
            nextActionDate: "2026/03/07",
            followupRequired: "N",
            isSalaryCut: "N",
            isBulkUpload: "N"
        };

        const res = await axios.post(
            "https://surecollect.ai:3000/saveCallNote",
            {
                noteData: JSON.stringify(payload)
            }
        );

        console.log("✅ saveCallNote success:", res.data);

    } catch (err) {

        console.log("❌ saveCallNote error:", err.message);

    }


}

function addConversation(callSid, role, message) {

    if (!message) return;

    if (!conversationBuffer[callSid]) {
        conversationBuffer[callSid] = [];
    }

    conversationBuffer[callSid].push({
        role,
        message,
        time: new Date()
    });

}

async function saveCallDetails(data) {

    await sequelizemodel.sequelizeretra.query(
        `INSERT INTO ai_call_details
   (call_sid, case_uuid, mobile, customer_name, due_amount, batch_id, row_id, start_time)
   VALUES (?, ?, ?, ?, ?, ?, ?, NOW())`,
        {
            replacements: [
                data.callSid,
                data.case_uuid,
                data.mobile,
                data.customerName,
                data.dueAmount,
                data.batchId,
                data.rowId
            ],
            type: Sequelize.QueryTypes.INSERT
        }
    ).catch(err => console.log("saveCallDetails error", err));

}

async function flushConversation(callSid) {

    const messages = conversationBuffer[callSid];

    if (!messages || messages.length === 0) return;

    const values = [];
    const replacements = [];

    messages.forEach(m => {

        values.push("(?, ?, ?)");

        replacements.push(
            callSid,
            m.role,
            m.message
        );

    });

    const sql = `
 INSERT INTO ai_call_conversation
 (call_sid, role, message)
 VALUES ${values.join(",")}
 `;

    await sequelizemodel.sequelizeretra.query(sql, {
        replacements,
        type: Sequelize.QueryTypes.INSERT
    });

    delete conversationBuffer[callSid];

}

async function updateCallEnd(callSid, status) {

    await sequelizemodel.sequelizeretra.query(
        `UPDATE ai_call_details
   SET call_status = ?, end_time = NOW()
   WHERE call_sid = ?`,
        {
            replacements: [status, callSid],
            type: Sequelize.QueryTypes.UPDATE
        }
    );

}



// (Optional) Upsample 8k -> 16k (duplicate samples)
// If in future Eleven expects pcm_16000 input, you’ll need this.
function upsample8kTo16k(pcm8kBuf) {
    const src = pcm16leToInt16(pcm8kBuf);
    const out = new Int16Array(src.length * 2);
    for (let i = 0; i < src.length; i++) {
        out[i * 2] = src[i];
        out[i * 2 + 1] = src[i];
    }
    return int16ToPcm16le(out);
}

function normalizeText(text) {
    return String(text || "")
        .toLowerCase()
        .replace(/[^\w\s\u0900-\u097F]/g, '') // punctuation remove
        .replace(/\s+/g, ' ')
        .trim();
}