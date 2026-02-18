import express from "express";
import morgan from "morgan";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import axios from "axios";
import { WebSocketServer } from "ws";
import WebSocket from "ws"; // for Eleven WS

dotenv.config();

const app = express();
app.use(express.json());
app.use(morgan("dev"));

const PORT = process.env.PORT || 8091;

/* =========================
   STORAGE
========================= */
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

// Smooth player: send PCM8k to Exotel in 100ms frames (1600 bytes = 100ms @ 8kHz PCM16 mono)
function createExotelPlayer(exoWs, { chunkBytes = 1600, intervalMs = 100 } = {}) {
    let queue = Buffer.alloc(0);
    let timer = null;

    const pump = () => {
        if (exoWs.readyState !== exoWs.OPEN) return stop();
        if (queue.length < chunkBytes) return;

        const chunk = queue.subarray(0, chunkBytes);
        queue = queue.subarray(chunkBytes);

        exoWs.send(
            JSON.stringify({
                event: "media",
                media: { payload: chunk.toString("base64") },
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

    return { push, stop };
}

/* =========================
   WAV WRITER
========================= */
function writeWavFile(filePath, pcmBuffer, { sampleRate = 8000, channels = 1, bitsPerSample = 16 } = {}) {
    const byteRate = sampleRate * channels * (bitsPerSample / 8);
    const blockAlign = channels * (bitsPerSample / 8);
    const dataSize = pcmBuffer.length;

    const header = Buffer.alloc(44);
    header.write("RIFF", 0);
    header.writeUInt32LE(36 + dataSize, 4);
    header.write("WAVE", 8);
    header.write("fmt ", 12);
    header.writeUInt32LE(16, 16);
    header.writeUInt16LE(1, 20);
    header.writeUInt16LE(channels, 22);
    header.writeUInt32LE(sampleRate, 24);
    header.writeUInt32LE(byteRate, 28);
    header.writeUInt16LE(blockAlign, 32);
    header.writeUInt16LE(bitsPerSample, 34);
    header.write("data", 36);
    header.writeUInt32LE(dataSize, 40);

    fs.writeFileSync(filePath, Buffer.concat([header, pcmBuffer]));
}

/* =========================
   ELEVENLABS WS
========================= */
async function getElevenSignedUrl(agentId) {
    const resp = await axios.get("https://api.elevenlabs.io/v1/convai/conversation/get-signed-url", {
        params: { agent_id: agentId },
        headers: { "xi-api-key": process.env.ELEVEN_API_KEY },
    });
    return resp.data.signed_url;
}

async function connectEleven(agentId) {
    const signedUrl = await getElevenSignedUrl(agentId);
    const elWs = new WebSocket(signedUrl);

    elWs.on("open", () => {
        console.log("✅ Eleven WS open");
        elWs.send(
            JSON.stringify({
                type: "conversation_initiation_client_data",
                conversation_config_override: {
                    agent: {
                        language: "hi",
                        // prompt: "Speak in natural, clear Hindi. Pronounce words correctly. Avoid mixing English unless the user speaks English."
                    }
                },
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

        if (msg.type === "user_transcript") console.log("📝 user:", msg.user_transcription_event?.user_transcript);
        if (msg.type === "agent_response") console.log("🤖 agent:", msg.agent_response_event?.agent_response);
    });

    elWs.on("close", () => console.log("❌ Eleven WS closed"));
    elWs.on("error", (e) => console.log("❌ Eleven WS error:", e.message));

    return elWs;
}

/* =========================
   EXOTEL WS SERVER
========================= */
const wss = new WebSocketServer({ noServer: true });

wss.on("connection", async (exoWs, req) => {
    console.log("✅ Exotel WS CONNECTED:", req?.headers?.host);

    const agentId = process.env.ELEVEN_AGENT_ID;
    let elWs = null;

    // audio player (smooth) for Exotel playback
    const player = createExotelPlayer(exoWs);

    // call buffers
    let callSid = `call_${Date.now()}`;
    let pcmChunks = [];

    // connect to eleven
    try {
        elWs = await connectEleven(agentId);
    } catch (e) {
        console.log("❌ Eleven connect failed:", e.message);
    }

    // Eleven -> Exotel (pcm_16000 -> pcm_8000)
    if (elWs) {
        elWs.on("eleven_audio", (audioB64) => {
            if (exoWs.readyState !== exoWs.OPEN) return;

            const pcm16k = Buffer.from(audioB64, "base64");
            const pcm8k = downsample16kTo8k(pcm16k);

            // smooth playback (no cut-cut)
            player.push(pcm8k);
        });
    }

    exoWs.on("message", (buf) => {
        let msg;
        try {
            msg = JSON.parse(buf.toString());
        } catch {
            return;
        }

        const logLine = JSON.stringify({ t: Date.now(), ...msg }) + "\n";

        // START
        if (msg.event === "start") {
            callSid =
                msg?.start?.call_sid ||
                msg?.start?.callSid ||
                msg?.start?.CallSid ||
                msg?.start?.call_id ||
                msg?.callSid ||
                msg?.call_id ||
                callSid;

            fs.appendFileSync(fileForCall(callSid, "events.jsonl"), logLine);
            console.log("📞 START:", callSid);
            return;
        }

        // MEDIA
        if (msg.event === "media" && msg?.media?.payload) {
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

            const pcm = Buffer.concat(pcmChunks);
            const wavPath = fileForCall(callSid, "wav");
            writeWavFile(wavPath, pcm, { sampleRate: 8000, channels: 1, bitsPerSample: 16 });
            console.log("✅ WAV SAVED:", wavPath, "size:", pcm.length);

            pcmChunks = [];
            return;
        }

        fs.appendFileSync(fileForCall(callSid, "events.jsonl"), logLine);
    });

    exoWs.on("close", () => {
        console.log("❌ Exotel WS closed:", callSid);
        try {
            player.stop();
        } catch { }
        try {
            elWs?.close();
        } catch { }
    });
});

/* =========================
   HTTP SERVER + WS UPGRADE
========================= */
const server = app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));

server.on("upgrade", (req, socket, head) => {
    if (req.url === "/ws/exotel") {
        wss.handleUpgrade(req, socket, head, (ws) => wss.emit("connection", ws, req));
    } else {
        socket.destroy();
    }
});

/* =========================
   API: TEST CALL
========================= */
app.post("/api/test-call", async (req, res) => {
    try {
        const { to, from } = req.body;

        console.log("\n================= /api/test-call HIT =================");
        console.log("REQ BODY:", req.body);

        const EXOTEL_SID = process.env.EXOTEL_SID;
        const KEY = process.env.EXOTEL_API_KEY;
        const TOKEN = process.env.EXOTEL_API_TOKEN;
        const SUB = process.env.EXOTEL_SUBDOMAIN; // api.in.exotel.com
        const CALLER_ID = process.env.EXOTEL_CALLER_ID; // Exophone
        const EXOTEL_APP_ID = process.env.EXOTEL_APP_ID;

        console.log("ENV CHECK:", {
            EXOTEL_SID,
            EXOTEL_API_KEY_PRESENT: !!KEY,
            EXOTEL_API_TOKEN_PRESENT: !!TOKEN,
            EXOTEL_SUBDOMAIN: SUB,
            EXOTEL_CALLER_ID: CALLER_ID,
            EXOTEL_APP_ID,
        });

        if (!to && !from) {
            return res.status(400).json({ ok: false, error: "Send 'from' or 'to' number." });
        }

        const finalFrom = from || to;
        const url = `https://${SUB}/v1/Accounts/${EXOTEL_SID}/Calls/connect.json`;
        const exomlAppUrl = `http://my.exotel.com/${EXOTEL_SID}/exoml/start_voice/${EXOTEL_APP_ID}`;

        const body = new URLSearchParams({
            From: finalFrom,
            CallerId: CALLER_ID,
            Url: exomlAppUrl,
            CallType: "trans",
        });

        console.log("EXOTEL URL:", url);
        console.log("EXOTEL BODY:", body.toString());
        console.log("=====================================================\n");

        const auth = Buffer.from(`${KEY}:${TOKEN}`).toString("base64");

        const exotelResp = await axios.post(url, body, {
            headers: {
                Authorization: `Basic ${auth}`,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            timeout: 20000,
        });

        return res.json({ ok: true, exotel: exotelResp.data });
    } catch (err) {
        const status = err?.response?.status || 500;
        const data = err?.response?.data || err.message;
        console.log("❌ EXOTEL ERROR STATUS:", status);
        console.log("❌ EXOTEL ERROR DATA:", data);
        return res.status(status).json({ ok: false, error: data });
    }
});

app.get("/health", (_, res) => res.send("ok"));
