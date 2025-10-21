import "dotenv/config";
import express from "express";
import morgan from "morgan";
import {
  Client,
  TopicMessageSubmitTransaction,
  PrivateKey,
  AccountId,
} from "@hashgraph/sdk";

const {
  HEDERA_NETWORK = "testnet",
  HEDERA_ACCOUNT_ID,
  HEDERA_PRIVATE_KEY,
  HEDERA_TOPIC_ID,
  PORT = 3001,
} = process.env;

if (!HEDERA_ACCOUNT_ID || !HEDERA_PRIVATE_KEY || !HEDERA_TOPIC_ID) {
  console.error("Missing Hedera envs. Check HEDERA_ACCOUNT_ID / HEDERA_PRIVATE_KEY / HEDERA_TOPIC_ID");
  process.exit(1);
}

const client =
  HEDERA_NETWORK === "mainnet" ? Client.forMainnet() :
  HEDERA_NETWORK === "previewnet" ? Client.forPreviewnet() :
  Client.forTestnet();

client.setOperator(AccountId.fromString(HEDERA_ACCOUNT_ID), PrivateKey.fromString(HEDERA_PRIVATE_KEY));

const app = express();
app.use(express.json({ limit: "2mb" }));
app.use(morgan("dev"));

app.get("/health", (_req, res) => res.json({ ok: true }));

app.post("/hcs/log", async (req, res) => {
  try {
    const { message = "", memo = "", referenceId = "", hash = "" } = req.body || {};
    const payload = {
      memo, referenceId, hash,
      message,
      ts: new Date().toISOString(),
    };
    const bytes = Buffer.from(JSON.stringify(payload), "utf8");

    const tx = await new TopicMessageSubmitTransaction()
      .setTopicId(HEDERA_TOPIC_ID)
      .setMessage(bytes)
      .execute(client);

    const receipt = await tx.getReceipt(client);
    const txId = tx.transactionId.toString();
    const consensusTimestamp = receipt.consensusTimestamp?.toDate()?.toISOString();

    res.json({ ok: true, txId, topicId: HEDERA_TOPIC_ID, consensusTimestamp });
  } catch (err) {
    console.error(err);
    res.status(500).json({ ok: false, error: String(err) });
  }
});

app.listen(PORT, () => console.log(`HCS server listening on :${PORT}`));
