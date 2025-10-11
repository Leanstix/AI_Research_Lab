const path = require("path");
require("dotenv").config({ path: path.resolve(__dirname, "../../.env") });
const { Client, TopicMessageSubmitTransaction } = require("@hashgraph/sdk");

(async () => {
  const network = process.env.HEDERA_NETWORK || "testnet";
  const operatorId = process.env.HEDERA_OPERATOR_ID;
  const operatorKey = process.env.HEDERA_OPERATOR_KEY;
  const topicId = process.env.HCS_TOPIC_ID;

  if (!operatorId || !operatorKey || !topicId) {
    console.error("Need HEDERA_OPERATOR_ID, HEDERA_OPERATOR_KEY, HCS_TOPIC_ID in .env");
    process.exit(1);
  }

  const client =
    network === "mainnet" ? Client.forMainnet()
    : network === "previewnet" ? Client.forPreviewnet()
    : Client.forTestnet();

  client.setOperator(operatorId, operatorKey);

  const msg = {
    schema: "amrra.test.v1",
    ping: "ok",
    ts: Math.floor(Date.now()/1000)
  };

  const tx = await new TopicMessageSubmitTransaction()
    .setTopicId(topicId)
    .setMessage(Buffer.from(JSON.stringify(msg)))
    .execute(client);

  const receipt = await tx.getReceipt(client);
  console.log("✅ Submitted test message. Status:", receipt.status.toString());
  console.log("Sequence #:", receipt.topicSequenceNumber?.toString());
})();
