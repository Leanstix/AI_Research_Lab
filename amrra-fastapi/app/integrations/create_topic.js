// app/integrations/create_topic.js
const path = require("path");
require("dotenv").config({ path: path.resolve(__dirname, "../../.env") });
const { Client, TopicCreateTransaction } = require("@hashgraph/sdk");

(async () => {
  const network = process.env.HEDERA_NETWORK || "testnet";
  const operatorId = process.env.HEDERA_OPERATOR_ID;
  const operatorKey = process.env.HEDERA_OPERATOR_KEY;
  const memo = process.argv[2] || "AMRRA research log";

  if (!operatorId || !operatorKey) {
    console.error("Missing HEDERA_OPERATOR_ID / HEDERA_OPERATOR_KEY in .env");
    process.exit(1);
  }

  const client =
    network === "mainnet" ? Client.forMainnet()
    : network === "previewnet" ? Client.forPreviewnet()
    : Client.forTestnet();

  client.setOperator(operatorId, operatorKey);

  const tx = await new TopicCreateTransaction()
    .setTopicMemo(memo)            // <- fix is here
    .execute(client);

  const receipt = await tx.getReceipt(client);
  const topicId = receipt.topicId?.toString();
  console.log("✅ HCS Topic created:", topicId);
  console.log("👉 Add to .env as HCS_TOPIC_ID=", topicId);
})().catch((e) => {
  console.error("Topic create failed:", e);
  process.exit(2);
});
