// app/integrations/hcs_client.js
const { Client, TopicMessageSubmitTransaction } = require('@hashgraph/sdk');

async function main() {
  const payload = JSON.parse(process.argv[2] || "{}");
  let { topicId, message, network, operatorId, operatorKey, awaitReceipt } = payload;

  topicId = (topicId || "").trim();
  if (!/^0\.0\.\d+$/.test(topicId)) {
    console.error("Invalid topic id:", topicId);
    process.exit(1);
  }
  if (!operatorId || !operatorKey) {
    console.error("Missing operator creds");
    process.exit(1);
  }

  const client = network === 'mainnet' ? Client.forMainnet()
               : network === 'previewnet' ? Client.forPreviewnet()
               : Client.forTestnet();
  client.setOperator(operatorId, operatorKey);

  try {
    const txResp = await new TopicMessageSubmitTransaction()
      .setTopicId(topicId)
      .setMessage(Buffer.from(JSON.stringify(message)))
      .execute(client);

    const result = { transactionId: txResp.transactionId.toString() };

    if (awaitReceipt) {
      const receipt = await txResp.getReceipt(client);
      result.status = receipt.status ? receipt.status.toString() : 'UNKNOWN';
      result.topicSequenceNumber = receipt.topicSequenceNumber ? Number(receipt.topicSequenceNumber) : null;
    }

    // PRINT ONLY JSON TO STDOUT
    console.log(JSON.stringify(result));
  } catch (err) {
    console.error(err);
    process.exit(2);
  } finally {
    // <<< THIS IS THE IMPORTANT PART
    try { client.close(); } catch (_) {}
  }
}

main();
