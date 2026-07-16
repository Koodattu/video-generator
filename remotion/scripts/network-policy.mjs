import {lookup} from "node:dns/promises";
import {BlockList, isIP} from "node:net";

const blockedAddresses = new BlockList();
for (const [network, prefix] of [
  ["0.0.0.0", 8], ["10.0.0.0", 8], ["100.64.0.0", 10], ["127.0.0.0", 8],
  ["169.254.0.0", 16], ["172.16.0.0", 12], ["192.0.0.0", 24], ["192.0.2.0", 24],
  ["192.168.0.0", 16], ["198.18.0.0", 15], ["198.51.100.0", 24], ["203.0.113.0", 24],
  ["224.0.0.0", 4], ["240.0.0.0", 4],
]) blockedAddresses.addSubnet(network, prefix, "ipv4");
for (const [network, prefix] of [
  ["::", 128], ["::1", 128], ["::ffff:0:0", 96], ["fc00::", 7],
  ["fe80::", 10], ["fec0::", 10], ["ff00::", 8], ["100::", 64],
  ["64:ff9b:1::", 48], ["2001:db8::", 32],
]) blockedAddresses.addSubnet(network, prefix, "ipv6");

const normalizedHostname = (value) => value.replace(/^\[|\]$/g, "").replace(/\.$/, "").toLowerCase();

export const createNetworkPolicy = (configuredHosts) => {
  const allowedHosts = [...new Set((configuredHosts || []).map(normalizedHostname).filter(Boolean))];
  if (!allowedHosts.length) throw new Error("Source screenshot host allowlist is empty");

  return async (value) => {
    const parsed = new URL(value);
    if (!["http:", "https:"].includes(parsed.protocol)) {
      if (["about:", "blob:", "data:"].includes(parsed.protocol)) return;
      throw new Error(`Blocked screenshot request protocol: ${parsed.protocol}`);
    }
    if (parsed.username || parsed.password) throw new Error("Blocked credential-bearing screenshot URL");
    const hostname = normalizedHostname(parsed.hostname);
    if (!hostname || hostname === "localhost" || hostname.endsWith(".localhost") || hostname.endsWith(".local") || hostname.endsWith(".internal") || hostname.endsWith(".home.arpa")) {
      throw new Error(`Blocked non-public screenshot host: ${hostname || "<missing>"}`);
    }
    if (!allowedHosts.some((allowed) => hostname === allowed || hostname.endsWith(`.${allowed}`))) {
      throw new Error(`Blocked screenshot host outside trust allowlist: ${hostname}`);
    }
    const literalFamily = isIP(hostname);
    const addresses = literalFamily
      ? [{address: hostname, family: literalFamily}]
      : await lookup(hostname, {all: true, verbatim: true});
    if (!addresses.length) throw new Error(`Screenshot host did not resolve: ${hostname}`);
    for (const answer of addresses) {
      const family = answer.family === 6 ? "ipv6" : "ipv4";
      if (blockedAddresses.check(answer.address.split("%", 1)[0], family)) {
        throw new Error(`Blocked non-public screenshot address: ${answer.address}`);
      }
    }
  };
};
