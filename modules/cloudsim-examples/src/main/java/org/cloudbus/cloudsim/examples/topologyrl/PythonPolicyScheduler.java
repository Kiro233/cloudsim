package org.cloudbus.cloudsim.examples.topologyrl;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.net.http.HttpTimeoutException;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Python policy bridge via HTTP.
 *
 * If Python call times out/fails/returns illegal action, fallback scheduler is used.
 */
public class PythonPolicyScheduler implements Scheduler {
    private final HttpClient httpClient;
    private final String policyUrl;
    private final int timeoutMs;
    private final Scheduler fallback;

    private int fallbackTimeoutCount = 0;
    private int fallbackServiceUnavailableCount = 0;
    private int fallbackIllegalActionCount = 0;
    private int policyRejectCount = 0;

    public PythonPolicyScheduler(String policyUrl, int timeoutMs, Scheduler fallback) {
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofMillis(timeoutMs))
                .build();
        this.policyUrl = policyUrl;
        this.timeoutMs = timeoutMs;
        this.fallback = fallback;
    }

    @Override
    public String name() {
        return "python_policy";
    }

    @Override
    public SchedulingDecision selectNode(User user, List<NodeState> nodes) {
        try {
            List<Integer> validMask = buildValidMask(user, nodes);
            String payload = buildRequestPayload(user, nodes, validMask);

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(policyUrl))
                    .timeout(Duration.ofMillis(timeoutMs))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(payload))
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                fallbackServiceUnavailableCount += 1;
                return fallback.selectNode(user, nodes);
            }

            Integer nodeId = parseNodeId(response.body());
            if (isReject(response.body()) || nodeId == null) {
                policyRejectCount += 1;
                return SchedulingDecision.reject();
            }

            if (nodeId < 0 || nodeId >= nodes.size()) {
                fallbackIllegalActionCount += 1;
                return fallback.selectNode(user, nodes);
            }
            if (!user.candidateNodes.contains(nodeId)) {
                fallbackIllegalActionCount += 1;
                return fallback.selectNode(user, nodes);
            }
            if (!nodes.get(nodeId).canAccept(user.tier.name)) {
                fallbackIllegalActionCount += 1;
                return fallback.selectNode(user, nodes);
            }
            return SchedulingDecision.accept(nodeId);
        } catch (HttpTimeoutException timeoutException) {
            fallbackTimeoutCount += 1;
            return fallback.selectNode(user, nodes);
        } catch (Exception ex) {
            fallbackServiceUnavailableCount += 1;
            return fallback.selectNode(user, nodes);
        }
    }

    public void resetStats() {
        fallbackTimeoutCount = 0;
        fallbackServiceUnavailableCount = 0;
        fallbackIllegalActionCount = 0;
        policyRejectCount = 0;
    }

    public int getFallbackTimeoutCount() {
        return fallbackTimeoutCount;
    }

    public int getFallbackServiceUnavailableCount() {
        return fallbackServiceUnavailableCount;
    }

    public int getFallbackIllegalActionCount() {
        return fallbackIllegalActionCount;
    }

    public int getPolicyRejectCount() {
        return policyRejectCount;
    }

    private List<Integer> buildValidMask(User user, List<NodeState> nodes) {
        List<Integer> mask = new ArrayList<>();
        for (int i = 0; i < nodes.size(); i++) {
            boolean candidate = user.candidateNodes.contains(i);
            boolean feasible = nodes.get(i).canAccept(user.tier.name);
            mask.add(candidate && feasible ? 1 : 0);
        }
        mask.add(1); // reject action always valid
        return mask;
    }

    private String buildRequestPayload(User user, List<NodeState> nodes, List<Integer> validMask) {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        sb.append("\"").append(ProtocolFields.KEY_SCHEMA_VERSION).append("\":\"")
                .append(ProtocolFields.SCHEMA_VERSION).append("\",");
        sb.append("\"").append(ProtocolFields.KEY_REQUEST_ID).append("\":\"")
                .append(UUID.randomUUID()).append("\",");
        sb.append("\"").append(ProtocolFields.KEY_SLOT).append("\":-1,");
        sb.append("\"").append(ProtocolFields.KEY_DEADLINE_MS).append("\":").append(timeoutMs).append(",");
        sb.append("\"").append(ProtocolFields.KEY_REQUEST).append("\":{");
        sb.append("\"").append(ProtocolFields.KEY_USER_ID).append("\":").append(user.userId).append(",");
        sb.append("\"").append(ProtocolFields.KEY_TIER).append("\":\"").append(user.tier.name).append("\",");
        sb.append("\"").append(ProtocolFields.KEY_CANDIDATE_MASK).append("\":")
                .append(buildCandidateMask(user, nodes.size()));
        sb.append("},");
        sb.append("\"").append(ProtocolFields.KEY_NODE_STATES).append("\":")
                .append(buildNodeStates(nodes)).append(",");
        sb.append("\"").append(ProtocolFields.KEY_VALID_ACTION_MASK).append("\":")
                .append(validMask.toString());
        sb.append("}");
        return sb.toString();
    }

    private String buildCandidateMask(User user, int nodeCount) {
        List<Integer> mask = new ArrayList<>();
        for (int i = 0; i < nodeCount; i++) {
            mask.add(user.candidateNodes.contains(i) ? 1 : 0);
        }
        return mask.toString();
    }

    private String buildNodeStates(List<NodeState> nodes) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < nodes.size(); i++) {
            NodeState node = nodes.get(i);
            sb.append("{");
            sb.append("\"node_id\":").append(node.nodeId).append(",");
            sb.append("\"nL\":").append(node.nL).append(",");
            sb.append("\"nH\":").append(node.nH).append(",");
            sb.append("\"remaining_capacity\":").append(node.remainingCapacity()).append(",");
            sb.append("\"enabled\":").append(node.isEnabled());
            sb.append("}");
            if (i < nodes.size() - 1) {
                sb.append(",");
            }
        }
        sb.append("]");
        return sb.toString();
    }

    private boolean isReject(String responseJson) {
        String key = "\"" + ProtocolFields.KEY_TYPE + "\":\"reject\"";
        return responseJson.contains(key);
    }

    private Integer parseNodeId(String responseJson) {
        String key = "\"" + ProtocolFields.KEY_NODE_ID + "\":";
        int idx = responseJson.indexOf(key);
        if (idx < 0) {
            return null;
        }
        int start = idx + key.length();
        int end = start;
        while (end < responseJson.length() && Character.isDigit(responseJson.charAt(end))) {
            end++;
        }
        if (end <= start) {
            return null;
        }
        return Integer.parseInt(responseJson.substring(start, end));
    }
}
