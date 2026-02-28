package org.cloudbus.cloudsim.examples.topologyrl;

public final class ProtocolFields {
    private ProtocolFields() {
    }

    public static final String SCHEMA_VERSION = "1.0";

    public static final String KEY_SCHEMA_VERSION = "schema_version";
    public static final String KEY_REQUEST_ID = "request_id";
    public static final String KEY_SLOT = "slot";
    public static final String KEY_DEADLINE_MS = "deadline_ms";
    public static final String KEY_REQUEST = "request";
    public static final String KEY_USER_ID = "user_id";
    public static final String KEY_TIER = "tier";
    public static final String KEY_CANDIDATE_MASK = "candidate_mask";
    public static final String KEY_NODE_STATES = "node_states";
    public static final String KEY_VALID_ACTION_MASK = "valid_action_mask";

    public static final String KEY_STATUS = "status";
    public static final String KEY_ACTION = "action";
    public static final String KEY_TYPE = "type";
    public static final String KEY_NODE_ID = "node_id";
}
