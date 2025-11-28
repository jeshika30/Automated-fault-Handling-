FAULT_TO_SCRIPT = {
    "LINK_DOWN": {
        "script_id": "link_recover_basic",
        "description": "Recover link on interface and verify status",
        "commands": [
            "show interfaces Gi0/1",
            "show logging | include Gi0/1",
            "configure terminal",
            "interface Gi0/1",
            "shutdown",
            "no shutdown",
            "end",
            "show interfaces Gi0/1",
        ],
    },
    "BGP_FLAP": {
        "script_id": "bgp_session_recover",
        "description": "Clear unstable BGP session and verify reachability",
        "commands": [
            "show ip bgp summary",
            "show logging | include BGP",
            "clear ip bgp * soft out",
            "show ip bgp summary",
        ],
    },
    "AUTH_FAILURE": {
        "script_id": "auth_debug",
        "description": "Debug AAA/RADIUS authentication issues",
        "commands": [
            "show aaa servers",
            "show logging | include AUTH",
            "ping <radius-server-ip>",
            "test aaa group radius <user> <password>",
        ],
    },
    "CPU_SPIKE": {
        "script_id": "cpu_investigate",
        "description": "Investigate CPU hog processes and top talkers",
        "commands": [
            "show processes cpu sorted",
            "show processes cpu history",
            "show platform hardware capacity",
            "show interfaces | include input rate|output rate",
        ],
    },
    "CONFIG_ERROR": {
        "script_id": "config_rollback",
        "description": "Rollback to last known good configuration",
        "commands": [
            "show archive",
            "configure replace nvram:startup-config",
            "show running-config | section CHANGES",
        ],
    },
    "NORMAL": {
        "script_id": "no_action",
        "description": "No corrective action required",
        "commands": [
            "show logging last 50",
            "show clock",
        ],
    },
}


def get_script_for_fault(fault_class: str):
    """Return script dict for given fault_class; fall back to no_action."""
    return FAULT_TO_SCRIPT.get(fault_class, FAULT_TO_SCRIPT["NORMAL"])