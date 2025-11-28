import random
import pandas as pd
from datetime import datetime, timedelta

fault_classes = {
    "LINK_DOWN": [
        "%LINEPROTO-5-UPDOWN: Line protocol on Interface Gi0/1, changed state to down",
        "%LINK-3-UPDOWN: Interface Gi0/1 is down",
        "%ETH-4-LINKFLAP: Interface Gi0/1 link flapping detected",
        "%OAM-3-LOST: Loss of signal on port Gi0/1"
    ],
    "BGP_FLAP": [
        "%BGP-5-NBR_RESET: Neighbor 10.1.1.1 reset, closed peer connection",
        "%BGP-4-MSG: BGP notification sent to neighbor 10.1.1.1",
        "%BGP-5-ADJCHANGE: BGP adjacency changed from Established to Idle",
        "%BGP-3-KEEPALIVE: Missing keepalive from neighbor 10.1.1.1"
    ],
    "AUTH_FAILURE": [
        "%AUTH-3-FAIL: Authentication failed for user admin",
        "%RADIUS-4-ERROR: RADIUS server timeout",
        "%AAA-5-NO_ATTR: Missing AAA attributes in request",
        "%SSH-5-FAIL: Failed login attempt for user root"
    ],
    "CPU_SPIKE": [
        "%SYS-3-CPUHOG: Task Scheduler detected CPU hog process",
        "%PLATFORM-3-CPUHIGH: CPU utilization above 95%",
        "%KERN-3-PROCESSBLOCK: Process blocked due to CPU saturation",
        "%SYS-4-RESCHED: High CPU caused delayed process scheduling"
    ],
    "CONFIG_ERROR": [
        "%SYS-5-CONFIG_I: Invalid configuration on interface Gi0/1",
        "%CONFIG-3-ERROR: Failed to apply running-config changes",
        "%PARSER-4-BADCMD: Bad command detected in config file",
        "%SYS-4-CONFIG: Rollback initiated due to config inconsistency"
    ],
    "NORMAL": [
        "%SYS-5-RELOAD: Reload requested by admin",
        "%LINK-5-UPDOWN: Interface Gi0/2 changed state to up",
        "%BGP-5-NBRUP: BGP adjacency established successfully",
        "%SYS-5-LOGGING: System logging restarted"
    ]
}

def generate_logs(n_logs=30000, outfile="data/raw/synthetic_syslogs.csv"):
    rows = []
    start = datetime.now()

    for _ in range(n_logs):
        fault = random.choice(list(fault_classes.keys()))
        msg = random.choice(fault_classes[fault])
        ts = start + timedelta(seconds=random.randint(0, 24*3600))  # 24h instead of 1h
        device = "R" + str(random.randint(1, 10))

        rows.append({
            "timestamp": ts.isoformat(),
            "device": device,
            "message": msg,
            "fault_class": fault
        })

    df = pd.DataFrame(rows)
    df.to_csv(outfile, index=False)
    print(f"[OK] Saved synthetic logs to {outfile}")


if __name__ == "__main__":
    generate_logs()