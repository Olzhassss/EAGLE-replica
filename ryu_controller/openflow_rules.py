"""Translate EAGLE's forwarding plan into OpenFlow 1.3 messages."""

COOKIE = 0x4541474C45
COOKIE_MASK = (1 << 64) - 1
PRIORITY = 10
TABLE_ID = 0


def request_master(datapath, generation_id):
    ofp = datapath.ofproto
    datapath.send_msg(datapath.ofproto_parser.OFPRoleRequest(
        datapath, ofp.OFPCR_ROLE_MASTER, generation_id
    ))


def request_slave(datapath, generation_id):
    ofp = datapath.ofproto
    datapath.send_msg(datapath.ofproto_parser.OFPRoleRequest(
        datapath, ofp.OFPCR_ROLE_SLAVE, generation_id
    ))


def apply_forwarding_rules(datapath, previous, desired):
    """Send changed rules only. A new owner rebuilds EAGLE's table entries."""
    ofp = datapath.ofproto
    parser = datapath.ofproto_parser
    delete_options = {
        "cookie": COOKIE, "cookie_mask": COOKIE_MASK, "table_id": TABLE_ID,
        "out_port": ofp.OFPP_ANY, "out_group": ofp.OFPG_ANY,
    }
    if previous is None:
        datapath.send_msg(parser.OFPFlowMod(
            datapath, command=ofp.OFPFC_DELETE, match=parser.OFPMatch(), **delete_options
        ))
        previous = {}
    for source, target in sorted(previous.keys() - desired.keys()):
        datapath.send_msg(parser.OFPFlowMod(
            datapath, command=ofp.OFPFC_DELETE_STRICT, priority=PRIORITY,
            match=parser.OFPMatch(eth_src=source, eth_dst=target), **delete_options
        ))
    for (source, target), port in sorted(desired.items()):
        if previous.get((source, target)) == port:
            continue
        actions = [parser.OFPActionOutput(port)]
        datapath.send_msg(parser.OFPFlowMod(
            datapath, cookie=COOKIE, table_id=TABLE_ID, command=ofp.OFPFC_ADD,
            priority=PRIORITY, match=parser.OFPMatch(eth_src=source, eth_dst=target),
            instructions=[parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        ))
