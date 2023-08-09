import ray


runtime_env = {"pip": ["ec2-metadata"]}

ray.init(runtime_env=runtime_env)

@ray.remote
def get_ec2_instance_id():
    from ec2_metadata import ec2_metadata
    return ec2_metadata.instance_id

def schedule_on_node(node, soft=False):
    return ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=node['NodeID'], soft=soft
    )

def alive(nodes):
    for node in nodes:
        if node['Alive']:
            yield node

alive_nodes = list(alive(ray.nodes()))

ec2_instance_ids = {
    node['NodeID'] : get_ec2_instance_id.options(
        scheduling_strategy=schedule_on_node(node),
    ).remote()
    for node in alive_nodes
}

ec2_instance_ids = {node_id : ray.get(ec2_id) for node_id, ec2_id in ec2_instance_ids.items()}

for alive_node in alive_nodes:
    node_id = alive_node['NodeID']
    node_name = alive_node['NodeName']
    ec2_id = ec2_instance_ids[node_id]
    print(f'{node_name=} {node_id=} {ec2_id=}')
