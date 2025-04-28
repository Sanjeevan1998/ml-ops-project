output "private_ports" {
  value = { for k, v in openstack_networking_port_v2.private_net_ports : k => v.id }
}

output "shared_ports" {
  value = { for k, v in openstack_networking_port_v2.sharednet1_ports : k => v.id }
}