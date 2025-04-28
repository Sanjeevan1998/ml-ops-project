output "private_net_id" {
  value = openstack_networking_network_v2.private_net.id
}

output "private_subnet_id" {
  value = openstack_networking_subnet_v2.private_subnet.id
}