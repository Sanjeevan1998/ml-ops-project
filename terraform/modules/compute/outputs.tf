output "instance_names" {
  value = [for instance in openstack_compute_instance_v2.nodes : instance.name]
}